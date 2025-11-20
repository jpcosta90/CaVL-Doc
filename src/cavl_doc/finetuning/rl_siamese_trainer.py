# src/finetuning/rl_siamese_trainer.py
import os
import csv
import logging
import math
from tqdm import tqdm
import random
import time

import numpy as np
from sklearn.metrics import roc_curve

import torch
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Subset

# Project imports
from src.models.professor import ProfessorNetwork
from src.finetuning.losses import ContrastiveLoss

# prepare_inputs_for_multimodal_embedding may be in different modules
try:
    from src.utils.embedding_utils import prepare_inputs_for_multimodal_embedding
except Exception:
    try:
        from src.metrics.evaluation import prepare_inputs_for_multimodal_embedding
        logging.warning("Importando 'prepare_inputs_for_multimodal_embedding' de 'metrics/evaluation.py'. Considere movê-lo para 'src/utils'.")
    except Exception:
        raise ImportError("prepare_inputs_for_multimodal_embedding não encontrado em src.utils.embedding_utils nem em src.metrics.evaluation")

# siam builder
from src.models.siamese_internVL import build_siamese_internvl

logger = logging.getLogger(__name__)
EMBEDDING_PROMPT = "<image> Analyze this document"

def rl_full_collate_fn(batch):
    """
    Collate function consistent with DocumentPairDataset returning:
      {"image_a": tensor([N,3,H,W]), "image_b": tensor([N,3,H,W]), "label": float}
    We want lists of per-sample pixel_values so we keep original behavior.
    """
    img_a_list = [item['image_a'] for item in batch]
    img_b_list = [item['image_b'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float32)
    return img_a_list, img_b_list, labels

# ----------------- Validation / EER helpers -----------------
def pairwise_eer_from_scores(labels: np.ndarray, scores: np.ndarray):
    """
    Compute EER given binary labels (1: same, 0: diff) and similarity scores (higher = more similar).
    Returns (eer, threshold).
    """
    if len(labels) == 0:
        return 1.0, 0.0
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    idx = np.argmin(abs_diffs)
    eer = (fpr[idx] + fnr[idx]) / 2.0
    thr = thresholds[idx]
    return eer, thr

def validate_siam_on_loader(siam, val_loader, device, student_criterion):
    """
    Runs siam.eval() on val_loader and computes:
      - mean_student_loss
      - EER using cosine similarity on embeddings
    Expects val_loader to yield batches via rl_full_collate_fn.
    """
    siam.eval()
    all_labels = []
    all_scores = []
    losses = []
    with torch.no_grad():
        for img_a_list, img_b_list, labels in val_loader:
            # process each sample in the batch (safe, compatible with current dataset)
            emb_a_list = []
            emb_b_list = []
            for i in range(len(img_a_list)):
                pv_a = img_a_list[i]
                pv_b = img_b_list[i]
                za = siam(images=pv_a.to(device))
                zb = siam(images=pv_b.to(device))
                emb_a_list.append(za)
                emb_b_list.append(zb)
            za = torch.cat(emb_a_list, dim=0)  # (B, D)
            zb = torch.cat(emb_b_list, dim=0)
            labels = labels.to(device)
            ind_losses = student_criterion.forward_individual(za, zb, labels)
            losses.append(ind_losses.mean().item())
            scores = torch.nn.functional.cosine_similarity(za, zb, dim=-1).cpu().numpy()
            all_scores.append(scores)
            all_labels.append(labels.cpu().numpy())

    if len(losses) == 0:
        return float('nan'), 1.0, 0.0
    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    mean_loss = float(np.mean(losses))
    eer, thr = pairwise_eer_from_scores(all_labels, all_scores)
    return mean_loss, eer, thr

# ----------------- Main Trainer -----------------
def run_rl_siamese_loop(
    base_model,
    student_head,
    professor_model,
    tokenizer,
    dataset,
    epochs,
    student_lr,
    professor_lr,
    device,
    output_dir,
    candidate_pool_size, # K
    student_batch_size,  # B
    max_num_image_tokens,
    cut_layer = 27,
    val_fraction = 0.05,
    val_min_size = 200,
    patience = 3,
    lr_reduce_factor = 0.5,
    baseline_alpha = 0.01,
    entropy_coeff = 0.01,
    seed = 42
):
    """
    RL loop (Siamese tail+head) with:
      - REINFORCE baseline (running mean)
      - advantage normalization
      - entropy bonus
      - validation per epoch (mean student loss + EER)
      - save best checkpoint by EER and early stopping
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print(f"Iniciando loop de RL (Siamese tail+head) — K={candidate_pool_size}, B={student_batch_size}")
    device = torch.device(device if isinstance(device, str) else device)

    # ---------------- build encode_fn ----------------
    def _encode_fn(backbone, pv_tensor, cut_layer: int = cut_layer, **kwargs):
        """
        pv_tensor: [N_patches, 3, H, W] (single doc)
        returns tokens, mask (mask currently unused)
        """
        inputs = prepare_inputs_for_multimodal_embedding(backbone, tokenizer, pv_tensor, EMBEDDING_PROMPT)
        out = backbone(
            input_ids=inputs['input_ids'].to(device),
            attention_mask=inputs['attention_mask'].to(device),
            pixel_values=inputs['pixel_values'].to(device, dtype=torch.bfloat16),
            image_flags=inputs['image_flags'].to(device),
            output_hidden_states=True,
            return_dict=True
        )
        hidden_states = out.hidden_states
        lm = backbone.language_model.model
        # choose index corresponding to cut_layer
        if len(hidden_states) == (len(lm.layers) + 1):
            idx = cut_layer + 1
        else:
            idx = cut_layer
        tokens = hidden_states[idx]  # (1, seq_len, hidden_dim) or (B, seq_len, hidden_dim)
        return tokens, None

    # ---------------- build siamese wrapper ----------------
    siam = build_siamese_internvl(backbone=base_model, cut_layer=cut_layer, encode_fn=_encode_fn,
                                 pool_dim=student_head.ln.normalized_shape[0] if hasattr(student_head, 'ln') else 1536,
                                 proj_hidden=getattr(student_head, 'fc1').out_features if hasattr(student_head, 'fc1') else 1024,
                                 proj_out=getattr(student_head, 'fc2').out_features if hasattr(student_head, 'fc2') else 256,
                                 set_trainable=True)
    siam.to(device)
    # enable expected trainable params (function inside should check dtype)
    siam.set_default_trainable()
    print("Trainable parameters in Siamese wrapper:")
    siam.trainable_summary()

    professor_model.to(device)
    professor_model.train()

    # ------- optimizers (reduced initial lr for stability) -------
    # train params: collect required grads from siam
    trainable_params = [p for n,p in siam.named_parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters detected in siam. Check set_default_trainable().")
    student_optimizer = optim.Adam(trainable_params, lr=student_lr * lr_reduce_factor)
    professor_optimizer = optim.Adam(professor_model.parameters(), lr=professor_lr * lr_reduce_factor)

    student_criterion = ContrastiveLoss(margin=1.0).to(device)

    # ---------------- create train/val splits ----------------
    # If dataset is a Subset, extract underlying dataset and indices
    if isinstance(dataset, Subset):
        full_ds = dataset.dataset
        ds_indices = list(dataset.indices)
    else:
        full_ds = dataset
        ds_indices = list(range(len(full_ds)))

    # deterministic val indices: last val_size of ds_indices
    val_size = min(max(val_min_size, int(len(ds_indices) * val_fraction)), len(ds_indices)//10)
    if val_size <= 0:
        val_size = min(val_min_size, len(ds_indices)//10)
    val_indices = ds_indices[-val_size:]
    train_indices = ds_indices[:-val_size]
    if len(train_indices) == 0:
        # fallback: random split
        random.shuffle(ds_indices)
        val_indices = ds_indices[:val_size]
        train_indices = ds_indices[val_size:]

    train_subset = Subset(full_ds, train_indices)
    val_subset = Subset(full_ds, val_indices)

    train_loader = DataLoader(train_subset, batch_size=candidate_pool_size, shuffle=True, num_workers=4, collate_fn=rl_full_collate_fn)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False, num_workers=2, collate_fn=rl_full_collate_fn)

    # CSV logger
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, "training_log.csv")
    log_file = open(log_file_path, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['epoch', 'batch', 'aluno_loss', 'prof_loss', 'reward', 'baseline', 'entropy', 'adv_std', 'val_mean_loss', 'val_eer'])
    print(f"  -> Salvando log de treino em: {log_file_path}")

    # training bookkeeping
    best_val_eer = 1.0
    no_improve = 0
    global_batch_step = 0
    baseline = 0.0  # running baseline for REINFORCE

    # helper: student forward producing embeddings for lists of pixel_value tensors
    def student_forward_pass(pv_list_a, pv_list_b, train_student=True):
        """
        pv_list_a/pv_list_b: lists of per-sample pixel_values tensors [N_patches,3,H,W]
        returns: emb_a (B, D), emb_b (B, D)
        """
        if train_student:
            siam.train()
        else:
            siam.eval()

        emb_a_list = []
        emb_b_list = []
        # process sequentially to preserve memory and compatibility
        for pv in pv_list_a:
            if train_student:
                z = siam(images=pv.to(device))
            else:
                with torch.no_grad():
                    z = siam(images=pv.to(device))
            emb_a_list.append(z)
        for pv in pv_list_b:
            if train_student:
                z = siam(images=pv.to(device))
            else:
                with torch.no_grad():
                    z = siam(images=pv.to(device))
            emb_b_list.append(z)
        emb_a = torch.cat(emb_a_list, dim=0)
        emb_b = torch.cat(emb_b_list, dim=0)
        return emb_a, emb_b

    print("Iniciando o loop de co-treinamento (Siamese)...")
    for epoch in range(epochs):
        print(f"\n--- Época {epoch + 1}/{epochs} ---")
        pbar = tqdm(train_loader, desc=f"Época {epoch+1}", unit="batch")

        for batch_idx, (img_a_list, img_b_list, labels) in enumerate(pbar):
            labels = labels.to(device).float()

            # --- 1) Professor turn: compute per-candidate difficulty ---
            professor_model.train()
            with torch.no_grad():
                emb_a_all, emb_b_all = student_forward_pass(img_a_list, img_b_list, train_student=False)
                # per-candidate individual losses -> tensor shape (K,)
                state_losses = student_criterion.forward_individual(emb_a_all, emb_b_all, labels.to(device))
                # normalize for stable input to professor
                denom = (state_losses.max() - state_losses.min()).item() if (state_losses.max() - state_losses.min()).item() != 0 else 1.0
                state_losses_norm = (state_losses - state_losses.min()) / (denom + 1e-6)
                state_input = state_losses_norm.unsqueeze(-1)  # (K,1)

            action_logits = professor_model(state_input).squeeze(-1)  # (K,)
            prob_dist = Categorical(logits=action_logits)
            # sample with replacement B indices
            selected_indices = prob_dist.sample((student_batch_size,))
            selected_log_probs = prob_dist.log_prob(selected_indices)  # (B,)

            # --- 2) Student turn: train on selected B pairs ---
            student_indices = selected_indices.tolist()
            student_img_a = [img_a_list[i] for i in student_indices]
            student_img_b = [img_b_list[i] for i in student_indices]
            student_labels = labels[student_indices]

            student_optimizer.zero_grad()
            student_emb_a, student_emb_b = student_forward_pass(student_img_a, student_img_b, train_student=True)
            student_loss = student_criterion(student_emb_a, student_emb_b, student_labels.to(device))
            student_loss.backward()
            student_optimizer.step()

            # per-sample student loss for the selected B (for professor reward)
            with torch.no_grad():
                student_individual = student_criterion.forward_individual(student_emb_a.detach(), student_emb_b.detach(), student_labels.to(device))
                # student_individual: tensor (B,)

            # --- 3) Professor update: REINFORCE with baseline + advantage normalization + entropy bonus ---
            rewards = student_individual.detach()  # (B,)
            avg_reward = float(rewards.mean().item())

            # advantage and normalization
            advantage = rewards - baseline  # (B,)
            adv_mean = advantage.mean()
            adv_std = advantage.std(unbiased=False).clamp(min=1e-6)
            advantage_norm = (advantage - adv_mean) / (adv_std + 1e-6)  # (B,)

            professor_optimizer.zero_grad()
            prof_loss_tensor = - (selected_log_probs * advantage_norm).mean()
            entropy = prob_dist.entropy().mean()
            prof_loss = prof_loss_tensor - entropy_coeff * entropy
            prof_loss.backward()
            professor_optimizer.step()

            # update baseline (running scalar)
            baseline = (1 - baseline_alpha) * baseline + baseline_alpha * avg_reward

            # Logging
            global_batch_step += 1
            pbar.set_postfix({
                'Aluno_Loss': f"{student_loss.item():.6f}",
                'Prof_Loss': f"{prof_loss.item():.6f}",
                'Reward': f"{avg_reward:.6f}",
                'Entropy': f"{float(entropy.item()):.6f}",
                'AdvStd': f"{float(adv_std.item()):.6f}"
            })

            # write CSV (val metrics filled at epoch end)
            log_writer.writerow([
                epoch + 1,
                global_batch_step,
                f"{student_loss.item():.6f}",
                f"{prof_loss.item():.6f}",
                f"{avg_reward:.6f}",
                f"{baseline:.6f}",
                f"{float(entropy.item()):.6f}",
                f"{float(adv_std.item()):.6f}",
                "", ""
            ])

        pbar.close()

        # ---------- End of epoch: validation ----------
        val_mean_loss, val_eer, val_thr = validate_siam_on_loader(siam, val_loader, device, student_criterion)
        print(f"Validation — mean_loss: {val_mean_loss:.6f}, EER: {100*val_eer:.3f}% (thr={val_thr:.4f})")

        # update CSV last written rows with val metrics (simple: append a row for epoch summary)
        log_writer.writerow([
            epoch + 1, "epoch_end", "", "", "", "", "", "", f"{val_mean_loss:.6f}", f"{val_eer:.6f}"
        ])

        # checkpoint/save logic (save best by EER)
        if val_eer < best_val_eer:
            best_val_eer = val_eer
            no_improve = 0
            # save full siam weights (pool + head + any unfrozen tail params)
            torch.save({
                'siam_pool': siam.pool.state_dict(),
                'siam_head': siam.head.state_dict(),
                'tail_layer_params': {n: p.detach().cpu() for n, p in siam.backbone.language_model.model.named_parameters() if f"layers.{cut_layer}" in n and p.requires_grad}
            }, os.path.join(output_dir, "best_siam.pt"))
            print("[CHECKPOINT] Saved new best siam (EER improved).")
        else:
            no_improve += 1
            print(f"[INFO] No improvement in EER (no_improve={no_improve}/{patience}).")

        # epoch-end generic checkpoints (keeps history)
        torch.save(siam.pool.state_dict(), os.path.join(output_dir, f"student_pool_epoch_{epoch+1}.pt"))
        torch.save(siam.head.state_dict(), os.path.join(output_dir, f"student_head_epoch_{epoch+1}.pt"))
        tail_sd = {}
        layer_sub = f"language_model.model.layers.{cut_layer}"
        for n, p in siam.backbone.language_model.model.named_parameters():
            if layer_sub in n and p.requires_grad:
                tail_sd[n] = p.detach().cpu()
        torch.save(tail_sd, os.path.join(output_dir, f"tail_layer_params_epoch_{epoch+1}.pt"))
        torch.save(professor_model.state_dict(), os.path.join(output_dir, f"professor_epoch_{epoch+1}.pt"))

        # Early stopping
        if no_improve >= patience:
            print(f"Early stopping triggered (no_improve={no_improve} >= patience={patience}).")
            break

    # close log
    log_file.close()
    print("\n✅ Treinamento de Currículo de RL (Siamese tail+head) concluído.")
