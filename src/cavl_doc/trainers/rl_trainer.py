# src/cavl_doc/trainers/rl_trainer.py
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

# Imports do Projeto
from cavl_doc.models.policy import ProfessorNetwork
from cavl_doc.modules.losses import build_loss
from cavl_doc.data.dataset import DocumentPairDataset
from cavl_doc.models.modeling_cavl import build_cavl_model

# Importa métrica de k-NN para validação robusta
try:
    from cavl_doc.evaluation.metrics import compute_knn_metrics
except ImportError:
    # Fallback simples se a função não existir ainda
    def compute_knn_metrics(*args, **kwargs): return {'R@1': 0.0}

try:
    import wandb
except ImportError:
    wandb = None

try:
    from cavl_doc.utils.embedding_utils import prepare_inputs_for_multimodal_embedding
except ImportError:
    try:
        from cavl_doc.metrics.evaluation import prepare_inputs_for_multimodal_embedding
    except ImportError:
        raise ImportError("prepare_inputs_for_multimodal_embedding não encontrado.")

logger = logging.getLogger(__name__)
EMBEDDING_PROMPT = "<image> Analyze this document"

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count

def rl_full_collate_fn(batch):
    img_a_list = [item['image_a'] for item in batch]
    img_b_list = [item['image_b'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float32)
    
    if 'class_a' in batch[0]:
        class_a = torch.tensor([item['class_a'] for item in batch], dtype=torch.long)
        class_b = torch.tensor([item['class_b'] for item in batch], dtype=torch.long)
    else:
        class_a = torch.zeros(len(batch), dtype=torch.long)
        class_b = torch.zeros(len(batch), dtype=torch.long)
        
    return img_a_list, img_b_list, labels, class_a, class_b

def pairwise_eer_from_scores(labels: np.ndarray, scores: np.ndarray):
    if len(labels) == 0: return 1.0, 0.0
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2.0, thresholds[idx]

def validate_siam_on_loader(siam, val_loader, device, student_criterion):
    siam.eval()
    losses = []
    all_scores = []
    all_labels = []
    
    # Listas para k-NN
    knn_embeds = []
    knn_classes = []

    with torch.no_grad():
        for img_a_list, img_b_list, labels, cls_a, cls_b in val_loader:
            emb_a_list = []
            emb_b_list = []
            
            for i in range(len(img_a_list)):
                pv_a = img_a_list[i].unsqueeze(0).to(device)
                pv_b = img_b_list[i].unsqueeze(0).to(device)
                
                # Inferência
                try:
                    za = siam(images=pv_a)
                    zb = siam(images=pv_b)
                except Exception:
                    # Fallback legado
                    if hasattr(siam, '_extract_tokens_via_encode_fn'):
                        tok_a, m_a = siam._extract_tokens_via_encode_fn(pv_a, device=device)
                        tok_b, m_b = siam._extract_tokens_via_encode_fn(pv_b, device=device)
                        za = siam.head(siam.pool(tok_a, m_a))
                        zb = siam.head(siam.pool(tok_b, m_b))
                    else: raise
                
                emb_a_list.append(za)
                emb_b_list.append(zb)
            
            za = torch.cat(emb_a_list, dim=0)
            zb = torch.cat(emb_b_list, dim=0)
            labels = labels.to(device)
            
            # --- [DIAGNÓSTICO DA LOSS] ---
            # Tenta calcular loss original, se falhar (ArcFace), usa proxy de cosseno
            try:
                ind_losses = student_criterion.forward_individual(za, zb, labels)
                losses.append(ind_losses.mean().item())
            except:
                cos_sim = torch.nn.functional.cosine_similarity(za, zb)
                dummy_loss = (labels * (1 - cos_sim) + (1 - labels) * torch.relu(cos_sim)).mean()
                losses.append(dummy_loss.item())
            # -----------------------------

            scores = torch.nn.functional.cosine_similarity(za, zb, dim=-1).cpu().numpy()
            all_scores.append(scores)
            all_labels.append(labels.cpu().numpy())

            # Acumula k-NN
            knn_embeds.append(za.cpu().numpy())
            knn_embeds.append(zb.cpu().numpy())
            knn_classes.append(cls_a.cpu().numpy())
            knn_classes.append(cls_b.cpu().numpy())

    if len(losses) == 0: return float('nan'), 1.0, 0.0, 0.0
    
    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    mean_loss = float(np.mean(losses))
    eer, thr = pairwise_eer_from_scores(all_labels, all_scores)

    # --- [DIAGNÓSTICO DO K-NN] ---
    r1 = 0.0
    try:
        full_embeds = np.vstack(knn_embeds)
        full_classes = np.concatenate(knn_classes, axis=0)
        
        mask = full_classes != -1
        
        if mask.sum() > 1:
            metrics = compute_knn_metrics(full_embeds[mask], full_classes[mask], k_vals=[1])
            r1 = metrics.get('R@1', 0.0)
        else:
            # Se não houver classes válidas, R@1 fica 0.0
            pass
            
    except Exception as e:
        print(f"\n[DEBUG k-NN] ❌ Erro no cálculo: {e}")

    return mean_loss, eer, thr, r1

def run_rl_siamese_loop(
    base_model, student_head, professor_model, tokenizer, dataset, epochs, student_lr, professor_lr,
    device, output_dir, candidate_pool_size, student_batch_size, max_num_image_tokens,
    val_csv_path=None, base_image_dir=None,
    cut_layer=27, projection_output_dim=512, val_fraction=0.05, val_min_size=200,
    patience=3, lr_reduce_factor=0.5, baseline_alpha=0.01, entropy_coeff=0.01, seed=42,
    use_wandb=False,
    loss_type="contrastive", pooler_type="attention", head_type="mlp", num_classes=None, num_queries=1,
    # Novos argumentos para losses
    margin=0.5, scale=64.0, num_sub_centers=3, std=0.05
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device(device if isinstance(device, str) else device)

    model_config = {
        'cut_layer': cut_layer,
        'projection_output_dim': projection_output_dim,
        'max_num_image_tokens': max_num_image_tokens,
        'hidden_dim': 1536,
        'loss_type': loss_type,
        'head_type': head_type,
        'pooler_type': pooler_type,
        'num_queries': num_queries,
        # Use as variáveis diretamente, pois elas são argumentos da função
        'margin': margin,  
        'scale': scale,
        'num_sub_centers': num_sub_centers,
        'std': std
    }

    # Define encode_fn (COM CORREÇÃO DE FLATTENING 5D)
    def _encode_fn(backbone, pv_tensor, cut_layer=cut_layer, **kwargs):
        # --- CORREÇÃO CRÍTICA ---
        if pv_tensor.dim() == 5:
            b, n, c, h, w = pv_tensor.shape
            pv_tensor = pv_tensor.view(b * n, c, h, w)
        # ------------------------

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
        idx = cut_layer + 1 if len(hidden_states) == (len(lm.layers) + 1) else cut_layer
        return hidden_states[idx], None

    siam = build_cavl_model(
        backbone=base_model, 
        cut_layer=cut_layer, 
        encode_fn=_encode_fn,
        pool_dim=student_head.ln.normalized_shape[0] if hasattr(student_head, 'ln') else 1536,
        proj_hidden=getattr(student_head, 'fc1').out_features if hasattr(student_head, 'fc1') else 4096,
        proj_out=projection_output_dim,
        set_trainable=True,
        tokenizer=tokenizer,
        pooler_type=pooler_type,
        head_type=head_type,
        num_queries=num_queries
    )
    siam.to(device)
    siam.set_default_trainable()
    
    professor_model.to(device).train()

    # Loss Builder com novos parâmetros
    print(f"Construindo Loss: {loss_type} (m={margin}, s={scale}, k={num_sub_centers}, std={std})")
    student_criterion = build_loss(
        loss_type, 
        margin=margin, 
        m=margin, # ArcFace/CosFace/Circle usam 'm'
        s=scale,
        gamma=scale, # Circle usa gamma
        num_classes=num_classes, 
        k=num_sub_centers,
        std=std,
        in_features=projection_output_dim
    ).to(device)

    trainable_params = [p for n,p in siam.named_parameters() if p.requires_grad]
    loss_params = list(student_criterion.parameters())
    
    student_optimizer = optim.Adam(trainable_params + loss_params, lr=student_lr * lr_reduce_factor)
    professor_optimizer = optim.Adam(professor_model.parameters(), lr=professor_lr * lr_reduce_factor)

    if val_csv_path and os.path.exists(val_csv_path):
        print(f"Carregando validação: {val_csv_path}")
        val_dataset = DocumentPairDataset(val_csv_path, base_image_dir, 448, max_num_image_tokens, 'cpu')
        train_dataset = dataset
    else:
        print("Split automático treino/val.")
        if isinstance(dataset, Subset): full_ds, ds_indices = dataset.dataset, list(dataset.indices)
        else: full_ds, ds_indices = dataset, list(range(len(dataset)))
        val_size = min(max(val_min_size, int(len(ds_indices) * val_fraction)), len(ds_indices)//10)
        if val_size <= 0: val_size = min(val_min_size, len(ds_indices)//10)
        val_indices = ds_indices[-val_size:]; train_indices = ds_indices[:-val_size]
        if not train_indices:
            random.shuffle(ds_indices); val_indices = ds_indices[:val_size]; train_indices = ds_indices[val_size:]
        train_dataset = Subset(full_ds, train_indices)
        val_dataset = Subset(full_ds, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=candidate_pool_size, shuffle=True, num_workers=4, collate_fn=rl_full_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, collate_fn=rl_full_collate_fn)

    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, "training_log.csv")
    log_file = open(log_file_path, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['epoch', 'batch', 'aluno_loss', 'prof_loss', 'reward', 'baseline', 'entropy', 'adv_std', 'val_mean_loss', 'val_eer'])

    best_val_eer = 1.0
    no_improve = 0
    global_batch_step = 0
    baseline = 0.0

    def student_forward_pass(pv_list_a, pv_list_b, train_student=True):
        if train_student: siam.train()
        else: siam.eval()
        emb_a_list, emb_b_list = [], []
        for pv in pv_list_a:
            pv = pv.unsqueeze(0).to(device)
            if train_student: z = siam(images=pv)
            else:
                with torch.no_grad(): z = siam(images=pv)
            emb_a_list.append(z)
        for pv in pv_list_b:
            pv = pv.unsqueeze(0).to(device)
            if train_student: z = siam(images=pv)
            else:
                with torch.no_grad(): z = siam(images=pv)
            emb_b_list.append(z)
        return torch.cat(emb_a_list, dim=0), torch.cat(emb_b_list, dim=0)

    print("Iniciando treinamento...")
    for epoch in range(epochs):
        print(f"\n--- Época {epoch + 1}/{epochs} ---")
        avg_loss = AverageMeter(); avg_rew = AverageMeter(); avg_prof_loss = AverageMeter()
        
        # Update lento para não travar UI
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}", mininterval=30.0, ncols=100)

        for batch_idx, (img_a, img_b, labels, cls_a, cls_b) in enumerate(pbar):
            labels = labels.to(device).float()
            cls_a = cls_a.to(device)
            cls_b = cls_b.to(device)

            # 1. Professor
            professor_model.train()
            with torch.no_grad():
                ea, eb = student_forward_pass(img_a, img_b, False)
                
                if loss_type in ['contrastive', 'angular']:
                    sl = student_criterion.forward_individual(ea, eb, labels)
                else:
                    # Para ArcFace/CosFace, calculamos a loss individual de cada braço e tiramos a média
                    loss_a = student_criterion.forward_individual(ea, cls_a)
                    loss_b = student_criterion.forward_individual(eb, cls_b)
                    sl = (loss_a + loss_b) / 2.0

                denom = (sl.max() - sl.min()).item() or 1.0
                sl_norm = (sl - sl.min()) / (denom + 1e-6)
                state = sl_norm.unsqueeze(-1)
            
            logits = professor_model(state).squeeze(-1)
            dist = Categorical(logits=logits)
            idxs = dist.sample((student_batch_size,))
            log_probs = dist.log_prob(idxs)

            # 2. Student
            sel_idxs = idxs.tolist()
            sa = [img_a[i] for i in sel_idxs]; sb = [img_b[i] for i in sel_idxs]
            slbs = labels[sel_idxs]
            s_cls_a = cls_a[sel_idxs]; s_cls_b = cls_b[sel_idxs]
            
            student_optimizer.zero_grad()
            sea, seb = student_forward_pass(sa, sb, True)
            
            if loss_type in ['contrastive', 'angular']:
                loss = student_criterion(sea, seb, slbs)
            else:
                loss = student_criterion(sea, s_cls_a) + student_criterion(seb, s_cls_b)
            
            if torch.isnan(loss): continue
            loss.backward()
            student_optimizer.step()

            # 3. Update Prof
            with torch.no_grad():
                if loss_type in ['contrastive', 'angular']:
                    s_ind = student_criterion.forward_individual(sea.detach(), seb.detach(), slbs)
                else:
                    l_a = student_criterion.forward_individual(sea.detach(), s_cls_a)
                    l_b = student_criterion.forward_individual(seb.detach(), s_cls_b)
                    s_ind = (l_a + l_b) / 2.0

            rew = s_ind.detach()
            avg_r = float(rew.mean().item())
            adv = rew - baseline
            adv_std_val = adv.std(unbiased=False).clamp(min=1e-6)
            adv_norm = (adv - adv.mean()) / (adv_std_val + 1e-6)
            
            professor_optimizer.zero_grad()
            ploss = - (log_probs * adv_norm).mean() - entropy_coeff * dist.entropy().mean()
            ploss.backward()
            professor_optimizer.step()

            baseline = (1 - baseline_alpha) * baseline + baseline_alpha * avg_r
            
            global_batch_step += 1
            avg_loss.update(loss.item()); avg_rew.update(avg_r); avg_prof_loss.update(ploss.item())
            pbar.set_postfix({'L': f"{avg_loss.avg:.4f}", 'R': f"{avg_rew.avg:.4f}"})
            
            if use_wandb and wandb:
                wandb.log({
                    "train/loss": loss.item(), "train/reward": avg_r, "train/prof_loss": ploss.item(),
                    "train/baseline": baseline, "train/entropy": dist.entropy().mean().item(),
                    "step": global_batch_step
                })

            log_writer.writerow([epoch+1, global_batch_step, f"{loss.item():.4f}", f"{ploss.item():.4f}", f"{avg_r:.4f}", "", "", "", "", ""])

        pbar.close()
        vloss, veer, vthr, vr1 = validate_siam_on_loader(siam, val_loader, device, student_criterion)
        print(f"Val: EER={veer*100:.2f}% | R@1={vr1*100:.2f}% | Loss={vloss:.4f}")
        
        if use_wandb and wandb:
            wandb.log({"val/eer": veer, "val/loss": vloss, "val/recall_at_1": vr1, "epoch": epoch + 1})

        log_writer.writerow([epoch+1, "end", "", "", "", "", "", "", f"{vloss:.4f}", f"{veer:.4f}"])

        if veer < best_val_eer:
            best_val_eer = veer
            no_improve = 0
            backbone_trainable = {n: p.detach().cpu() for n, p in siam.backbone.named_parameters() if p.requires_grad}
            ckpt = {
                'epoch': epoch, 'metrics': {'eer': veer, 'loss': vloss}, 'config': model_config,
                'siam_pool': siam.pool.state_dict(), 'siam_head': siam.head.state_dict(),
                'backbone_trainable': backbone_trainable, 'professor_state': professor_model.state_dict()
            }
            torch.save(ckpt, os.path.join(output_dir, "best_siam.pt"))
            print("Saved best_siam.pt")
            if use_wandb and wandb: wandb.log({"val/best_eer": best_val_eer})
        else:
            no_improve += 1
        
        if no_improve >= patience: break

    log_file.close()