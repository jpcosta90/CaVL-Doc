#!/usr/bin/env python3
"""
Fine-tunes jinaai/jina-embeddings-v4 com novo LoRA de ~59M parâmetros
especializado em recuperação de documentos visuais no LA-CDIP (protocolo ZSL).

Segue o método de "jina-embeddings-v4: Universal Embeddings for Multimodal
Multilingual Retrieval" (Günther et al., 2025):
  - Backbone Qwen2.5-VL-3B-Instruct frozen (3.8B params)
  - Novo LoRA: r=128, módulos q/k/v/o_proj do decoder LLM → ~59M params
  - Fase 1 (pair training): InfoNCE simétrico em pares doc-doc (Eq. 4 do paper)
  - Fase 2 (task-specific): InfoNCE+ com hard negatives in-batch (Eq. 7)
  - Loss conjunta: single-vector + Matryoshka truncation (Seção 5 do paper)

Benchmark LA-CDIP (estilo Jina-VDR, Seção 6 do paper):
  - Protocolo ZSL: 5 splits (seen/unseen classes)
  - Query e galeria são imagens de documentos (retrieval simétrico)
  - Métricas: EER, R@1, R@5, mAP@10, NDCG@10

Diferença dos outros baselines:
  eval_vlm_prompt.py      → zero-shot VLM com prompt de similaridade
  train_lora_baseline.py  → SFT generativo (cross-entropy em tokens JSON)
  train_jina_v4_lacdip.py → Contrastive embedding learning (InfoNCE) ← este

Uso (todos os 5 splits — padrão recomendado):
  python scripts/baselines/train_jina_v4_lacdip.py \
      --data-root /mnt/nas/joaopaulo/LA-CDIP \
      --base-image-dir /mnt/nas/joaopaulo/LA-CDIP/data \
      --epochs 5 --gpu-id 0

  # Um split específico:
  python scripts/baselines/train_jina_v4_lacdip.py \
      --data-root /mnt/nas/joaopaulo/LA-CDIP \
      --base-image-dir /mnt/nas/joaopaulo/LA-CDIP/data \
      --split 0 --epochs 5 --gpu-id 0

Nota sobre épocas: o paper não especifica número de épocas para o fine-tuning
task-specific. 5 épocas é um ponto de partida razoável; ajustar conforme a curva
de validação (EER) no W&B.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve, ndcg_score, average_precision_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PREP_SCRIPT    = WORKSPACE_ROOT / "scripts" / "utils" / "prepare_protocol_split.py"

WANDB_ENTITY  = "jpcosta1990-university-of-brasilia"
WANDB_PROJECT = "CaVL-Doc_LA-CDIP_JinaV4_LoRA"

MODEL_HF_ID   = "jinaai/jina-embeddings-v4"
EMBEDDING_DIM = 2048   # single-vector output do Jina-v4
TEMPERATURE   = 0.02   # τ do paper (Seção 5.2.3)

# Módulos do decoder Qwen2.5-VL-3B alvo para LoRA
# r=16 → ~22M parâmetros (cabe em 24 GB L4)
# r=128 ultrapassa o L4 pois add_adapter também cobre os projectors do Jina (~176M)
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Dimensões para Matryoshka loss (paper Seção 5, truncatable to 128)
MATRYOSHKA_DIMS = [2048, 1024, 512, 256, 128]

# Template de texto para imagens — formato interno do Jina-v4 (process_images).
# O task_label seleciona o adapter LoRA; o texto acompanha a imagem no VLM.
JINA_IMG_TEXT = (
    "<|im_start|>user\n"
    "<|vision_start|><|image_pad|><|vision_end|>"
    "Represent this document image for retrieval."
    "<|im_end|>\n"
)


# ---------------------------------------------------------------------------
# Split preparation
# ---------------------------------------------------------------------------

def _prepare_split(data_root: str, split_idx: int) -> Tuple[Path, Path]:
    out_dir   = WORKSPACE_ROOT / "data" / "generated_splits" / f"split{split_idx}"
    train_csv = out_dir / "train_pairs.csv"
    val_csv   = out_dir / "validation_pairs.csv"
    if train_csv.exists() and val_csv.exists():
        return train_csv, val_csv
    cmd = [
        sys.executable, str(PREP_SCRIPT),
        "--data-root",            data_root,
        "--output-dir",           str(out_dir),
        "--val-split-idx",        str(split_idx),
        "--protocol",             "zsl",
        "--exclude-train-splits", "5",
    ]
    print(f"[PREP] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return train_csv, val_csv


# ---------------------------------------------------------------------------
# Model: Jina-v4 + novo LoRA domain-specific
# ---------------------------------------------------------------------------

def _count_lora_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _make_model(lora_r: int, lora_dropout: float, load_in_4bit: bool = False):
    """
    Carrega jinaai/jina-embeddings-v4 (backbone frozen) e adiciona novo
    LoRA domain-specific seguindo a Seção 4.3 do paper.

    Backbone: Qwen2.5-VL-3B-Instruct (frozen, 3.8B params)
    Adapters existentes do Jina-v4 (retrieval/text-matching/code) também frozen.
    Novo LoRA: treinável, ~59M params com r=128.
    """
    from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model

    print(f"Carregando {MODEL_HF_ID}...")
    load_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    base = AutoModel.from_pretrained(MODEL_HF_ID, **load_kwargs)

    # Congela todos os parâmetros existentes (backbone + adapters Jina-v4)
    for param in base.parameters():
        param.requires_grad = False

    # Novo LoRA domain-specific (Seção 4.3 do paper)
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,       # alpha = 2r (padrão do paper)
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )

    # Jina-v4 já é um PeftModel — usa add_adapter() para adicionar o nosso
    # sem empacotar novamente (o que duplicaria os adapters e parâmetros)
    if hasattr(base, "peft_config"):
        base.add_adapter("lacdip", lora_cfg)
        base.set_adapter("lacdip")
        model = base
    else:
        model = get_peft_model(base, lora_cfg)

    n_trainable = _count_lora_params(model)
    print(f"LoRA adicionado: {n_trainable / 1e6:.1f}M parâmetros treináveis "
          f"(r={lora_r}, target={LORA_TARGET_MODULES})")

    processor = AutoProcessor.from_pretrained(MODEL_HF_ID, trust_remote_code=True)
    return model, processor


# ---------------------------------------------------------------------------
# Embedding extraction (trainable — gradiente flui pelo LoRA)
# ---------------------------------------------------------------------------

# task_label para retrieval simétrico (texto/imagem → image)
# 'text-matching' = symmetric similarity (paper Sec 5.2.2)
# 'retrieval'     = asymmetric query→document (paper Sec 5.2.1)
JINA_TASK_LABEL = "text-matching"


def _embed(model, inputs: dict) -> torch.Tensor:
    """
    Extrai single-vector embeddings via o pipeline nativo do Jina-v4.

    O modelo retorna JinaEmbeddingsV4ModelOutput com single_vec_emb
    já mean-pooled, projetado e L2-normalizado (2048-dim).
    O gradiente flui pelo nosso LoRA "lacdip" adicionado sobre o backbone.

    task_label seleciona qual dos adapters internos do Jina-v4 é ativado
    em conjunto com o nosso. Usamos 'text-matching' (simétrico) pois
    query e documento são ambos imagens de documentos.
    """
    outputs = model(task_label=JINA_TASK_LABEL, **inputs)
    # single_vec_emb: [B, 2048], já normalizado pelo modelo
    return outputs.single_vec_emb


# ---------------------------------------------------------------------------
# Loss: InfoNCE simétrico (Eq. 4) + Matryoshka (Seção 5)
# ---------------------------------------------------------------------------

class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE para pares (a_i, b_i).
    Pares i=j são positivos; i≠j são negativos in-batch.
    Temperatura τ fixa (paper usa 0.02).
    """

    def __init__(self, temperature: float = TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        # S[i,j] = cosine(a_i, b_j) / τ
        S = torch.mm(emb_a, emb_b.T) / self.temperature  # [N, N]
        targets = torch.arange(S.shape[0], device=S.device)
        loss = (F.cross_entropy(S, targets) + F.cross_entropy(S.T, targets)) / 2
        return loss


class MatryoshkaInfoNCELoss(nn.Module):
    """
    InfoNCE com Matryoshka Representation Learning (Seção 5 do paper).
    Aplica InfoNCE em múltiplas dimensões de truncação; pesa por 1/log(d).
    """

    def __init__(self, dims: List[int] = MATRYOSHKA_DIMS,
                 temperature: float = TEMPERATURE):
        super().__init__()
        self.dims = dims
        self.base = InfoNCELoss(temperature)

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        total, weight_sum = 0.0, 0.0
        for d in self.dims:
            if d > emb_a.shape[-1]:
                continue
            a_d = F.normalize(emb_a[:, :d], dim=-1)
            b_d = F.normalize(emb_b[:, :d], dim=-1)
            w = 1.0 / (d ** 0.5)   # peso inversamente proporcional à raiz da dim
            total      += w * self.base(a_d, b_d)
            weight_sum += w
        return total / (weight_sum + 1e-8)


class InfoNCEPlusLoss(nn.Module):
    """
    InfoNCE+ com hard negatives explícitos (Eq. 7 do paper, Seção 5.2.1).
    Usado na Fase 2 (task-specific). Hard negatives são amostrados in-batch
    como os exemplos negativos com maior similaridade (mining simples).
    """

    def __init__(self, temperature: float = TEMPERATURE, n_hard: int = 3):
        super().__init__()
        self.temperature = temperature
        self.n_hard = n_hard

    def forward(self, emb_q: torch.Tensor, emb_p: torch.Tensor) -> torch.Tensor:
        τ = self.temperature
        S = torch.mm(emb_q, emb_p.T) / τ   # [N, N]
        N = S.shape[0]

        losses = []
        for i in range(N):
            s_pos = S[i, i]
            # Negatives = todos os outros positivos do batch (in-batch negatives)
            neg_mask = torch.ones(N, dtype=torch.bool, device=S.device)
            neg_mask[i] = False
            s_neg = S[i][neg_mask]

            # Hard negatives: os top-k negativos mais similares (Eq. 7)
            if self.n_hard > 0 and len(s_neg) > self.n_hard:
                hard_idx = s_neg.topk(self.n_hard).indices
                s_neg = s_neg[hard_idx]

            denom = torch.cat([s_pos.unsqueeze(0), s_neg]).exp().sum()
            losses.append(-s_pos + denom.log())

        return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Dataset: pares de imagens de documentos
# ---------------------------------------------------------------------------

class _JinaDocDataset(Dataset):
    """
    Pares (img_a, img_b, is_equal) do LA-CDIP para contrastive learning.
    Cada item retorna as imagens PIL sem preprocessamento — feito no collate.
    """

    def __init__(self, csv_path: Path, base_dir: str,
                 max_pairs: Optional[int] = None):
        import pandas as pd
        self.df       = pd.read_csv(csv_path)
        self.base_dir = base_dir

        if max_pairs and len(self.df) > max_pairs:
            pos = self.df[self.df["is_equal"] == 1].sample(
                n=min(max_pairs // 2, int((self.df["is_equal"] == 1).sum())),
                random_state=42,
            )
            neg = self.df[self.df["is_equal"] == 0].sample(
                n=min(max_pairs - len(pos), int((self.df["is_equal"] == 0).sum())),
                random_state=42,
            )
            self.df = pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"  Dataset: {len(self.df)} pares  "
              f"(pos={int(self.df['is_equal'].sum())}, "
              f"neg={int((self.df['is_equal'] == 0).sum())})")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        from PIL import Image as PILImage
        row = self.df.iloc[idx]

        def _load(rel_path):
            p = os.path.join(self.base_dir, str(rel_path))
            return PILImage.open(p).convert("RGB")

        return {
            "img_a":    _load(row["file_a_path"]),
            "img_b":    _load(row["file_b_path"]),
            "is_equal": int(row["is_equal"]),
        }


def _proc_single(img, processor, device):
    """
    Processa uma única imagem PIL no formato esperado pelo Jina-v4.

    O Jina-v4 espera pixel_values como [num_images, max_patches, features]
    (stacked + padded), não o formato flat [total_patches, features] do
    processor Qwen2.5-VL padrão. Fazemos a conversão manualmente, seguindo
    o que processor.process_images() faz internamente (linhas 65-88 do modelo).
    """
    out = processor(images=[img], text=[JINA_IMG_TEXT],
                    padding="longest", return_tensors="pt")
    # Converte pixel_values para formato [num_images, max_patches, features]
    if "pixel_values" in out and "image_grid_thw" in out:
        offsets = out["image_grid_thw"][:, 1] * out["image_grid_thw"][:, 2]
        pvs = torch.split(out["pixel_values"], offsets.tolist())
        max_len = max(len(pv) for pv in pvs)
        pvs_padded = [
            torch.cat([pv, torch.zeros(max_len - len(pv), pv.shape[1],
                                       dtype=pv.dtype)]) if len(pv) < max_len else pv
            for pv in pvs
        ]
        out["pixel_values"] = torch.stack(pvs_padded)  # [num_images, max_patches, features]
    return {k: v.to(device) for k, v in out.items() if isinstance(v, torch.Tensor)}


def _collate_jina(batch, processor, device="cuda"):
    """
    Qwen2.5-VL usa pixel_values flat (não batched), então processa cada
    imagem individualmente e retorna listas de inputs. O treino embeda
    cada item e concatena os embeddings para a loss InfoNCE.
    """
    labels = torch.tensor([item["is_equal"] for item in batch], dtype=torch.float32)
    list_a = [_proc_single(item["img_a"], processor, device) for item in batch]
    list_b = [_proc_single(item["img_b"], processor, device) for item in batch]
    return list_a, list_b, labels.to(device)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train_epoch(
    model, optimizer, loader, loss_fn,
    device, accum_steps, use_matryoshka, max_steps=None,
) -> float:
    model.train()
    losses = []
    optimizer.zero_grad()

    for step, (list_a, list_b, labels) in enumerate(
        tqdm(loader, desc="Train", ncols=100, leave=False,
             total=min(len(loader), max_steps) if max_steps else len(loader))
    ):
        if max_steps and step >= max_steps:
            break
        try:
            emb_a = torch.cat([_embed(model, inp) for inp in list_a], dim=0)
            emb_b = torch.cat([_embed(model, inp) for inp in list_b], dim=0)

            if use_matryoshka:
                loss = loss_fn(emb_a, emb_b)
            else:
                loss = loss_fn(emb_a, emb_b)

            (loss / accum_steps).backward()
            losses.append(loss.item())
        except Exception as e:
            print(f"\n  [WARN] step {step} falhou: {e}")
            optimizer.zero_grad()
            continue

        if (step + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            optimizer.zero_grad()

    # Flush último acúmulo parcial
    if losses:
        optimizer.step()
        optimizer.zero_grad()

    return float(np.mean(losses)) if losses else float("nan")


# ---------------------------------------------------------------------------
# Benchmark LA-CDIP (estilo Jina-VDR)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _build_gallery(
    model, processor, csv_path: Path, base_dir: str,
    device: str, max_items: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extrai embeddings de todos os documentos do split de validação.
    Retorna (embeddings [N, D], labels [N]).
    """
    import pandas as pd
    from PIL import Image as PILImage

    df = pd.read_csv(csv_path)

    # Pega imagens únicas (file_a_path) com suas classes
    col_img = "file_a_path" if "file_a_path" in df.columns else df.columns[0]
    col_cls = "class_a_name" if "class_a_name" in df.columns else None

    unique = df[[col_img] + ([col_cls] if col_cls else [])].drop_duplicates(subset=col_img)
    if len(unique) > max_items:
        unique = unique.sample(n=max_items, random_state=42)

    all_embs, all_labels = [], []
    label_to_int = {}

    for _, row in tqdm(unique.iterrows(), total=len(unique),
                       desc="Gallery", ncols=100, leave=False):
        try:
            img_path = os.path.join(base_dir, str(row[col_img]))
            img = PILImage.open(img_path).convert("RGB")
            inputs = _proc_single(img, processor, device)
            emb = _embed(model, inputs).cpu().float().numpy()
            all_embs.append(emb[0])

            if col_cls:
                cls = str(row[col_cls])
                if cls not in label_to_int:
                    label_to_int[cls] = len(label_to_int)
                all_labels.append(label_to_int[cls])
            else:
                all_labels.append(0)

        except Exception:
            continue

    return np.array(all_embs, dtype=np.float32), np.array(all_labels, dtype=np.int64)


@torch.no_grad()
def benchmark_lacdip(
    model, processor, val_csv: Path, base_dir: str,
    device: str, max_gallery: int = 2000, max_queries: int = 500,
) -> dict:
    """
    Benchmark de recuperação de documentos no estilo Jina-VDR (Seção 6).

    - Galeria: todas as imagens do split de validação (unseen classes)
    - Query: subconjunto da galeria (cada imagem como query contra o restante)
    - Ground truth: galeria items da mesma classe documental = relevantes
    - Métricas: EER, R@1, R@5, mAP@10, NDCG@10
    """
    model.eval()

    gallery_embs, gallery_labels = _build_gallery(
        model, processor, val_csv, base_dir,
        device, max_items=max_gallery,
    )

    N = len(gallery_embs)
    if N < 10:
        return {"error": "galeria insuficiente"}

    # Sub-amostra de queries (evita O(N²) total)
    query_idx = np.random.choice(N, size=min(max_queries, N), replace=False)

    # Matriz de similaridade [|queries|, |gallery|]
    q_embs = gallery_embs[query_idx]                       # [Q, D]
    S = q_embs @ gallery_embs.T                            # [Q, N]

    # Exclui auto-similaridade (query == gallery item)
    for i, qi in enumerate(query_idx):
        S[i, qi] = -1.0

    q_labels = gallery_labels[query_idx]                   # [Q]

    r1_list, r5_list, ap_list, ndcg_list, scores_flat, labels_flat = [], [], [], [], [], []

    for i in range(len(query_idx)):
        relevance = (gallery_labels == q_labels[i]).astype(np.float32)
        sorted_idx = np.argsort(-S[i])

        r1 = float(relevance[sorted_idx[0]])
        r5 = float(relevance[sorted_idx[:5]].max())
        r1_list.append(r1)
        r5_list.append(r5)

        # AP@10
        rel_sorted = relevance[sorted_idx[:10]]
        ap = float(average_precision_score(rel_sorted, np.linspace(1, 0, len(rel_sorted)))
                   ) if rel_sorted.sum() > 0 else 0.0
        ap_list.append(ap)

        # NDCG@10
        ndcg = float(ndcg_score(
            rel_sorted[np.newaxis], S[i][sorted_idx[:10]][np.newaxis]
        )) if rel_sorted.sum() > 0 else 0.0
        ndcg_list.append(ndcg)

        # EER (acumula scores de similaridade)
        scores_flat.extend(S[i].tolist())
        labels_flat.extend(relevance.tolist())

    # EER global
    fpr, tpr, _ = roc_curve(labels_flat, scores_flat)
    fnr = 1 - tpr
    idx_eer = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fpr[idx_eer] + fnr[idx_eer]) / 2)

    return {
        "eer":     eer,
        "r@1":     float(np.mean(r1_list)),
        "r@5":     float(np.mean(r5_list)),
        "map@10":  float(np.mean(ap_list)),
        "ndcg@10": float(np.mean(ndcg_list)),
        "n_queries":  len(query_idx),
        "n_gallery":  N,
    }


# ---------------------------------------------------------------------------
# Training pipeline para um split
# ---------------------------------------------------------------------------

def _train_split(args, split_idx: int, run_name: str) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_csv, val_csv = _prepare_split(args.data_root, split_idx)

    model, processor = _make_model(
        args.lora_r, args.lora_dropout, args.load_in_4bit
    )

    # Limita resolução máxima → reduz tokens visuais e VRAM durante treino
    if hasattr(processor, "image_processor") and args.max_pixels:
        processor.image_processor.max_pixels = args.max_pixels

    # Loss: Fase 1 = MatryoshkaInfoNCE; Fase 2 = InfoNCE+
    if args.phase == 1 or not args.use_hard_negatives:
        loss_fn = MatryoshkaInfoNCELoss(MATRYOSHKA_DIMS, TEMPERATURE) if args.matryoshka \
                  else InfoNCELoss(TEMPERATURE)
        use_matryoshka = args.matryoshka
    else:
        loss_fn = InfoNCEPlusLoss(TEMPERATURE, n_hard=args.n_hard_negatives)
        use_matryoshka = False

    # Dataset e DataLoader
    train_ds = _JinaDocDataset(
        train_csv, args.base_image_dir, max_pairs=args.max_train_pairs
    )

    def _collate(batch):
        return _collate_jina(batch, processor, device)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=_collate,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )

    ckpt_dir = Path(args.checkpoint_root) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # W&B
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                entity=WANDB_ENTITY,
                project=WANDB_PROJECT,
                name=run_name,
                config={**vars(args), "split": split_idx,
                        "lora_params_M": _count_lora_params(model) / 1e6,
                        "train_pairs": len(train_ds)},
                reinit=True,
            )
        except Exception as e:
            print(f"W&B init falhou: {e}")
            use_wandb = False

    best_eer  = 1.0
    best_info = {}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = _train_epoch(
            model, optimizer, train_loader,
            loss_fn, device, args.grad_accum, use_matryoshka,
            max_steps=args.max_steps_per_epoch,
        )
        scheduler.step()

        # Benchmark completo a cada época
        metrics = benchmark_lacdip(
            model, processor, val_csv, args.base_image_dir,
            device, max_gallery=args.max_val_gallery, max_queries=args.max_val_queries,
        )

        elapsed = time.time() - t0
        print(
            f"  E{epoch:02d}/{args.epochs}  loss={train_loss:.4f}  "
            f"EER={metrics.get('eer', 1.0)*100:.2f}%  "
            f"R@1={metrics.get('r@1', 0)*100:.1f}%  "
            f"R@5={metrics.get('r@5', 0)*100:.1f}%  "
            f"mAP@10={metrics.get('map@10', 0)*100:.1f}%  "
            f"({elapsed:.0f}s)"
        )

        if use_wandb:
            try:
                import wandb
                wandb.log({
                    "train/loss":  train_loss,
                    "val/eer":     metrics.get("eer", 1.0),
                    "val/r@1":     metrics.get("r@1", 0.0),
                    "val/r@5":     metrics.get("r@5", 0.0),
                    "val/map@10":  metrics.get("map@10", 0.0),
                    "val/ndcg@10": metrics.get("ndcg@10", 0.0),
                    "epoch":       epoch,
                    "lr":          scheduler.get_last_lr()[0],
                })
            except Exception:
                pass

        if metrics.get("eer", 1.0) < best_eer:
            best_eer  = metrics["eer"]
            best_info = {"val_eer": best_eer, "epoch": epoch, **metrics}
            # Salva apenas o adapter LoRA (~120 MB para r=128)
            model.save_pretrained(str(ckpt_dir))
            (ckpt_dir / "best_benchmark.json").write_text(
                json.dumps(best_info, indent=2)
            )
            print(f"  ✓ Adapter salvo  EER={best_eer*100:.2f}% → {ckpt_dir.name}")

    if use_wandb:
        try:
            import wandb
            wandb.log({"val/best_eer": best_eer})
            wandb.finish()
        except Exception:
            pass

    return {"split": split_idx, "best_val_eer": best_eer, "ckpt_dir": str(ckpt_dir),
            "best_metrics": best_info}


# ---------------------------------------------------------------------------
# Benchmark standalone (avalia adapter já treinado)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_benchmark(args):
    """
    Avalia um adapter LoRA salvo sem re-treinar.
    Útil para avaliar no split de teste 5 após treinar nos splits 0-4.
    """
    from peft import PeftModel
    from transformers import AutoModel, AutoProcessor

    print(f"Carregando adapter de {args.lora_checkpoint}...")
    base = AutoModel.from_pretrained(
        MODEL_HF_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto",
    )
    model = PeftModel.from_pretrained(base, args.lora_checkpoint)
    processor = AutoProcessor.from_pretrained(MODEL_HF_ID, trust_remote_code=True)
    device = next(model.parameters()).device

    val_csv = Path(args.data_root) / "splits.csv"   # ajustar conforme protocolo
    if not val_csv.exists():
        raise FileNotFoundError(f"val_csv não encontrado: {val_csv}")

    metrics = benchmark_lacdip(
        model, processor, val_csv, args.base_image_dir,
        str(device), max_gallery=args.max_val_gallery,
        max_queries=args.max_val_queries,
    )
    print("\nBenchmark LA-CDIP:")
    for k, v in metrics.items():
        print(f"  {k}: {v*100:.2f}%" if isinstance(v, float) else f"  {k}: {v}")
    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Treino contrastivo do Jina-v4 com LoRA no LA-CDIP (ZSL)."
    )

    # Dados
    p.add_argument("--data-root",       required=True,
                   help="Raiz do LA-CDIP (contém splits.csv e protocol.csv)")
    p.add_argument("--base-image-dir",  required=True,
                   help="Diretório raiz das imagens originais do LA-CDIP")
    p.add_argument("--checkpoint-root", default=None)

    # Split / modo
    p.add_argument("--split",   type=int, default=None,
                   help="Índice do split ZSL (0–4). Omita para treinar todos.")
    p.add_argument("--lora-checkpoint", default=None,
                   help="Pasta com adapter salvo (modo benchmark sem treino).")

    # Treino
    p.add_argument("--epochs",           type=int,   default=5)
    p.add_argument("--lr",               type=float, default=2e-4)
    p.add_argument("--batch-size",       type=int,   default=4,
                   help="Pares por step. InfoNCE beneficia de batches maiores.")
    p.add_argument("--grad-accum",       type=int,   default=8,
                   help="Acumulação de gradiente (batch efetivo = batch × grad_accum)")
    p.add_argument("--max-train-pairs",    type=int,   default=8000)
    p.add_argument("--max-steps-per-epoch", type=int, default=None,
                   help="Limite de steps por época (mesmo protocolo dos outros scripts; "
                        "None = usa todos os pares)")
    p.add_argument("--max-pixels",        type=int,   default=401408,
                   help="Máx de pixels por imagem no processor (~392K = 448×896, "
                        "reduz tokens visuais e VRAM; default Qwen é 12M)")
    p.add_argument("--phase",            type=int,   default=1, choices=[1, 2],
                   help="1=InfoNCE simétrico (pair training); 2=InfoNCE+ com hard negatives")

    # LoRA
    p.add_argument("--lora-r",       type=int,   default=16,
                   help="Rank LoRA. r=128 → ~59M params (paper usa 60M por adapter)")
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--load-in-4bit", action="store_true",
                   help="Carrega backbone em 4-bit (economiza VRAM, reduz precisão)")

    # Loss
    p.add_argument("--matryoshka",         action="store_true", default=True,
                   help="Usa Matryoshka loss (truncatable embeddings, Seção 5 do paper)")
    p.add_argument("--no-matryoshka",      dest="matryoshka", action="store_false")
    p.add_argument("--use-hard-negatives", action="store_true",
                   help="Fase 2: usa InfoNCE+ com hard negatives in-batch")
    p.add_argument("--n-hard-negatives",   type=int, default=3)
    p.add_argument("--temperature",        type=float, default=TEMPERATURE,
                   help="Temperatura τ do InfoNCE (paper usa 0.02)")

    # Avaliação
    p.add_argument("--max-val-gallery",  type=int, default=2000,
                   help="Tamanho máximo da galeria no benchmark")
    p.add_argument("--max-val-queries",  type=int, default=500,
                   help="Número de queries no benchmark por época")

    # Misc
    p.add_argument("--gpu-id",    type=int, default=None)
    p.add_argument("--no-wandb",  action="store_true")
    p.add_argument("--seed",      type=int, default=42)

    args = p.parse_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.checkpoint_root is None:
        for candidate in [
            "/mnt/nas/joaopaulo/CaVL-Doc/checkpoints",
            "/mnt/large/checkpoints",
        ]:
            if Path(candidate).exists():
                args.checkpoint_root = candidate
                break
        else:
            args.checkpoint_root = str(WORKSPACE_ROOT / "checkpoints")

    # Modo benchmark (sem treino)
    if args.lora_checkpoint:
        run_benchmark(args)
        return

    splits    = list(range(5)) if args.split is None else [args.split]
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    all_results = []
    for split_idx in splits:
        run_name = (
            f"JinaV4_LoRA_LA-CDIP_S{split_idx}"
            f"_r{args.lora_r}_ph{args.phase}_{timestamp}"
        )
        print(f"\n{'='*70}")
        print(f"[SPLIT {split_idx}]  {run_name}")
        print(f"{'='*70}")
        result = _train_split(args, split_idx, run_name)
        all_results.append(result)
        m = result.get("best_metrics", {})
        print(
            f"\n  → Split {split_idx}  "
            f"EER={result['best_val_eer']*100:.2f}%  "
            f"R@1={m.get('r@1', 0)*100:.1f}%  "
            f"R@5={m.get('r@5', 0)*100:.1f}%"
        )

    eers = [r["best_val_eer"] for r in all_results]
    r1s  = [r.get("best_metrics", {}).get("r@1", 0) for r in all_results]
    r5s  = [r.get("best_metrics", {}).get("r@5", 0) for r in all_results]
    maps = [r.get("best_metrics", {}).get("map@10", 0) for r in all_results]

    print(f"\n{'='*70}")
    print("RESUMO BENCHMARK LA-CDIP (Jina-VDR style)")
    for r in all_results:
        m = r.get("best_metrics", {})
        print(
            f"  Split {r['split']}: "
            f"EER={r['best_val_eer']*100:.2f}%  "
            f"R@1={m.get('r@1',0)*100:.1f}%  "
            f"R@5={m.get('r@5',0)*100:.1f}%  "
            f"mAP@10={m.get('map@10',0)*100:.1f}%"
        )
    if len(all_results) > 1:
        print(f"  Média EER:  {np.mean(eers)*100:.2f}% ± {np.std(eers)*100:.2f} pp")
        print(f"  Média R@1:  {np.mean(r1s)*100:.2f}% ± {np.std(r1s)*100:.2f} pp")

    # Run de resumo consolidado no W&B (igual ao padrão dos outros baselines)
    if not args.no_wandb and len(all_results) > 1:
        try:
            import wandb
            summary_name = f"JinaV4_LoRA_LA-CDIP_SUMMARY_r{args.lora_r}_{timestamp}"
            wandb.init(
                entity=WANDB_ENTITY,
                project=WANDB_PROJECT,
                name=summary_name,
                config={**vars(args), "n_splits": len(all_results)},
                reinit=True,
            )
            summary = {
                "summary/mean_eer":    float(np.mean(eers)),
                "summary/std_eer":     float(np.std(eers)),
                "summary/mean_r@1":    float(np.mean(r1s)),
                "summary/mean_r@5":    float(np.mean(r5s)),
                "summary/mean_map@10": float(np.mean(maps)),
            }
            for r in all_results:
                s = r["split"]
                summary[f"split_{s}/eer"] = r["best_val_eer"]
                summary[f"split_{s}/r@1"] = r.get("best_metrics", {}).get("r@1", 0)
                summary[f"split_{s}/r@5"] = r.get("best_metrics", {}).get("r@5", 0)
            wandb.log(summary)
            wandb.finish()
        except Exception as e:
            print(f"W&B summary falhou: {e}")

    print(f"\nAdapters salvos em: {args.checkpoint_root}")
    print("Para avaliar no split de teste 5, use --lora-checkpoint <ckpt_dir>")
    print("="*70)


if __name__ == "__main__":
    main()
