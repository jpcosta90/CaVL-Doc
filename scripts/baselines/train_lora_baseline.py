#!/usr/bin/env python3
"""
Fine-tunes InternVL3-2B com QLoRA (SFT generativo) como baseline para o CaVL-Doc.

O modelo aprende a gerar um JSON de similaridade {"similarity_score": N} ao receber
dois documentos + prompt de similaridade — o mesmo prompt do eval_vlm_prompt.py.

Diferença entre os três métodos:
  eval_vlm_prompt.py     → zero-shot VLM (sem adaptação)
  train_lora_baseline.py → este script: adapta o VLM com LoRA SFT nos pares de treino
  CaVL-Doc               → arquitetura siamese especializada (pooler + cabeça métrica)

Dados de treino: mesmos train_pairs.csv do protocolo ZSL (prepare_protocol_split.py)
Target:
  is_equal=1 → {"similarity_score": 90}   (par do mesmo tipo de documento)
  is_equal=0 → {"similarity_score": 10}   (par de tipos diferentes)

Loss: cross-entropy apenas nos tokens da resposta (instrução mascarada com -100)

Avaliação pós-treino: usar eval_vlm_prompt.py com --lora-checkpoint <ckpt_dir>
  (ou rodar com --eval-split 5 ao final do treino)

Checkpoint: checkpoints/<run_name>/  (adapter LoRA, ~30-60 MB)
W&B:        CaVL-Doc_LA-CDIP_LoRA_Baseline

Uso (um split)
  python scripts/baselines/train_lora_baseline.py \
      --data-root /mnt/data/la-cdip \
      --base-image-dir /mnt/data/la-cdip/data \
      --split 0 --epochs 3 --lora-r 16

Uso (todos os splits)
  for S in 0 1 2 3 4; do
    python scripts/baselines/train_lora_baseline.py \
        --data-root /mnt/data/la-cdip \
        --base-image-dir /mnt/data/la-cdip/data \
        --split $S --epochs 3
  done
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
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PREP_SCRIPT    = WORKSPACE_ROOT / "scripts" / "utils" / "prepare_protocol_split.py"

WANDB_ENTITY  = "jpcosta1990-university-of-brasilia"
WANDB_PROJECT = "CaVL-Doc_LA-CDIP_LoRA_Baseline"

MODEL_HF_ID = "OpenGVLab/InternVL3-2B"

# Prompt idêntico ao eval_vlm_prompt.py — consistência entre treino e avaliação
SIMILARITY_PROMPT = """\
Image-1: <image>
Image-2: <image>

You are an AI assistant specialized in document analysis. Your task is to compare two company documents and assess their **visual similarity** based on their layout structure.

**Instructions:**
Analyze the two provided document images and measure their **visual similarity** based on:
- **Shapes and Elements:** Compare the presence of graphical components, tables, sections, headers, and any other visual elements.
- **Layout Consistency:** Evaluate the spatial arrangement of text blocks, margins, and alignments.
- **Content Type:** Ensure that both documents contain similar types of content (e.g., tables, forms, paragraphs), regardless of specific wording.

**Similarity Scoring:**
Assign a **similarity score** between **0 and 100**, where:
- **90-100** → **Nearly identical**: Documents have almost no visual differences.
- **70-89** → **Highly similar**: Documents share the same structure with minor variations.
- **50-69** → **Moderately similar**: Key components remain, but there are noticeable structural differences.
- **30-49** → **Weak similarity**: Some elements are shared, but the overall layout is significantly different.
- **0-29** → **Completely different**: The documents do not share a recognizable visual structure.

**Output Format:**
Respond **only** with a JSON object:
{"similarity_score": <integer 0-100>}"""

# Score alvo para treino SFT
SCORE_POSITIVE = 90   # pares do mesmo tipo de documento
SCORE_NEGATIVE = 10   # pares de tipos diferentes

IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMG_START_TOKEN   = "<img>"
IMG_END_TOKEN     = "</img>"

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


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
# Model
# ---------------------------------------------------------------------------

def _make_model(lora_r: int, lora_dropout: float):
    """
    Carrega InternVL3-2B em 4-bit e aplica adapters LoRA.
    Os adapters são treinados para adaptar a geração de texto, não os embeddings.
    """
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print(f"Carregando {MODEL_HF_ID} em 4-bit QLoRA...")
    base = AutoModel.from_pretrained(
        MODEL_HF_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_cfg,
    )
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_HF_ID, trust_remote_code=True, use_fast=False
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# SFT data preparation
# ---------------------------------------------------------------------------

def _img_token_str(n_patches: int, num_image_token: int) -> str:
    """Constrói o bloco de tokens de imagem no formato InternVL3."""
    return IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token * n_patches + IMG_END_TOKEN


def _build_sft_tokens(
    tokenizer,
    num_image_token: int,
    n_patches_a: int,
    n_patches_b: int,
    is_equal: int,
) -> Tuple[List[int], List[int]]:
    """
    Constrói (input_ids, labels) para um exemplo de treino SFT.

    A instrução recebe label -100 (não contribui para o loss).
    Apenas os tokens da resposta JSON têm labels reais.

    Formato da conversa (InternVL3 / InternLM2 chatml):
      <|im_start|>user
      {question com tokens de imagem}
      <|im_end|>
      <|im_start|>assistant
      {{"similarity_score": 90 ou 10}}
      <|im_end|>
    """
    # Embutir tokens de imagem no prompt (mesmo processo que model.chat() internamente)
    question = SIMILARITY_PROMPT
    question = question.replace("<image>", _img_token_str(n_patches_a, num_image_token), 1)
    question = question.replace("<image>", _img_token_str(n_patches_b, num_image_token), 1)

    target_score = SCORE_POSITIVE if is_equal else SCORE_NEGATIVE
    response     = json.dumps({"similarity_score": target_score})

    messages = [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": response},
    ]

    # Tenta apply_chat_template (InternLM2 / chatml suportam)
    try:
        full_text  = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        instr_text = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback manual compatível com InternVL3
        im_s = "<|im_start|>"
        im_e = "<|im_end|>"
        instr_text = (f"{im_s}user\n{question}{im_e}\n{im_s}assistant\n")
        full_text  = instr_text + response + im_e + "\n"

    full_ids  = tokenizer(full_text,  add_special_tokens=False)["input_ids"]
    instr_ids = tokenizer(instr_text, add_special_tokens=False)["input_ids"]

    n_instr = len(instr_ids)
    labels  = [-100] * n_instr + full_ids[n_instr:]

    return full_ids, labels


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class _SFTDataset(Dataset):
    """Par de documentos → exemplo de fine-tuning generativo."""

    def __init__(self, csv_path: Path, base_dir: str,
                 tokenizer, num_image_token: int,
                 max_pairs: Optional[int] = None, max_tiles: int = 6):
        import pandas as pd
        self.df       = pd.read_csv(csv_path)
        self.base_dir = base_dir
        self.tokenizer       = tokenizer
        self.num_image_token = num_image_token
        self.max_tiles       = max_tiles

        if max_pairs and len(self.df) > max_pairs:
            # Balanceia positivos e negativos no subset
            pos = self.df[self.df["is_equal"] == 1].sample(
                n=min(max_pairs // 2, (self.df["is_equal"] == 1).sum()),
                random_state=42
            )
            neg = self.df[self.df["is_equal"] == 0].sample(
                n=min(max_pairs - len(pos), (self.df["is_equal"] == 0).sum()),
                random_state=42
            )
            self.df = pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"  SFT dataset: {len(self.df)} pares  "
              f"(pos={int(self.df['is_equal'].sum())}, "
              f"neg={int((self.df['is_equal']==0).sum())})")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        from cavl_doc.data.transforms import build_transform, dynamic_preprocess
        from PIL import Image as PILImage

        row = self.df.iloc[idx]
        tfm = build_transform(448)

        def _load(rel_path):
            path = os.path.join(self.base_dir, str(rel_path))
            img  = PILImage.open(path).convert("RGB")
            tiles = dynamic_preprocess(img, image_size=448,
                                       use_thumbnail=True, max_num=self.max_tiles)
            return torch.stack([tfm(t) for t in tiles])  # [N, 3, H, W]

        img_a    = _load(row["file_a_path"])
        img_b    = _load(row["file_b_path"])
        is_equal = int(row["is_equal"])

        input_ids, labels = _build_sft_tokens(
            self.tokenizer, self.num_image_token,
            img_a.shape[0], img_b.shape[0], is_equal,
        )

        return {
            "input_ids":    torch.tensor(input_ids, dtype=torch.long),
            "labels":       torch.tensor(labels,    dtype=torch.long),
            "pixel_values": torch.cat([img_a, img_b], dim=0),  # [Na+Nb, 3, H, W]
            "image_flags":  torch.ones(img_a.shape[0] + img_b.shape[0], dtype=torch.long),
            "n_patches_a":  img_a.shape[0],
            "n_patches_b":  img_b.shape[0],
        }


def _collate_sft(batch):
    """Padding dinâmico para input_ids e labels; pixel_values sem padding."""
    pad_id = 0
    max_len = max(b["input_ids"].shape[0] for b in batch)

    input_ids_padded = []
    labels_padded    = []
    attn_masks       = []
    for b in batch:
        pad = max_len - b["input_ids"].shape[0]
        input_ids_padded.append(F.pad(b["input_ids"], (0, pad), value=pad_id))
        labels_padded.append(F.pad(b["labels"],    (0, pad), value=-100))
        attn_masks.append(F.pad(torch.ones(b["input_ids"].shape[0]), (0, pad), value=0))

    return {
        "input_ids":    torch.stack(input_ids_padded),
        "labels":       torch.stack(labels_padded),
        "attention_mask": torch.stack(attn_masks),
        # pixel_values: cada item tem tamanho diferente → lista de tensores
        "pixel_values": [b["pixel_values"] for b in batch],
        "image_flags":  [b["image_flags"]  for b in batch],
        "n_patches_a":  [b["n_patches_a"]  for b in batch],
        "n_patches_b":  [b["n_patches_b"]  for b in batch],
    }


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def _forward_loss(model, batch, device) -> torch.Tensor:
    """
    Forward pass SFT: computa LM loss apenas nos tokens de resposta.
    pixel_values de imagens com diferentes números de tiles são concatenados
    separadamente por exemplo (não fazem padding entre si).
    """
    input_ids       = batch["input_ids"].to(device)
    labels          = batch["labels"].to(device)
    attention_mask  = batch["attention_mask"].to(device)

    # Concatena pixel_values de todos os exemplos do batch (tamanhos variáveis OK)
    pixel_values = torch.cat(batch["pixel_values"], dim=0).to(device, dtype=torch.bfloat16)
    image_flags  = torch.cat(batch["image_flags"],  dim=0).to(device)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_flags=image_flags,
        labels=labels,
        return_dict=True,
    )
    return out.loss


def _train_epoch(model, optimizer, loader, device, accum_steps) -> float:
    model.train()
    losses = []
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(loader, desc="Train", ncols=100, leave=False)):
        try:
            loss = _forward_loss(model, batch, device)
            (loss / accum_steps).backward()
            losses.append(loss.item())
        except Exception as e:
            print(f"\n  [WARN] step {step} falhou: {e}")
            optimizer.zero_grad()
            continue

        if (step + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            optimizer.zero_grad()

    optimizer.step()
    optimizer.zero_grad()
    return float(np.mean(losses)) if losses else float("nan")


# ---------------------------------------------------------------------------
# Validation (generate + parse score → EER)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _validate(model, tokenizer, val_csv: Path, base_dir: str,
              device: str, max_val: int = 500) -> Tuple[float, float]:
    """
    Roda model.chat() nos pares de validação e calcula EER com o score gerado.
    Replica exatamente o que eval_vlm_prompt.py faz para o modelo base.
    """
    import re, pandas as pd
    from cavl_doc.data.transforms import build_transform, dynamic_preprocess
    from PIL import Image as PILImage

    model.eval()
    df = pd.read_csv(val_csv)
    if len(df) > max_val:
        df = pd.concat([
            df[df["is_equal"] == 1].sample(n=min(max_val // 2, (df["is_equal"] == 1).sum()), random_state=42),
            df[df["is_equal"] == 0].sample(n=min(max_val // 2, (df["is_equal"] == 0).sum()), random_state=42),
        ]).sample(frac=1, random_state=42).reset_index(drop=True)

    tfm = build_transform(448)

    def _load(rel_path):
        img = PILImage.open(os.path.join(base_dir, str(rel_path))).convert("RGB")
        tiles = dynamic_preprocess(img, image_size=448, use_thumbnail=True, max_num=6)
        return torch.stack([tfm(t) for t in tiles])

    def _parse(text: str) -> Optional[float]:
        try:
            clean = re.sub(r"```[a-z]*", "", text).strip().strip("`")
            return float(json.loads(clean)["similarity_score"])
        except Exception:
            m = re.search(r'"similarity_score"\s*:\s*(\d+(?:\.\d+)?)', text)
            return float(m.group(1)) if m else None

    all_scores, all_labels = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Val", ncols=100, leave=False):
        try:
            pv_a = _load(row["file_a_path"])
            pv_b = _load(row["file_b_path"])
            pv   = torch.cat([pv_a, pv_b]).to(torch.bfloat16).to(device)

            # Usa o MESMO prompt de treino para consistência
            prompt = SIMILARITY_PROMPT
            response = model.chat(
                tokenizer, pv, prompt,
                generation_config={"max_new_tokens": 64, "do_sample": False},
                num_patches_list=[pv_a.shape[0], pv_b.shape[0]],
            )
            score = _parse(response)
            if score is not None:
                all_scores.append(score)
                all_labels.append(int(row["is_equal"]))
        except Exception:
            continue

    if len(all_scores) < 10:
        return 1.0, 0.0

    scores = np.array(all_scores)
    labels = np.array(all_labels)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    thr = float(thresholds[idx])
    return eer, thr


# ---------------------------------------------------------------------------
# Full training loop for one split
# ---------------------------------------------------------------------------

def _train_split(args, split_idx: int, run_name: str) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_csv, val_csv = _prepare_split(args.data_root, split_idx)

    model, tokenizer = _make_model(args.lora_r, args.lora_dropout)

    # Descobre num_image_token do modelo base
    base_model = model
    while hasattr(base_model, "base_model"):
        base_model = base_model.base_model
        if hasattr(base_model, "model"):
            base_model = base_model.model
    num_image_token = base_model.num_image_token

    # Dataset de treino (balanceado, limitado a max_train_pairs)
    train_ds = _SFTDataset(
        train_csv, args.base_image_dir, tokenizer,
        num_image_token, max_pairs=args.max_train_pairs, max_tiles=6,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=_collate_sft,
    )

    # Optimizer: apenas parâmetros treináveis (LoRA adapters)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4,
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
        train_loss = _train_epoch(model, optimizer, train_loader, device, args.grad_accum)

        # Validação a cada época
        val_eer, val_thr = _validate(
            model, tokenizer, val_csv, args.base_image_dir, device, max_val=args.max_val_pairs
        )
        elapsed = time.time() - t0
        print(f"  E{epoch:02d}/{args.epochs}  loss={train_loss:.4f}  "
              f"val_eer={val_eer*100:.2f}%  thr={val_thr:.1f}  ({elapsed:.0f}s)")

        if use_wandb:
            try:
                import wandb
                wandb.log({"train/loss": train_loss, "val/eer": val_eer,
                           "val/threshold": val_thr, "epoch": epoch})
            except Exception:
                pass

        if val_eer < best_eer:
            best_eer  = val_eer
            best_info = {"val_eer": best_eer, "epoch": epoch, "threshold": val_thr}
            model.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))
            (ckpt_dir / "best_val_eer.json").write_text(json.dumps(best_info, indent=2))
            print(f"  ✓ Adapter salvo  val_eer={best_eer*100:.2f}%  → {ckpt_dir.name}")

    if use_wandb:
        try:
            import wandb
            wandb.log({"val/best_eer": best_eer})
            wandb.finish()
        except Exception:
            pass

    return {"split": split_idx, "best_val_eer": best_eer, "ckpt_dir": str(ckpt_dir)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="SFT generativo do InternVL3-2B com LoRA como baseline para o CaVL-Doc."
    )
    p.add_argument("--data-root",       required=True,
                   help="Raiz do LA-CDIP (contém splits.csv e protocol.csv)")
    p.add_argument("--base-image-dir",  required=True,
                   help="Diretório raiz das imagens")
    p.add_argument("--checkpoint-root", default=None)
    p.add_argument("--split",   type=int, default=None,
                   help="Índice do split (0–4). Omita para treinar os 5 splits sequencialmente.")
    p.add_argument("--epochs",  type=int,   default=3,
                   help="Épocas de treino (3 é suficiente para SFT; mais pode gerar overfitting)")
    p.add_argument("--lr",      type=float, default=2e-4)
    p.add_argument("--batch-size",        type=int, default=1,
                   help="Pares por step. Manter 1 para 16 GB VRAM.")
    p.add_argument("--grad-accum",        type=int, default=16,
                   help="Acumulação de gradiente (batch efetivo = batch_size × grad_accum)")
    p.add_argument("--max-train-pairs",   type=int, default=4000,
                   help="Limite de pares de treino por split (balanceados pos/neg)")
    p.add_argument("--max-val-pairs",     type=int, default=500,
                   help="Limite de pares para validação por época (para não demorar)")
    p.add_argument("--lora-r",            type=int,   default=16,
                   help="Rank LoRA. r=8: ~2.4M params; r=16: ~4.7M params")
    p.add_argument("--lora-dropout",      type=float, default=0.05)
    p.add_argument("--gpu-id",  type=int, default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.checkpoint_root is None:
        for candidate in ["/mnt/large/checkpoints", "/mnt/nas/joaopaulo/checkpoints"]:
            if Path(candidate).exists():
                args.checkpoint_root = candidate
                break
        else:
            args.checkpoint_root = str(WORKSPACE_ROOT / "checkpoints")

    splits    = list(range(5)) if args.split is None else [args.split]
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    all_results = []
    for split_idx in splits:
        run_name = f"LoRA_LA-CDIP_InternVL3-2B_S{split_idx}_r{args.lora_r}_{timestamp}"
        print(f"\n{'='*70}")
        print(f"[SPLIT {split_idx}]  {run_name}")
        print(f"{'='*70}")
        result = _train_split(args, split_idx, run_name)
        all_results.append(result)
        print(f"\n  → Split {split_idx}  best_val_eer={result['best_val_eer']*100:.2f}%")

    print(f"\n{'='*70}")
    print("RESUMO")
    for r in all_results:
        print(f"  Split {r['split']}: {r['best_val_eer']*100:.2f}%  → {r['ckpt_dir']}")
    if len(all_results) > 1:
        eers = [r["best_val_eer"] for r in all_results]
        print(f"  Média: {np.mean(eers)*100:.2f}%  ±  {np.std(eers)*100:.2f} pp")
    print(f"\nPara avaliar no split de teste 5, use eval_vlm_prompt.py com")
    print(f"o adapter salvo (--lora-checkpoint <ckpt_dir>).")
    print("="*70)


if __name__ == "__main__":
    main()
