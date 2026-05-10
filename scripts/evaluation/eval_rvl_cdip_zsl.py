#!/usr/bin/env python3
"""
Avaliação zero-shot inter-dataset: modelo CaVL-Doc (treinado em LA-CDIP) → RVL-CDIP.

Baixa o checkpoint do Hugging Face Hub ou usa um caminho local.
Avalia nos 4 splits ZSL (RVL-CDIP_zsl_split_0 … _3).
Métrica principal: EER (Equal Error Rate) via similaridade de cosseno.

Uso:
    # Modelo do HuggingFace (default):
    python scripts/evaluation/eval_rvl_cdip_zsl.py \
        --base-image-dir /mnt/data/zs_rvl_cdip/data \
        --gpu-id 0

    # Checkpoint local:
    python scripts/evaluation/eval_rvl_cdip_zsl.py \
        --checkpoint-path /mnt/large/checkpoints/.../best_siam.pt \
        --base-image-dir /mnt/data/zs_rvl_cdip/data \
        --gpu-id 0
"""
from __future__ import annotations

import argparse
import csv as _csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from cavl_doc.data.transforms import build_transform, dynamic_preprocess
from cavl_doc.models.backbone_loader import load_model
from cavl_doc.models.modeling_cavl import build_cavl_model
from cavl_doc.utils.embedding_utils import prepare_inputs_for_multimodal_embedding

WANDB_ENTITY     = "jpcosta1990-university-of-brasilia"
WANDB_PROJECT    = "CaVL-Doc_RVL-CDIP_ZSL"
DEFAULT_REPO_ID  = "Jpcosta90/cavl-doc-lacdip"
EMBEDDING_PROMPT = "<image> Analyze this document"

_TRANSFORM = build_transform(input_size=448)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_config(ckpt_path: Path, ckpt: dict) -> dict:
    config = ckpt.get("config", {})
    if not config and "args" in ckpt:
        obj = ckpt["args"]
        config = vars(obj) if hasattr(obj, "__dict__") else {}
    if not config:
        jpath = ckpt_path.parent / "training_config.json"
        if jpath.exists():
            config = json.loads(jpath.read_text())
    if not config:
        raise ValueError("Não foi possível recuperar config do checkpoint.")
    return config


def _load_siam(ckpt_path: Path, backbone, tokenizer, device: str):
    """Carrega pool + head do checkpoint. Backbone já está carregado."""
    ckpt   = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    config = _load_config(ckpt_path, ckpt)

    cut_layer = int(config.get("cut_layer", 27))
    proj_out  = int(config.get("projection_output_dim", 1536))

    # Usamos encode_fn=None para que siam.forward não tente chamar _encode_fn.
    # O embedding será calculado diretamente via backbone + pool + head.
    siam = build_cavl_model(
        backbone=backbone,
        cut_layer=cut_layer,
        encode_fn=None,          # desativado — usamos _embed_direct abaixo
        pool_dim=1536,
        proj_hidden=4096,
        proj_out=proj_out,
        set_trainable=False,
        tokenizer=tokenizer,
        pooler_type=config.get("pooler_type", "attention"),
        head_type=config.get("head_type", "mlp"),
        num_queries=int(config.get("num_queries", 1)),
    )

    if "siam_pool" in ckpt and "siam_head" in ckpt:
        siam.pool.load_state_dict(ckpt["siam_pool"])
        siam.head.load_state_dict(ckpt["siam_head"])
        if ckpt.get("backbone_trainable"):
            siam.backbone.load_state_dict(ckpt["backbone_trainable"], strict=False)
    elif "model_state_dict" in ckpt:
        siam.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        raise ValueError("Checkpoint sem pesos reconhecidos.")

    siam.to(device).eval()
    return siam, config, cut_layer


def _resolve_checkpoint(repo_id: str | None, checkpoint_path: str | None, cache_dir: str | None) -> Path:
    if checkpoint_path:
        p = Path(checkpoint_path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint não encontrado: {p}")
        return p
    from huggingface_hub import hf_hub_download
    # Tenta best_model.pt primeiro (novo nome); fallback para best_siam.pt (legado no Hub)
    for fname in ["best_model.pt", "best_siam.pt"]:
        try:
            print(f"Baixando {fname} de {repo_id} ...")
            return Path(hf_hub_download(repo_id=repo_id, filename=fname, cache_dir=cache_dir))
        except Exception:
            continue
    raise FileNotFoundError(f"Nenhum checkpoint encontrado em {repo_id} (best_model.pt / best_siam.pt)")


# ---------------------------------------------------------------------------
# Direct embedding — bypasses _encode_fn, igual ao _InternVL3Embedder
# ---------------------------------------------------------------------------

@torch.no_grad()
def _embed(img: Image.Image, backbone, tokenizer, siam, cut_layer: int,
           device: str, max_num: int) -> torch.Tensor:
    """
    Executa o backbone diretamente (como _InternVL3Embedder) e aplica
    pool + head do CaVL-Doc sobre os hidden states do cut_layer.
    Evita o bug de mismatch de tokens do _encode_fn.
    """
    tiles        = dynamic_preprocess(img, image_size=448, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([_TRANSFORM(t) for t in tiles]).to(torch.bfloat16)

    inp          = prepare_inputs_for_multimodal_embedding(backbone, tokenizer, pixel_values, EMBEDDING_PROMPT)
    input_ids    = inp["input_ids"].to(device)
    pixel_values = inp["pixel_values"].to(device, dtype=torch.bfloat16)
    image_flags  = inp["image_flags"].to(device)

    result = backbone(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_flags=image_flags,
        output_hidden_states=True,
        return_dict=True,
    )

    hidden = result.hidden_states
    lm     = backbone.language_model.model
    idx    = cut_layer + 1 if len(hidden) == (len(lm.layers) + 1) else cut_layer
    tokens = hidden[idx]  # [1, seq_len, hidden_dim]

    pooled = siam.pool(tokens, mask=None)
    return siam.head(pooled)  # [1, proj_out]


# ---------------------------------------------------------------------------
# EER
# ---------------------------------------------------------------------------

def _compute_eer(scores: np.ndarray, labels: np.ndarray):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2), float(thr[idx])


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def _load_pairs(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(_csv.DictReader(f))


@torch.no_grad()
def _eval_split(backbone, tokenizer, siam, cut_layer: int, val_csv: Path,
                base_image_dir: str, device: str, limit: int | None,
                max_num: int) -> tuple[float, float, int]:
    pairs = _load_pairs(val_csv)
    if limit:
        pairs = pairs[:limit]

    all_scores, all_labels = [], []

    for r in tqdm(pairs, desc=f"  {val_csv.parent.name}", ncols=90):
        path_a = Path(base_image_dir) / r["file_a_path"]
        path_b = Path(base_image_dir) / r["file_b_path"]
        try:
            img_a = Image.open(path_a).convert("RGB")
            img_b = Image.open(path_b).convert("RGB")
        except Exception as e:
            print(f"\n  [WARN] erro ao abrir imagem: {e}")
            continue

        za = _embed(img_a, backbone, tokenizer, siam, cut_layer, device, max_num)
        zb = _embed(img_b, backbone, tokenizer, siam, cut_layer, device, max_num)

        score = F.cosine_similarity(za, zb, dim=-1).item()
        all_scores.append(score)
        all_labels.append(int(r["is_equal"]))

    eer, thr = _compute_eer(np.array(all_scores), np.array(all_labels))
    return eer, thr, len(all_labels)


# ---------------------------------------------------------------------------
# W&B
# ---------------------------------------------------------------------------

def _log_wandb(run_label: str, split_idx: int, eer: float, thr: float,
               n_pairs: int, entity: str, project: str, wb_config: dict) -> None:
    try:
        import wandb
        run_name = f"{run_label}_split{split_idx}"
        run = wandb.init(entity=entity, project=project, name=run_name,
                         config={**wb_config, "split": split_idx, "n_pairs": n_pairs},
                         reinit=True)
        wandb.log({"test/eer": eer, "test/threshold": thr, "test/n_pairs": n_pairs})
        run.finish()
        print(f"  W&B logged: {run_name}")
    except Exception as e:
        print(f"  ⚠️  W&B log falhou: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Avaliação ZSL inter-dataset CaVL-Doc → RVL-CDIP")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--repo-id",         default=DEFAULT_REPO_ID,
                     help=f"Repo HuggingFace Hub (default: {DEFAULT_REPO_ID})")
    src.add_argument("--checkpoint-path", default=None,
                     help="Caminho local para best_siam.pt")
    p.add_argument("--base-image-dir",    required=True,
                   help="Diretório base das imagens RVL-CDIP")
    p.add_argument("--splits",            default="0,1,2,3")
    p.add_argument("--run-label",         default=None)
    p.add_argument("--max-num",           type=int, default=12,
                   help="Max patches por imagem em dynamic_preprocess (default: 12, igual ao treino)")
    p.add_argument("--gpu-id",            type=int, default=None)
    p.add_argument("--hf-cache-dir",      default=None)
    p.add_argument("--wandb-entity",      default=WANDB_ENTITY)
    p.add_argument("--wandb-project",     default=WANDB_PROJECT)
    p.add_argument("--no-wandb",          action="store_true")
    p.add_argument("--limit",             type=int, default=None)
    args = p.parse_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    split_indices = [int(s.strip()) for s in args.splits.split(",") if s.strip()]

    ckpt_path = _resolve_checkpoint(args.repo_id, args.checkpoint_path, args.hf_cache_dir)
    print(f"Checkpoint: {ckpt_path}")

    run_label = args.run_label or (
        args.repo_id.split("/")[-1] if (args.repo_id and not args.checkpoint_path)
        else ckpt_path.parent.name
    )

    # Backbone em bfloat16 para carregar backbone_trainable do checkpoint
    print("Carregando backbone InternVL3-2B (bfloat16) ...")
    backbone, _, tokenizer, _, _ = load_model("InternVL3-2B", load_in_4bit=False)
    backbone = backbone.to(device)
    # Required: cached InternVL3-2B initializes img_context_token_id=None and never
    # sets it from the tokenizer, causing (input_ids == None) → Python False in forward.
    backbone.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")

    print("Carregando pool + head do checkpoint ...")
    siam, ckpt_config, cut_layer = _load_siam(ckpt_path, backbone, tokenizer, device)

    ckpt_raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    eer_treino = ckpt_raw.get("metrics", {}).get("eer")
    if eer_treino is not None:
        print(f"  EER no treino (LA-CDIP): {eer_treino*100:.2f}%")
    print(f"  max_num={args.max_num}  cut_layer={cut_layer}")

    wb_config = {
        "checkpoint": str(ckpt_path), "run_label": run_label,
        "dataset": "RVL-CDIP", "protocol": "zsl_inter_dataset",
        "cut_layer": cut_layer, "max_num": args.max_num,
    }

    summary_rows = []

    for split_idx in split_indices:
        csv_path = (
            WORKSPACE_ROOT / "data" / "generated_splits"
            / f"RVL-CDIP_zsl_split_{split_idx}" / "validation_pairs.csv"
        )
        if not csv_path.exists():
            print(f"[WARN] Split {split_idx} CSV não encontrado: {csv_path}")
            continue

        print(f"\n{'='*60}")
        print(f"SPLIT {split_idx}  ({csv_path.parent.name})")
        print(f"{'='*60}")

        t0 = time.time()
        eer, thr, n_pairs = _eval_split(
            backbone, tokenizer, siam, cut_layer,
            csv_path, args.base_image_dir,
            device, args.limit, args.max_num,
        )
        elapsed = time.time() - t0

        print(f"  EER={eer*100:.2f}%  threshold={thr:.4f}  pairs={n_pairs}  ({elapsed/60:.1f} min)")

        summary_rows.append({
            "split": split_idx, "n_pairs": n_pairs,
            "eer": eer, "eer_pct": round(eer * 100, 2),
            "threshold": thr, "elapsed_min": round(elapsed / 60, 1),
        })

        if not args.no_wandb:
            _log_wandb(run_label, split_idx, eer, thr, n_pairs,
                       args.wandb_entity, args.wandb_project, wb_config)

    if summary_rows:
        import pandas as pd
        df = pd.DataFrame(summary_rows)
        print(f"\n{'='*60}")
        print(f"RESUMO — {run_label} @ RVL-CDIP ZSL")
        print(f"{'='*60}")
        print(df[["split", "n_pairs", "eer_pct", "threshold"]].to_string(index=False))
        eers = df["eer"].values
        print(f"\n  Média EER:   {eers.mean()*100:.2f}%")
        print(f"  Std EER:     {eers.std()*100:.2f} pp")
        print(f"  Mediana EER: {np.median(eers)*100:.2f}%")

        out_csv = WORKSPACE_ROOT / "results" / f"RVL-CDIP_zsl_{run_label}.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"\n  Resultados: {out_csv}")


if __name__ == "__main__":
    main()
