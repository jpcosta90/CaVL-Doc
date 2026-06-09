#!/usr/bin/env python3
"""
Baixa as variantes CosDoc do Hugging Face e avalia no split indicado.

Uso:
    python scripts/evaluation/eval_cosdoc_hf.py \
        --base-image-dir /mnt/data/la-cdip/data \
        --split 0

    # Só algumas variantes:
    python scripts/evaluation/eval_cosdoc_hf.py \
        --base-image-dir /mnt/data/la-cdip/data \
        --split 0 \
        --variants "" nq2
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

# Reutiliza funções do eval principal
from eval_lacdip_full import (
    _build_backbone,
    _build_siam,
    _load_weights,
    _run_eval,
    PROJ_OUT_DIM,
    SPLITS_DIR_TEMPLATE,
)

# ---------------------------------------------------------------------------
# Variantes
# ---------------------------------------------------------------------------

VARIANTS: dict[str, str] = {
    "":                          "cosdoc",
    "nq2":                       "cosdoc-nq2",
    "cross_modal":               "cosdoc-cross-modal",
    "cross_modal_richprompt_cor": "cosdoc-cross-modal-richprompt",
}

LABELS: dict[str, str] = {
    "":                          "Attention nq=1 (baseline)",
    "nq2":                       "Attention nq=2",
    "cross_modal":               "Cross-Modal",
    "cross_modal_richprompt_cor": "Cross-Modal + Rich Prompt",
}


def _download_checkpoint(repo_id: str, cache_dir: Path | None = None) -> Path:
    from huggingface_hub import hf_hub_download
    local = hf_hub_download(
        repo_id=repo_id,
        filename="best_model.pt",
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return Path(local)


def _subsample_csv(csv_path: Path, max_pairs: int, seed: int = 42) -> Path:
    """Retorna um arquivo CSV temporário com subset balanceado (max_pairs/2 de cada classe)."""
    df = pd.read_csv(csv_path)
    half = max_pairs // 2
    pos = df[df["is_equal"] == 1].sample(n=min(half, (df["is_equal"] == 1).sum()), random_state=seed)
    neg = df[df["is_equal"] == 0].sample(n=min(half, (df["is_equal"] == 0).sum()), random_state=seed)
    subset = pd.concat([pos, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    subset.to_csv(tmp.name, index=False)
    print(f"  Subset: {len(pos)} genuínos + {len(neg)} impostores = {len(subset)} pares")
    return Path(tmp.name)


def _get_val_csv(split: int) -> Path:
    # Tenta o padrão usado no eval_lacdip_full
    splits_base = WORKSPACE_ROOT / "data" / "generated_splits"
    for subdir in [
        SPLITS_DIR_TEMPLATE.format(split=split),
        f"eval_split{split}",
        f"sprint3_zsl_val_{split}",
    ]:
        csv = splits_base / subdir / "validation_pairs.csv"
        if csv.exists():
            return csv
    raise FileNotFoundError(
        f"Não encontrei validation_pairs.csv para split={split} em {splits_base}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-user",        default="Jpcosta90")
    p.add_argument("--split",          type=int, default=0)
    p.add_argument("--base-image-dir", required=True,
                   help="Diretório raiz das imagens LA-CDIP (ex: /mnt/data/la-cdip/data)")
    p.add_argument("--device",         default="cuda:0")
    p.add_argument("--batch-size",     type=int, default=4)
    p.add_argument("--variants",       nargs="*", default=None,
                   help="Variantes a avaliar (ex: '' nq2). Default: todas.")
    p.add_argument("--cache-dir",      default=None,
                   help="Diretório local para cache do HF (default: ~/.cache/huggingface)")
    p.add_argument("--max-pairs",      type=int, default=None,
                   help="Subset balanceado para teste rápido (ex: 200 → 100 gen + 100 imp)")
    args = p.parse_args()

    wanted = set(args.variants) if args.variants is not None else set(VARIANTS.keys())
    val_csv = _get_val_csv(args.split)
    if args.max_pairs:
        val_csv = _subsample_csv(val_csv, args.max_pairs)
    print(f"CSV de validação: {val_csv}")
    print(f"Split: {args.split}  |  Device: {args.device}  |  Batch: {args.batch_size}")
    print()

    print("Carregando backbone InternVL3-2B (uma vez para todas as variantes)...")
    backbone, tokenizer = _build_backbone(args.device)
    print()

    results: list[dict] = []

    for variant, repo_suffix in VARIANTS.items():
        if variant not in wanted:
            continue

        label   = LABELS[variant]
        repo_id = f"{args.hf_user}/{repo_suffix}"
        print(f"{'='*60}")
        print(f"  Variante : {label}")
        print(f"  Repo HF  : {repo_id}")

        print("  Baixando best_model.pt...")
        try:
            ckpt_path = _download_checkpoint(
                repo_id,
                cache_dir=Path(args.cache_dir) if args.cache_dir else None,
            )
        except Exception as e:
            print(f"  ERRO ao baixar: {e}")
            results.append({"variant": label, "eer": None, "error": str(e)})
            continue
        print(f"  Checkpoint: {ckpt_path}")

        # Lê config do checkpoint para reconstruir o modelo correto
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg  = raw.get("config", {})

        siam = _build_siam(backbone, tokenizer, args.device, cfg)
        _load_weights(siam, ckpt_path, args.device)

        print("  Rodando eval...")
        metrics = _run_eval(
            siam, val_csv, args.base_image_dir,
            args.device, "subcenter_cosface", args.batch_size,
        )
        eer_pct = metrics["eer"] * 100
        r1      = metrics.get("recall_at_1", 0.0) * 100
        print(f"  EER={eer_pct:.2f}%  |  R@1={r1:.2f}%")
        results.append({"variant": label, "eer": metrics["eer"], "recall_at_1": metrics.get("recall_at_1")})

    # --- Resumo final ---
    print()
    print("=" * 60)
    print(f"  RESUMO — Split {args.split}")
    print("=" * 60)
    header = f"  {'Variante':<40} {'EER':>7}  {'R@1':>7}"
    print(header)
    print("  " + "-" * 58)
    for r in results:
        if r["eer"] is None:
            print(f"  {r['variant']:<40}  ERRO")
        else:
            eer = r["eer"] * 100
            r1  = (r.get("recall_at_1") or 0) * 100
            print(f"  {r['variant']:<40} {eer:>6.2f}%  {r1:>6.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
