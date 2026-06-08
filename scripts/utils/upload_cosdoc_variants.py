#!/usr/bin/env python3
"""
Upload das 4 variantes CosDoc (subcenter_cosface) para o Hugging Face Hub.

Lê data/eval_lacdip_cache.csv, seleciona o checkpoint com menor EER por variante
de pooler e faz upload de cada um para um repositório HF separado.

Variantes e repositórios:
  ""                           → Jpcosta90/cosdoc
  "nq2"                        → Jpcosta90/cosdoc-nq2
  "cross_modal"                → Jpcosta90/cosdoc-cross-modal
  "cross_modal_richprompt_cor" → Jpcosta90/cosdoc-cross-modal-richprompt

Uso (servidor gpds2, na raiz do projeto):
    python scripts/utils/upload_cosdoc_variants.py \\
        --cache data/eval_lacdip_cache.csv \\
        --hf-user Jpcosta90

    # Dry-run (sem enviar):
    python scripts/utils/upload_cosdoc_variants.py \\
        --cache data/eval_lacdip_cache.csv \\
        --hf-user Jpcosta90 \\
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PROMPT = "<image> Analyze this document"

RICHPROMPT_COR = (
    "<image> Analyze the provided document image and give me its visual description"
    " based on: Shapes and Elements: presence of graphical components, tables,"
    " sections, headers, and any other visual elements. Layout Consistency: Evaluate"
    " the spatial arrangement of text blocks, margins, and alignments. Content Type:"
    " Ensure the document types of content (e.g., tables, forms, paragraphs),"
    " regardless of specific wording."
)

# ---------------------------------------------------------------------------
# Variant config
# ---------------------------------------------------------------------------

VARIANT_META: dict[str, dict] = {
    "": {
        "repo_suffix":     "cosdoc",
        "label":           "Attention nq=1 (baseline)",
        "description":     "Standard attention pooler, num_queries=1.",
        "embedding_prompt": DEFAULT_PROMPT,
    },
    "nq2": {
        "repo_suffix":     "cosdoc-nq2",
        "label":           "Attention nq=2",
        "description":     "Multi-query attention pooler, num_queries=2.",
        "embedding_prompt": DEFAULT_PROMPT,
    },
    "cross_modal": {
        "repo_suffix":     "cosdoc-cross-modal",
        "label":           "Cross-Modal Pooler",
        "description":     "Bidirectional cross-modal attention pooler (visual ↔ text).",
        "embedding_prompt": DEFAULT_PROMPT,
    },
    "cross_modal_richprompt_cor": {
        "repo_suffix":     "cosdoc-cross-modal-richprompt",
        "label":           "Cross-Modal + Rich Prompt",
        "description":     (
            "Bidirectional cross-modal pooler trained with a rich visual-description prompt "
            "describing shapes, layout consistency and content type."
        ),
        "embedding_prompt": RICHPROMPT_COR,
    },
}


def _variant_from_run_name(run_name: str) -> str | None:
    """Extracts pooler variant from checkpoint directory name.
    Returns variant string (may be empty for baseline) or None if no match.
    """
    m = re.search(r"_noinit_(.+?)_fase", run_name.lower())
    if not m:
        return None
    return m.group(1).strip("_")


# ---------------------------------------------------------------------------
# Best checkpoint selection
# ---------------------------------------------------------------------------

def _best_per_variant(cache_path: Path, split: int | None = None) -> dict[str, dict]:
    """
    Reads eval cache CSV and returns the row with min EER per pooler variant,
    selecting the best phase for each (variant, split) — i.e. melhor acumulado.
    If split is given, restricts to that split only.
    """
    df = pd.read_csv(cache_path)
    required = {"checkpoint_path", "run_name", "eer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cache CSV missing columns: {missing}")

    df = df.dropna(subset=["checkpoint_path", "eer"])

    if split is not None and "split" in df.columns:
        df = df[df["split"].astype(str) == str(split)]
        if df.empty:
            raise ValueError(f"Nenhum resultado no cache para split={split}.")

    best: dict[str, dict] = {}
    for _, row in df.iterrows():
        run_name = str(row["run_name"])
        if "subcenter_cosface" not in run_name.lower():
            continue
        variant = _variant_from_run_name(run_name)
        if variant is None:
            variant = ""  # old-format run → baseline
        if variant not in VARIANT_META:
            continue
        eer = float(row["eer"])
        # Keep best (minimum) EER across all phases for this variant
        if variant not in best or eer < best[variant]["eer"]:
            best[variant] = {
                "checkpoint_path": Path(row["checkpoint_path"]),
                "eer":             eer,
                "run_name":        run_name,
                "phase":           str(row.get("phase", "unknown")),
                "split":           row.get("split"),
            }
    return best


# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------

def _model_card(
    repo_id: str,
    run_name: str,
    eer: float,
    config: dict,
    variant: str,
) -> str:
    meta        = VARIANT_META[variant]
    proj_out    = config.get("projection_output_dim", "1536")
    cut_layer   = config.get("cut_layer", "27")
    pooler      = config.get("pooler_type", "attention")
    num_queries = config.get("num_queries", 1)
    margin      = config.get("margin", "?")
    scale       = config.get("scale", "?")
    k           = config.get("num_sub_centers", "?")
    prompt      = config.get("embedding_prompt", "<image> Analyze this document")

    return f"""\
---
language: pt
license: mit
tags:
  - document-understanding
  - document-retrieval
  - metric-learning
  - siamese-network
  - internvl
  - cosdoc
datasets:
  - LA-CDIP
metrics:
  - eer
model-index:
  - name: CosDoc ({meta['label']})
    results:
      - task:
          type: document-retrieval
        dataset:
          name: LA-CDIP
          type: la-cdip
        metrics:
          - type: eer
            value: {eer:.4f}
---

# CosDoc — {meta['label']}

**CosDoc** is a visual document embedding model trained with supervised metric learning
and hard-example selection via a Reinforcement Learning professor network.

Pooler variant: **{meta['label']}** — {meta['description']}

## Architecture

| Component | Value |
|---|---|
| Backbone | InternVL3-2B (`OpenGVLab/InternVL3-2B`) |
| Cut layer | {cut_layer} |
| Pooler | {pooler} (num_queries={num_queries}) |
| Embedding dim | {proj_out} |
| Loss | Sub-Center CosFace (m={margin}, s={scale}, k={k}) |
| Embedding prompt | `{prompt[:80]}{'...' if len(str(prompt)) > 80 else ''}` |

## Performance (LA-CDIP, full validation pairs)

| Dataset | EER |
|---|---|
| LA-CDIP (5-fold CV) | **{eer*100:.2f}%** |

Source run: `{run_name}`

## Usage

```python
import torch
from huggingface_hub import hf_hub_download
from cavl_doc.models.backbone_loader import load_model
from cavl_doc.models.modeling_cavl import build_cavl_model

device = "cuda" if torch.cuda.is_available() else "cpu"

# Download fine-tuned weights
ckpt_path = hf_hub_download(repo_id="{repo_id}", filename="best_model.pt")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
cfg  = ckpt["config"]

backbone, _, tokenizer, _, _ = load_model("InternVL3-2B")
model = build_cavl_model(
    backbone=backbone,
    cut_layer=cfg["cut_layer"],
    pooler_type=cfg["pooler_type"],
    num_queries=cfg.get("num_queries", 1),
)
model.pool.load_state_dict(ckpt["siam_pool"])
model.head.load_state_dict(ckpt["siam_head"])
model.eval().to(device)
```

## Citation

```bibtex
@misc{{cosdoc2026,
  title  = {{CosDoc: Cosine-Margin Document Embeddings with RL-guided Hard Mining}},
  author = {{Costa, João Paulo}},
  year   = {{2026}},
  url    = {{https://huggingface.co/{repo_id}}}
}}
```
"""


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def upload_variant(
    repo_id: str,
    checkpoint_path: Path,
    eer: float,
    run_name: str,
    variant: str,
    private: bool,
    dry_run: bool,
) -> None:
    print(f"\n{'='*65}")
    print(f"  Variant    : {VARIANT_META[variant]['label']}")
    print(f"  Run        : {run_name}")
    print(f"  EER        : {eer*100:.2f}%")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Repo HF    : {repo_id}")
    print(f"{'='*65}")

    if dry_run:
        print("  [DRY-RUN] Nada enviado.")
        return

    import os
    from huggingface_hub import HfApi, create_repo

    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN") or True
    api   = HfApi(token=token)

    print("  Criando repositório (se não existir)...")
    try:
        create_repo(repo_id=repo_id, repo_type="model", private=private,
                    exist_ok=True, token=token)
    except Exception as e:
        print(f"  Aviso: {e}. Assumindo que o repo já existe.")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Inject correct embedding_prompt into config (old checkpoints may not have it)
        correct_prompt = VARIANT_META[variant]["embedding_prompt"]
        if config.get("embedding_prompt") != correct_prompt:
            print(f"  Injetando embedding_prompt no config ({correct_prompt[:60]}...)")
            config = {**config, "embedding_prompt": correct_prompt}

        # Weights (inference only — no optimizer/professor)
        inference_ckpt = {
            "epoch":   ckpt.get("epoch"),
            "metrics": ckpt.get("metrics"),
            "config":  config,
            "siam_pool": ckpt["siam_pool"],
            "siam_head": ckpt["siam_head"],
        }
        if "backbone_trainable" in ckpt:
            inference_ckpt["backbone_trainable"] = ckpt["backbone_trainable"]

        weights_path = tmp / "best_model.pt"
        torch.save(inference_ckpt, weights_path)
        print(f"  Pesos (inferência): {weights_path.stat().st_size / 1e6:.1f} MB")

        # Config JSON
        config_path = tmp / "cavl_config.json"
        config_path.write_text(json.dumps(config, indent=2, default=str))

        # Model card
        readme_path = tmp / "README.md"
        readme_path.write_text(_model_card(repo_id, run_name, eer, config, variant))

        print("  Enviando para o HF Hub...")
        for fpath in [weights_path, config_path, readme_path]:
            print(f"    -> {fpath.name}")
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=fpath.name,
                repo_id=repo_id,
                repo_type="model",
                token=token,
            )

    print(f"  ✅ https://huggingface.co/{repo_id}")


# ---------------------------------------------------------------------------
# Fallback: scan checkpoint directory directly
# ---------------------------------------------------------------------------

def _best_per_variant_from_disk(checkpoint_root: Path, split: int | None = None) -> dict[str, dict]:
    """
    Scans checkpoint_root for best_model.pt / best_siam.pt matching
    subcenter_cosface + known variants. Reads EER from checkpoint metadata.
    Selects best phase per variant (min EER across fase1/fase2_*).
    If split is given, restricts to _S{split}_ in the run name.
    """
    best: dict[str, dict] = {}
    seen_dirs: set = set()
    split_tag = f"_S{split}_" if split is not None else None

    for ckpt_name in ["best_model.pt", "best_siam.pt"]:
        for ckpt_path in checkpoint_root.rglob(ckpt_name):
            if ckpt_path.parent in seen_dirs:
                continue
            run_name = ckpt_path.parent.name
            if "subcenter_cosface" not in run_name.lower():
                continue
            if split_tag and split_tag not in run_name:
                continue

            variant = _variant_from_run_name(run_name)
            if variant is None or variant not in VARIANT_META:
                continue

            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                eer = ckpt.get("metrics", {}).get("eer")
                if eer is None:
                    continue
                eer = float(eer)
                seen_dirs.add(ckpt_path.parent)
                # Keep best (min) EER across all phases for this variant
                if variant not in best or eer < best[variant]["eer"]:
                    best[variant] = {
                        "checkpoint_path": ckpt_path,
                        "eer":             eer,
                        "run_name":        run_name,
                        "phase":           "disk",
                        "split":           split,
                    }
            except Exception as e:
                print(f"  [SKIP] {run_name}: {e}")

    return best


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Upload das 4 variantes CosDoc (subcenter_cosface) para o HF Hub."
    )
    p.add_argument("--cache", default="data/eval_lacdip_cache.csv",
                   help="CSV de cache do eval_lacdip_full.py (preferido sobre --checkpoint-root).")
    p.add_argument("--checkpoint-root", default=None,
                   help="Fallback: raiz dos checkpoints quando o cache não existe.")
    p.add_argument("--split", type=int, default=0,
                   help="Split a usar (default: 0). Seleciona o melhor checkpoint desse split.")
    p.add_argument("--hf-user", default="Jpcosta90",
                   help="Username do Hugging Face (default: Jpcosta90).")
    p.add_argument("--variants", default=",".join(VARIANT_META.keys()),
                   help="Variantes a fazer upload (vírgula). Default: todas.")
    p.add_argument("--private", action="store_true", default=False)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cache_path = Path(args.cache)
    wanted_variants = {v.strip() for v in args.variants.split(",") if v.strip() in VARIANT_META}

    if cache_path.exists():
        print(f"Lendo cache: {cache_path}  (split={args.split})")
        best = _best_per_variant(cache_path, split=args.split)
    elif args.checkpoint_root:
        ckpt_root = Path(args.checkpoint_root)
        print(f"Cache não encontrado — varrendo checkpoints em: {ckpt_root}  (split={args.split})")
        best = _best_per_variant_from_disk(ckpt_root, split=args.split)
    else:
        print(f"❌ Cache não encontrado: {cache_path}")
        print("   Passe --checkpoint-root <dir> para usar fallback por disco.")
        sys.exit(1)

    if not best:
        print("❌ Nenhum checkpoint encontrado no cache para subcenter_cosface.")
        sys.exit(1)

    print(f"\nMelhores checkpoints encontrados ({len(best)}/{len(VARIANT_META)} variantes):")
    for v, info in best.items():
        label = VARIANT_META[v]["label"]
        print(f"  [{label}]  EER={info['eer']*100:.2f}%  split={info['split']}  {info['phase']}")
        print(f"    {info['run_name']}")

    for variant, info in best.items():
        if variant not in wanted_variants:
            print(f"\n  Pulando variante '{variant}' (não está em --variants).")
            continue

        if not info["checkpoint_path"].exists():
            print(f"\n  ❌ Checkpoint não encontrado no disco: {info['checkpoint_path']}")
            continue

        repo_id = f"{args.hf_user}/{VARIANT_META[variant]['repo_suffix']}"
        upload_variant(
            repo_id=repo_id,
            checkpoint_path=info["checkpoint_path"],
            eer=info["eer"],
            run_name=info["run_name"],
            variant=variant,
            private=args.private,
            dry_run=args.dry_run,
        )

    print("\n✅ Upload concluído para todas as variantes solicitadas.")


if __name__ == "__main__":
    main()
