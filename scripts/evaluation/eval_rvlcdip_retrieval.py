#!/usr/bin/env python3
"""
Retrieval-based evaluation on RVL-CDIP: Recall@1 and Recall@5.

Protocol:
  1. For each class, take the first --n-gallery images as the gallery.
  2. Embed all gallery images with each model.
  3. For each query image, rank the rest of the gallery by similarity
     (self excluded). Recall@K = fraction of queries whose top-K contains
     at least one image from the same class.
  4. Report per-class and overall Recall@1 / Recall@5.

No prototypes, no classifiers — the embedding space does all the work.

Usage:
    python scripts/evaluation/eval_rvlcdip_retrieval.py \
        --base-image-dir /mnt/data/zs_rvl_cdip/data \
        --gpu-id 0

    # Larger gallery, specific models:
    python scripts/evaluation/eval_rvlcdip_retrieval.py \
        --base-image-dir /mnt/data/zs_rvl_cdip/data \
        --n-gallery 200 \
        --models arcdoc jina-v4 \
        --gpu-id 0
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

# Re-use all Embedder classes and helpers from the top-1 script
sys.path.insert(0, str(Path(__file__).parent))
from eval_rvlcdip_top1 import (  # noqa: E402
    ALL_MODEL_KEYS,
    DEFAULT_REPO_ID,
    _collect_images,
    _delete_model_cache,
    _make_embedder,
)


# ---------------------------------------------------------------------------
# Retrieval evaluation
# ---------------------------------------------------------------------------

def _embed_gallery(embedder, gallery_items: list[tuple[str, str]],
                   base: Path) -> tuple[np.ndarray, list[str]]:
    """
    Embed all gallery images. Returns (E, labels) where E is [N, D] float32
    (already L2-normalised for cosine models) and labels is a list of class
    names aligned with the rows of E.
    """
    embeddings, labels = [], []
    for rel, cls in tqdm(gallery_items,
                         desc=f"  {embedder.name} | embed", ncols=90, leave=False):
        full = base / rel
        try:
            img = Image.open(full).convert("RGB")
            v = embedder.embed(img).astype(np.float32)
            if embedder.metric == "cosine":
                norm = np.linalg.norm(v)
                if norm > 0:
                    v /= norm
            embeddings.append(v)
            labels.append(cls)
        except Exception as e:
            print(f"\n  [WARN] skipping {full}: {e}")

    return np.stack(embeddings), labels  # [N, D], [N]


def _sim_matrix(E: np.ndarray, metric: str) -> np.ndarray:
    """
    Compute [N, N] similarity matrix (higher = more similar).
    Cosine: dot product (vectors already normalised).
    L2:     negative squared distance.
    """
    if metric == "cosine":
        return E @ E.T
    # L2: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b  (memory-efficient)
    sq = (E ** 2).sum(axis=1)
    return -(sq[:, None] + sq[None, :] - 2.0 * (E @ E.T))


def _evaluate_retrieval(embedder,
                        gallery_items: list[tuple[str, str]],
                        base: Path,
                        k_vals: list[int] = (1, 5)) -> dict[str, dict]:
    """
    Returns {class_name: {"R@1": float, "R@5": float, "total": int}}
    plus an "OVERALL" key.
    """
    E, labels = _embed_gallery(embedder, gallery_items, base)
    labels_arr = np.array(labels)
    all_classes = sorted(set(labels))

    S = _sim_matrix(E, embedder.metric)          # [N, N]
    np.fill_diagonal(S, -np.inf)                 # exclude self-match

    per_class: dict[str, dict] = {
        cls: {f"R@{k}": 0 for k in k_vals} | {"total": 0}
        for cls in all_classes
    }

    for i, cls in enumerate(tqdm(labels, desc=f"  {embedder.name} | recall",
                                 ncols=90, leave=False)):
        ranked = np.argsort(S[i])[::-1]          # descending similarity
        for k in k_vals:
            if cls in labels_arr[ranked[:k]]:
                per_class[cls][f"R@{k}"] += 1
        per_class[cls]["total"] += 1

    results: dict[str, dict] = {}
    totals = {f"R@{k}": 0 for k in k_vals}
    total_n = 0

    for cls in all_classes:
        d = per_class[cls]
        t = d["total"]
        entry = {"total": t}
        for k in k_vals:
            entry[f"R@{k}"] = round(100.0 * d[f"R@{k}"] / t, 2) if t > 0 else 0.0
            totals[f"R@{k}"] += d[f"R@{k}"]
        results[cls] = entry
        total_n += t

    overall = {"total": total_n}
    for k in k_vals:
        overall[f"R@{k}"] = round(100.0 * totals[f"R@{k}"] / total_n, 2) if total_n > 0 else 0.0
    results["OVERALL"] = overall
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Retrieval Recall@K evaluation on RVL-CDIP")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--repo-id",         default=DEFAULT_REPO_ID)
    src.add_argument("--checkpoint-path", default=None)
    p.add_argument("--base-image-dir",    required=True)
    p.add_argument("--n-gallery",         type=int, default=100,
                   help="Gallery images per class (default: 100). "
                        "Full sim matrix = (n_gallery*n_classes)^2 — keep ≤200 "
                        "for GPU memory safety.")
    p.add_argument("--max-num",           type=int, default=12)
    p.add_argument("--models",            nargs="+", default=ALL_MODEL_KEYS,
                   choices=ALL_MODEL_KEYS, metavar="MODEL")
    p.add_argument("--gpu-id",            type=int, default=None)
    p.add_argument("--hf-cache-dir",      default=None)
    p.add_argument("--run-label",         default=None)
    args = p.parse_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- build gallery ---
    class_to_paths = _collect_images(WORKSPACE_ROOT)
    all_classes    = sorted(class_to_paths.keys())
    print(f"\nClasses ({len(all_classes)}): {', '.join(all_classes)}")

    n_gallery = args.n_gallery
    for cls, paths in class_to_paths.items():
        if len(paths) < n_gallery:
            raise ValueError(
                f"Class '{cls}' has only {len(paths)} images "
                f"(need --n-gallery {n_gallery}).")

    gallery_items: list[tuple[str, str]] = []
    for cls in all_classes:
        for rel in class_to_paths[cls][:n_gallery]:
            gallery_items.append((rel, cls))

    n_total = len(gallery_items)
    print(f"Gallery: {n_gallery}/class  ×  {len(all_classes)} classes  =  {n_total} images")
    mem_mb  = n_total ** 2 * 4 / 1024 / 1024
    print(f"Sim-matrix footprint (float32): ~{mem_mb:.0f} MB\n")

    base = Path(args.base_image_dir)
    run_label = args.run_label or (
        args.repo_id.split("/")[-1] if (args.repo_id and not args.checkpoint_path)
        else Path(args.checkpoint_path).parent.name
    )

    all_results: dict[str, dict] = {}

    hf_id_last_use: dict[str, int] = {}
    for i, key in enumerate(args.models):
        ids_for_key = {
            "arcdoc":        ["OpenGVLab/InternVL3-2B"]
                             + ([] if args.checkpoint_path else [args.repo_id]),
            "internvl3-out": ["OpenGVLab/InternVL3-2B"],
            "internvl3-in":  ["OpenGVLab/InternVL3-2B"],
            "jina-v4":       ["jinaai/jina-embeddings-v4"],
            "pixel-cosine":  [],
            "pixel-l2":      [],
        }.get(key, [])
        for hf_id in ids_for_key:
            hf_id_last_use[hf_id] = i

    for i, key in enumerate(args.models):
        print(f"\n{'='*60}")
        print(f"MODEL: {key.upper()}")
        print(f"{'='*60}")
        t0 = time.time()
        embedder = _make_embedder(key, args, device)
        try:
            results = _evaluate_retrieval(embedder, gallery_items, base)
        finally:
            embedder.cleanup()
            for hf_id in embedder.hf_model_ids:
                if hf_id_last_use.get(hf_id) == i:
                    _delete_model_cache(hf_id)

        elapsed = time.time() - t0
        ov = results["OVERALL"]
        print(f"  Overall  R@1={ov['R@1']:.2f}%  R@5={ov['R@5']:.2f}%  "
              f"({elapsed/60:.1f} min)")
        all_results[embedder.name] = results

    # --- print summary table ---
    model_names = list(all_results.keys())
    print(f"\n{'='*60}")
    print(f"SUMMARY — Retrieval Recall@K @ RVL-CDIP  (n_gallery={n_gallery})")
    print(f"{'='*60}")

    metrics = ["R@1", "R@5"]
    col_heads = [f"{n} {m}" for n in model_names for m in metrics]
    col_w = max(len(h) for h in col_heads) + 2
    header = f"  {'Class':<28}" + "".join(f"  {h:>{col_w}}" for h in col_heads)
    print(header)
    print("  " + "-" * (28 + (col_w + 2) * len(col_heads)))

    for cls in all_classes + ["OVERALL"]:
        if cls == "OVERALL":
            print("  " + "=" * (28 + (col_w + 2) * len(col_heads)))
        row = f"  {cls:<28}"
        for name in model_names:
            for m in metrics:
                val = all_results[name].get(cls, {}).get(m, 0.0)
                row += f"  {val:>{col_w}.2f}%"
        print(row)

    # --- save CSV ---
    rows = []
    for cls in all_classes + ["OVERALL"]:
        r: dict = {"class": cls}
        for name in model_names:
            for m in metrics:
                col = f"{name}_{m.replace('@', 'at')}"
                r[col] = all_results[name].get(cls, {}).get(m, None)
        rows.append(r)
    df = pd.DataFrame(rows)

    out_dir = WORKSPACE_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"RVL-CDIP_retrieval_{run_label}_ng{n_gallery}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n  Results saved to: {out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
