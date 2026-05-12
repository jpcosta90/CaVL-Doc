#!/usr/bin/env python3
"""
Zero-shot Top-1 Accuracy evaluation on RVL-CDIP for all methods sequentially:
  - ArcDoc          (Sub-Center ArcFace, InternVL3-2B + trained head, from HuggingFace)
  - InternVL3 out   (InternVL3-2B, mean pool of hidden_states[27])
  - InternVL3 in    (InternVL3-2B, mean pool of hidden_states[0] — embedding layer)
  - Jina-v4         (jinaai/jina-embeddings-v4)
  - Pixel Cosine    (448x448 flattened, cosine similarity)
  - Pixel L2        (448x448 flattened, L2 distance)

Protocol (no ZSL splits needed — model never sees any RVL-CDIP class):
  1. Collect all unique (image_path, class) pairs from all available CSVs.
  2. For each class, sort images by filename (reproducible); first --n-support
     images form the support set, the rest form the query set.
  3. For each model: embed all images, build per-class prototype from support
     set, classify queries by nearest prototype.
  4. Report Top-1 Accuracy overall and per class.
  Models are loaded and unloaded sequentially to avoid OOM.

Usage:
    # All models (default):
    python scripts/evaluation/eval_rvlcdip_top1.py \
        --base-image-dir /mnt/data/zs_rvl_cdip/data \
        --gpu-id 0

    # ArcDoc from local checkpoint, only some models:
    python scripts/evaluation/eval_rvlcdip_top1.py \
        --base-image-dir /mnt/data/zs_rvl_cdip/data \
        --checkpoint-path /mnt/.../best_model.pt \
        --models arcdoc internvl3-out pixel-cosine \
        --gpu-id 0
"""
from __future__ import annotations

import argparse
import contextlib
import csv as _csv
import io
import json
import os
import sys
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Optional

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

EMBEDDING_PROMPT = "<image> Analyze this document"
CUT_LAYER        = 27
MODEL_NAME       = "InternVL3-2B"
PIXEL_SIZE       = 448
DEFAULT_REPO_ID  = "Jpcosta90/cavl-doc-lacdip"

_TRANSFORM_ARCDOC = build_transform(input_size=448)


# ---------------------------------------------------------------------------
# Embedder base class
# ---------------------------------------------------------------------------

class Embedder(ABC):
    """Returns a raw numpy vector for one image. Normalisation is handled by
    the evaluation loop (cosine models) or not applied (L2 model)."""
    name:   str
    metric: str  # "cosine" or "l2"

    @abstractmethod
    def embed(self, img: Image.Image) -> np.ndarray:
        ...

    def cleanup(self) -> None:
        pass


# ---------------------------------------------------------------------------
# ArcDoc embedder
# ---------------------------------------------------------------------------

def _resolve_checkpoint(repo_id: str | None, checkpoint_path: str | None,
                        cache_dir: str | None) -> Path:
    if checkpoint_path:
        p = Path(checkpoint_path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p
    from huggingface_hub import hf_hub_download
    for fname in ["best_model.pt", "best_siam.pt"]:
        try:
            print(f"  Downloading {fname} from {repo_id} ...")
            return Path(hf_hub_download(repo_id=repo_id, filename=fname,
                                        cache_dir=cache_dir))
        except Exception:
            continue
    raise FileNotFoundError(
        f"No checkpoint found in {repo_id} (best_model.pt / best_siam.pt)")


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
        raise ValueError("Cannot recover config from checkpoint.")
    return config


class ArcDocEmbedder(Embedder):
    name   = "ArcDoc"
    metric = "cosine"

    def __init__(self, repo_id: str, checkpoint_path: str | None,
                 hf_cache_dir: str | None, device: str, max_num: int):
        ckpt_path = _resolve_checkpoint(repo_id, checkpoint_path, hf_cache_dir)
        print(f"  Checkpoint: {ckpt_path}")

        print("  Loading backbone InternVL3-2B (bfloat16) ...")
        self._backbone, _, self._tokenizer, _, _ = load_model(
            "InternVL3-2B", load_in_4bit=False)
        self._backbone = self._backbone.to(device)
        self._backbone.img_context_token_id = (
            self._tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>"))

        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        cfg  = _load_config(ckpt_path, ckpt)

        cut_layer = int(cfg.get("cut_layer", 27))
        proj_out  = int(cfg.get("projection_output_dim", 1536))

        self._siam = build_cavl_model(
            backbone=self._backbone,
            cut_layer=cut_layer,
            encode_fn=None,
            pool_dim=1536,
            proj_hidden=4096,
            proj_out=proj_out,
            set_trainable=False,
            tokenizer=self._tokenizer,
            pooler_type=cfg.get("pooler_type", "attention"),
            head_type=cfg.get("head_type", "mlp"),
            num_queries=int(cfg.get("num_queries", 1)),
        )
        if "siam_pool" in ckpt and "siam_head" in ckpt:
            self._siam.pool.load_state_dict(ckpt["siam_pool"])
            self._siam.head.load_state_dict(ckpt["siam_head"])
        elif "model_state_dict" in ckpt:
            self._siam.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            raise ValueError("Checkpoint has no recognised weight keys.")

        self._siam.to(device).eval()
        self._cut_layer = cut_layer
        self._device    = device
        self._max_num   = max_num

        eer_train = ckpt.get("metrics", {}).get("eer")
        if eer_train is not None:
            print(f"  Training EER (LA-CDIP): {eer_train*100:.2f}%")
        print(f"  cut_layer={cut_layer}  max_num={max_num}")

    @torch.no_grad()
    def embed(self, img: Image.Image) -> np.ndarray:
        tiles        = dynamic_preprocess(img, image_size=448, use_thumbnail=True,
                                          max_num=self._max_num)
        pixel_values = torch.stack([_TRANSFORM_ARCDOC(t) for t in tiles]).to(torch.bfloat16)
        inp          = prepare_inputs_for_multimodal_embedding(
            self._backbone, self._tokenizer, pixel_values, EMBEDDING_PROMPT)
        input_ids    = inp["input_ids"].to(self._device)
        pixel_values = inp["pixel_values"].to(self._device, dtype=torch.bfloat16)
        image_flags  = inp["image_flags"].to(self._device)

        result = self._backbone(
            input_ids=input_ids, pixel_values=pixel_values,
            image_flags=image_flags, output_hidden_states=True, return_dict=True)

        hidden = result.hidden_states
        lm     = self._backbone.language_model.model
        idx    = (self._cut_layer + 1
                  if len(hidden) == (len(lm.layers) + 1) else self._cut_layer)
        tokens = hidden[idx]
        pooled = self._siam.pool(tokens, mask=None)
        return self._siam.head(pooled).squeeze(0).float().cpu().numpy()

    def cleanup(self):
        del self._siam, self._backbone
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# InternVL3 unadapted embedder (output or input layer)
# ---------------------------------------------------------------------------

class InternVL3Embedder(Embedder):
    metric = "cosine"

    def __init__(self, layer: str, device: str, max_num: int = 6):
        assert layer in ("input", "output")
        self.name  = f"InternVL3 ({layer})"
        self._layer  = layer
        self._device = device
        self._max_num = max_num

        from cavl_doc.models.backbone_loader import load_model, warm_up_model
        from torchvision import transforms
        print(f"  Loading OpenGVLab/InternVL3-2B (layer={layer}) ...")
        backbone, _, tokenizer, _, _ = load_model(
            model_name=MODEL_NAME, adapter_path=None, load_in_4bit=False,
            projection_output_dim=1536)
        backbone.requires_grad_(False)
        warm_up_model(backbone, tokenizer)
        self._backbone  = backbone
        self._tokenizer = tokenizer
        self._tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def embed(self, img: Image.Image) -> np.ndarray:
        tiles        = dynamic_preprocess(img, image_size=448, use_thumbnail=True,
                                          max_num=self._max_num)
        pixel_values = torch.stack([self._tfm(t) for t in tiles]).to(torch.bfloat16)
        out = prepare_inputs_for_multimodal_embedding(
            self._backbone, self._tokenizer, pixel_values, EMBEDDING_PROMPT)
        input_ids    = out["input_ids"].to(self._device)
        pixel_values = out["pixel_values"].to(self._device, dtype=torch.bfloat16)
        image_flags  = out["image_flags"].to(self._device)

        result = self._backbone(
            input_ids=input_ids, pixel_values=pixel_values,
            image_flags=image_flags, output_hidden_states=True, return_dict=True)

        hidden = result.hidden_states
        if self._layer == "input":
            h = hidden[0]
        else:
            lm  = self._backbone.language_model.model
            idx = CUT_LAYER + 1 if len(hidden) == (len(lm.layers) + 1) else CUT_LAYER
            h   = hidden[idx]

        return h.mean(dim=1).squeeze(0).float().cpu().numpy()

    def cleanup(self):
        del self._backbone
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Jina-v4 embedder
# ---------------------------------------------------------------------------

class JinaV4Embedder(Embedder):
    name   = "Jina-v4"
    metric = "cosine"

    def __init__(self, device: str):
        from transformers import AutoModel
        print("  Loading jinaai/jina-embeddings-v4 ...")
        self._model = AutoModel.from_pretrained(
            "jinaai/jina-embeddings-v4",
            trust_remote_code=True,
            dtype=torch.bfloat16,
        ).to(device).eval()
        self._device = device

    def embed(self, img: Image.Image) -> np.ndarray:
        with contextlib.redirect_stderr(io.StringIO()):
            if hasattr(self._model, "encode_image"):
                vec = self._model.encode_image([img], task="retrieval")
            else:
                vec = self._model.encode([img], task="retrieval.passage",
                                         truncate_dim=None)
        return vec[0].cpu().float().numpy()

    def cleanup(self):
        del self._model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Pixel embedder
# ---------------------------------------------------------------------------

class PixelEmbedder(Embedder):
    def __init__(self, metric: str):
        assert metric in ("cosine", "l2")
        self.name   = f"Pixel ({metric.capitalize()})"
        self.metric = metric

    def embed(self, img: Image.Image) -> np.ndarray:
        return np.array(
            img.resize((PIXEL_SIZE, PIXEL_SIZE)).convert("RGB"),
            dtype=np.float32
        ).flatten()


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _collect_images(data_root: Path) -> dict[str, list[str]]:
    """Returns {class_name: sorted list of unique relative image paths}."""
    class_to_paths: dict[str, set[str]] = defaultdict(set)
    csv_files = (
        list((data_root / "data" / "RVL-CDIP").glob("*.csv"))
        + list((data_root / "data" / "generated_splits").glob(
            "RVL-CDIP_zsl_split_*/*.csv"))
    )
    for csv_file in csv_files:
        with open(csv_file, newline="") as fh:
            for row in _csv.DictReader(fh):
                class_to_paths[row["class_a_name"]].add(row["file_a_path"])
                class_to_paths[row["class_b_name"]].add(row["file_b_path"])
    return {cls: sorted(paths) for cls, paths in class_to_paths.items()}


# ---------------------------------------------------------------------------
# Evaluation (generic, works for any Embedder)
# ---------------------------------------------------------------------------

def _evaluate(embedder: Embedder,
              support: dict[str, list[str]],
              query:   dict[str, list[str]],
              base:    Path) -> dict[str, dict]:
    """
    Returns {class_name: {"correct": int, "total": int, "acc_pct": float}},
    plus an "OVERALL" key.
    """
    all_classes = sorted(support.keys())

    def safe_embed(rel: str) -> np.ndarray | None:
        full = base / rel
        try:
            img = Image.open(full).convert("RGB")
            return embedder.embed(img)
        except Exception as e:
            print(f"\n  [WARN] cannot open {full}: {e}")
            return None

    # --- build prototypes ---
    prototypes: dict[str, np.ndarray] = {}
    for cls in tqdm(all_classes, desc=f"  {embedder.name} | support", ncols=90,
                    leave=False):
        vecs = [v for rel in support[cls] if (v := safe_embed(rel)) is not None]
        if not vecs:
            raise RuntimeError(f"No valid support images for class '{cls}'.")
        proto = np.mean(vecs, axis=0)
        if embedder.metric == "cosine":
            norm = np.linalg.norm(proto)
            proto = proto / norm if norm > 0 else proto
        prototypes[cls] = proto

    proto_matrix = np.stack([prototypes[c] for c in all_classes])  # [C, D]

    # --- classify queries ---
    per_class: dict[str, dict] = {c: {"correct": 0, "total": 0} for c in all_classes}

    for cls in tqdm(all_classes, desc=f"  {embedder.name} | query ", ncols=90,
                    leave=False):
        for rel in query[cls]:
            v = safe_embed(rel)
            if v is None:
                continue
            if embedder.metric == "cosine":
                norm = np.linalg.norm(v)
                v = v / norm if norm > 0 else v
                scores   = proto_matrix @ v          # [C], higher = better
                pred_idx = int(np.argmax(scores))
            else:  # l2
                dists    = np.linalg.norm(proto_matrix - v, axis=1)  # [C]
                pred_idx = int(np.argmin(dists))
            pred_class = all_classes[pred_idx]
            per_class[cls]["correct"] += int(pred_class == cls)
            per_class[cls]["total"]   += 1

    results: dict[str, dict] = {}
    total_c, total_t = 0, 0
    for cls in all_classes:
        c, t = per_class[cls]["correct"], per_class[cls]["total"]
        acc = 100.0 * c / t if t > 0 else 0.0
        results[cls] = {"correct": c, "total": t, "acc_pct": round(acc, 2)}
        total_c += c
        total_t += t
    overall = 100.0 * total_c / total_t if total_t > 0 else 0.0
    results["OVERALL"] = {
        "correct": total_c, "total": total_t, "acc_pct": round(overall, 2)}
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_MODEL_KEYS = ["arcdoc", "internvl3-out", "internvl3-in",
                  "jina-v4", "pixel-cosine", "pixel-l2"]


def _make_embedder(key: str, args, device: str) -> Embedder:
    if key == "arcdoc":
        return ArcDocEmbedder(
            repo_id=args.repo_id,
            checkpoint_path=args.checkpoint_path,
            hf_cache_dir=args.hf_cache_dir,
            device=device,
            max_num=args.max_num,
        )
    if key == "internvl3-out":
        return InternVL3Embedder(layer="output", device=device)
    if key == "internvl3-in":
        return InternVL3Embedder(layer="input", device=device)
    if key == "jina-v4":
        return JinaV4Embedder(device=device)
    if key == "pixel-cosine":
        return PixelEmbedder(metric="cosine")
    if key == "pixel-l2":
        return PixelEmbedder(metric="l2")
    raise ValueError(f"Unknown model key: {key}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Zero-shot Top-1 Accuracy: all methods → RVL-CDIP")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--repo-id",         default=DEFAULT_REPO_ID,
                     help=f"HuggingFace Hub repo for ArcDoc (default: {DEFAULT_REPO_ID})")
    src.add_argument("--checkpoint-path", default=None,
                     help="Local path to ArcDoc best_model.pt")
    p.add_argument("--base-image-dir",    required=True,
                   help="Base directory for RVL-CDIP images")
    p.add_argument("--n-support",         type=int, default=10,
                   help="Support images per class for prototype (default: 10)")
    p.add_argument("--max-num",           type=int, default=12,
                   help="Max patches per image for ArcDoc/InternVL3 (default: 12)")
    p.add_argument("--models",            nargs="+", default=ALL_MODEL_KEYS,
                   choices=ALL_MODEL_KEYS, metavar="MODEL",
                   help=f"Models to evaluate (default: all). Choices: {ALL_MODEL_KEYS}")
    p.add_argument("--gpu-id",            type=int, default=None)
    p.add_argument("--hf-cache-dir",      default=None)
    p.add_argument("--run-label",         default=None)
    args = p.parse_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- collect image catalogue ---
    class_to_paths = _collect_images(WORKSPACE_ROOT)
    all_classes    = sorted(class_to_paths.keys())
    print(f"\nClasses ({len(all_classes)}): {', '.join(all_classes)}")

    # --- support / query split ---
    n_support = args.n_support
    support: dict[str, list[str]] = {}
    query:   dict[str, list[str]] = {}
    for cls, paths in class_to_paths.items():
        if len(paths) <= n_support:
            raise ValueError(
                f"Class '{cls}' has only {len(paths)} images, "
                f"which is <= --n-support {n_support}.")
        support[cls] = paths[:n_support]
        query[cls]   = paths[n_support:]

    n_q = sum(len(v) for v in query.values())
    print(f"Support: {n_support}/class | Query: {n_q} total "
          f"({n_q // len(all_classes)}/class avg)\n")

    base = Path(args.base_image_dir)
    run_label = args.run_label or (
        args.repo_id.split("/")[-1] if (args.repo_id and not args.checkpoint_path)
        else Path(args.checkpoint_path).parent.name
    )

    all_results: dict[str, dict] = {}  # model_name → per-class results

    # --- run each model sequentially ---
    for key in args.models:
        print(f"\n{'='*60}")
        print(f"MODEL: {key.upper()}")
        print(f"{'='*60}")
        t0 = time.time()
        embedder = _make_embedder(key, args, device)
        try:
            results = _evaluate(embedder, support, query, base)
        finally:
            embedder.cleanup()

        elapsed = time.time() - t0
        overall = results["OVERALL"]["acc_pct"]
        print(f"  Overall Top-1 Accuracy: {overall:.2f}%  ({elapsed/60:.1f} min)")
        all_results[embedder.name] = results

    # --- combined summary ---
    print(f"\n{'='*60}")
    print(f"SUMMARY — Zero-Shot Top-1 Accuracy @ RVL-CDIP  (n_support={n_support})")
    print(f"{'='*60}")

    model_names = list(all_results.keys())
    col_w = max(len(n) for n in model_names) + 2
    header = f"  {'Class':<30s}" + "".join(f"  {n:>{col_w}}" for n in model_names)
    print(header)
    print("  " + "-" * (30 + (col_w + 2) * len(model_names)))

    for cls in all_classes + ["OVERALL"]:
        if cls == "OVERALL":
            print("  " + "=" * (30 + (col_w + 2) * len(model_names)))
        row = f"  {cls:<30s}" + "".join(
            f"  {all_results[n].get(cls, {}).get('acc_pct', 0.0):>{col_w}.2f}%"
            for n in model_names
        )
        print(row)

    # --- save CSV ---
    import pandas as pd
    rows = []
    for cls in all_classes + ["OVERALL"]:
        r: dict = {"class": cls}
        for name in model_names:
            r[name] = all_results[name].get(cls, {}).get("acc_pct", None)
        rows.append(r)
    df = pd.DataFrame(rows)

    out_dir = WORKSPACE_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"RVL-CDIP_top1_{run_label}_ns{n_support}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n  Results saved to: {out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
