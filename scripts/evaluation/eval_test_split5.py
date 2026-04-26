#!/usr/bin/env python3
"""
Avalia os modelos treinados (best_siam.pt) no split de teste 5 (never-seen).

Para cada checkpoint encontrado no checkpoint_root que corresponda ao padrão
Sprint3/Sprint4, carrega o modelo, roda inferência no split 5 e loga os
resultados no W&B. Ao final, imprime estatísticas agregadas (média, std,
mediana, mín, máx) por loss/fase/experimento.

Uso:
  # Local
  python scripts/evaluation/eval_test_split5.py \
      --data-root /mnt/data/la-cdip \
      --base-image-dir /mnt/data/la-cdip/data

  # UNB
  TMPDIR=/tmp python scripts/evaluation/eval_test_split5.py \
      --data-root /mnt/nas/joaopaulo/LA-CDIP \
      --base-image-dir /mnt/nas/joaopaulo/LA-CDIP/data \
      --checkpoint-root /mnt/nas/joaopaulo/checkpoints
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PREP_SCRIPT = WORKSPACE_ROOT / "scripts" / "utils" / "prepare_protocol_split.py"

WANDB_ENTITY        = "jpcosta1990-university-of-brasilia"
WANDB_PROJECT       = "CaVL-Doc_LA-CDIP_Sprint3_TestSplit5"
WANDB_TRAIN_PROJECT = "CaVL-Doc_LA-CDIP_Sprint3_Staged5x5"
TEST_SPLIT          = 5

# Fixed model architecture (same across all Sprint3/4 runs)
MODEL_NAME         = "InternVL3-2B"
PROJ_OUT_DIM       = 1536
CUT_LAYER          = 27
POOLER_TYPE        = "attention"
HEAD_TYPE          = "mlp"
NUM_QUERIES        = 1

EMBEDDING_PROMPT = "<image> Analyze this document"

KNOWN_LOSSES = [
    "subcenter_cosface", "subcenter_arcface",
    "contrastive", "cosface", "arcface", "triplet", "circle",
]


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def _find_checkpoints(checkpoint_root: Path, name_filter: str) -> List[Path]:
    """Return best_siam.pt files matching name_filter AND containing 'fase' in the name."""
    found = []
    pattern = name_filter.lower()
    for ckpt in checkpoint_root.rglob("best_siam.pt"):
        name = ckpt.parent.name.lower()
        if pattern in name and "fase" in name:
            found.append(ckpt)
    return sorted(found, key=lambda p: p.parent.name)


def _parse_run_name(name: str) -> dict:
    """Extract experiment, loss, split, phase from a checkpoint directory name."""
    info: dict = {"name": name, "experiment": None, "loss": None,
                  "split": None, "phase": None}

    # Experiment
    if name.startswith("Sprint3_"):
        info["experiment"] = "Sprint3"
    elif name.startswith("Sprint4_"):
        info["experiment"] = "Sprint4"
    else:
        return info

    # Split
    m = re.search(r"_S(\d+)_", name)
    if m:
        info["split"] = int(m.group(1))

    # Loss
    name_l = name.lower()
    for loss in sorted(KNOWN_LOSSES, key=len, reverse=True):
        if loss in name_l:
            info["loss"] = loss
            break

    # Phase / mode
    if "fase1" in name_l or ("prof_off" in name_l and "_e10" in name_l):
        info["phase"] = "fase1"
    elif "fase2_profon" in name_l or "prof_on" in name_l:
        info["phase"] = "fase2_profON"
    elif "fase2_profoff" in name_l:
        info["phase"] = "fase2_profOFF"
    elif "transfer" in name_l:
        info["phase"] = "transfer"
    elif "direct" in name_l:
        info["phase"] = "direct"

    return info


# ---------------------------------------------------------------------------
# Split 5 validation data
# ---------------------------------------------------------------------------

def _prepare_test_split(data_root: str, output_base: Path) -> Path:
    """Prepare split-5 validation pairs CSV (all other splits used as train)."""
    split_dir = output_base / f"eval_test_split{TEST_SPLIT}"
    val_csv   = split_dir / "validation_pairs.csv"
    if val_csv.exists():
        return split_dir

    cmd = [
        sys.executable, str(PREP_SCRIPT),
        "--data-root",     data_root,
        "--output-dir",    str(split_dir),
        "--val-split-idx", str(TEST_SPLIT),
        "--protocol",      "zsl",
    ]
    print(f"[PREP] Preparando split {TEST_SPLIT}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    if not val_csv.exists():
        raise FileNotFoundError(f"validation_pairs.csv não criado em {split_dir}")
    return split_dir


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _build_backbone(device: str):
    """Load backbone once; reused across all checkpoints."""
    from cavl_doc.models.backbone_loader import load_model, warm_up_model
    backbone, _, tokenizer, _, _ = load_model(
        model_name=MODEL_NAME,
        adapter_path=None,
        load_in_4bit=False,
        projection_output_dim=PROJ_OUT_DIM,
    )
    backbone.requires_grad_(False)
    warm_up_model(backbone, tokenizer)
    return backbone, tokenizer


def _build_siam(backbone, tokenizer, device: str):
    """Build the Siamese model shell (no weights loaded)."""
    from cavl_doc.models.modeling_cavl import build_cavl_model
    from cavl_doc.utils.embedding_utils import prepare_inputs_for_multimodal_embedding

    def _encode_fn(backbone, images, cut_layer=CUT_LAYER, **kwargs):
        if isinstance(images, torch.Tensor) and images.dim() == 5:
            images_list = [images[i] for i in range(images.shape[0])]
        elif isinstance(images, torch.Tensor):
            images_list = [images]
        else:
            images_list = images

        input_ids_list, pixel_values_list, image_flags_list = [], [], []
        for img in images_list:
            out = prepare_inputs_for_multimodal_embedding(backbone, tokenizer, img, EMBEDDING_PROMPT)
            input_ids_list.append(out["input_ids"][0])
            pixel_values_list.append(out["pixel_values"])
            image_flags_list.append(out["image_flags"])

        max_len = max(len(ids) for ids in input_ids_list)
        pad_id  = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        padded_ids, padded_mask = [], []
        for ids in input_ids_list:
            pad_len = max_len - len(ids)
            padded_ids.append(torch.cat([ids, torch.full((pad_len,), pad_id, device=ids.device, dtype=ids.dtype)]))
            padded_mask.append(torch.cat([torch.ones_like(ids), torch.zeros(pad_len, device=ids.device, dtype=ids.dtype)]))

        batch_input_ids    = torch.stack(padded_ids).to(device)
        batch_attn_mask    = torch.stack(padded_mask).to(device)
        batch_pixel_values = torch.cat(pixel_values_list, dim=0).to(device, dtype=torch.bfloat16)
        batch_image_flags  = torch.cat(image_flags_list,  dim=0).to(device)

        out = backbone(
            input_ids=batch_input_ids,
            attention_mask=batch_attn_mask,
            pixel_values=batch_pixel_values,
            image_flags=batch_image_flags,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = out.hidden_states
        lm  = backbone.language_model.model
        idx = cut_layer + 1 if len(hidden_states) == (len(lm.layers) + 1) else cut_layer
        return hidden_states[idx], None

    return build_cavl_model(
        backbone=backbone,
        cut_layer=CUT_LAYER,
        encode_fn=_encode_fn,
        pool_dim=PROJ_OUT_DIM,
        proj_hidden=4096,
        proj_out=PROJ_OUT_DIM,
        set_trainable=False,
        tokenizer=tokenizer,
        pooler_type=POOLER_TYPE,
        head_type=HEAD_TYPE,
        num_queries=NUM_QUERIES,
    ).to(device)


def _load_weights(siam, ckpt_path: Path, device: str) -> None:
    """Load pool/head/backbone weights into an existing siam model."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "siam_pool" in ckpt:
        siam.pool.load_state_dict(ckpt["siam_pool"])
    if "siam_head" in ckpt:
        siam.head.load_state_dict(ckpt["siam_head"])
    if "backbone_trainable" in ckpt and ckpt["backbone_trainable"]:
        siam.backbone.load_state_dict(ckpt["backbone_trainable"], strict=False)
    siam.eval()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _make_loader(val_csv: Path, base_image_dir: str):
    from torch.utils.data import DataLoader
    from cavl_doc.data.dataset import DocumentPairDataset

    dataset = DocumentPairDataset(
        csv_path=str(val_csv),
        base_dir=base_image_dir,
        input_size=448,
        max_num=12,
        device="cpu",
    )

    def _collate(batch):
        imgs_a  = [s["image_a"]  for s in batch]
        imgs_b  = [s["image_b"]  for s in batch]
        labels  = torch.tensor([int(s["label"])   for s in batch], dtype=torch.long)
        cls_a   = torch.tensor([int(s["class_a"]) for s in batch], dtype=torch.long)
        cls_b   = torch.tensor([int(s["class_b"]) for s in batch], dtype=torch.long)
        return imgs_a, imgs_b, labels, cls_a, cls_b

    loader = DataLoader(dataset, batch_size=12, shuffle=False,
                        num_workers=0, collate_fn=_collate)
    return dataset, loader


def _run_eval(siam, val_csv: Path, base_image_dir: str, device: str,
              loss_type: str) -> dict:
    from cavl_doc.modules.losses import build_loss
    from cavl_doc.trainers.rl_trainer import validate_siam_on_loader

    dataset, loader = _make_loader(val_csv, base_image_dir)
    num_classes = getattr(dataset, "num_classes", max(100, len(dataset) // 10))

    criterion = build_loss(
        loss_type,
        margin=0.35, scale=24.0, num_sub_centers=2,
        num_classes=num_classes, embedding_dim=PROJ_OUT_DIM,
        std=0.05,
    ).to(device)
    criterion.eval()

    vloss, veer, vthr, vr1, v_batch_recall = validate_siam_on_loader(
        siam, loader, device, criterion,
    )
    return {
        "eer":          float(veer),
        "loss":         float(vloss),
        "recall_at_1":  float(vr1),
        "threshold":    float(vthr),
        "batch_recall": float(v_batch_recall),
    }


# ---------------------------------------------------------------------------
# W&B logging
# ---------------------------------------------------------------------------

def _log_wandb(run_info: dict, metrics: dict, wandb_entity: str,
               wandb_project: str) -> None:
    try:
        import wandb
        run_name = (
            f"Test5_{run_info['experiment']}_S{run_info['split']}"
            f"_{run_info['loss']}_{run_info['phase']}"
        )
        run = wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            name=run_name,
            config={**run_info, "test_split": TEST_SPLIT},
            reinit=True,
        )
        wandb.log({
            "test/eer":          metrics["eer"],
            "test/loss":         metrics["loss"],
            "test/recall_at_1":  metrics["recall_at_1"],
            "test/threshold":    metrics["threshold"],
            "test/batch_recall": metrics["batch_recall"],
        })
        run.finish()
        print(f"  ✅ W&B logged: {run_name}")
    except Exception as e:
        print(f"  ⚠️  W&B log falhou: {e}")


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def _compute_accumulated(df) -> "pd.DataFrame":
    """
    For each (experiment, loss, split), compute accumulated best EER:
      accumulated_profOFF = min(fase1_eer, fase2_profOFF_eer)
      accumulated_profON  = min(fase1_eer, fase2_profON_eer)

    Returns a DataFrame with columns:
      experiment, loss, split, accum_mode, eer
    where accum_mode is 'accum_profOFF' or 'accum_profON'.
    """
    import pandas as pd

    fase1   = df[df["phase"] == "fase1"].set_index(["experiment", "loss", "split"])["eer"]
    p2off   = df[df["phase"] == "fase2_profOFF"].set_index(["experiment", "loss", "split"])["eer"]
    p2on    = df[df["phase"] == "fase2_profON"].set_index(["experiment", "loss", "split"])["eer"]

    rows = []
    all_keys = set(fase1.index) | set(p2off.index) | set(p2on.index)
    for key in sorted(all_keys):
        exp, loss, split = key
        f1 = fase1.get(key)

        for mode, p2 in [("accum_profOFF", p2off), ("accum_profON", p2on)]:
            p2v = p2.get(key)
            candidates = [v for v in (f1, p2v) if v is not None]
            if not candidates:
                continue
            rows.append({
                "experiment": exp, "loss": loss, "split": split,
                "accum_mode": mode, "eer": min(candidates),
            })

    return pd.DataFrame(rows)


def _print_stats(results: List[dict], wandb_entity: str, wandb_project: str,
                 log_wandb: bool = True) -> None:
    if not results:
        print("Nenhum resultado para agregar.")
        return

    import pandas as pd
    df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("RESULTADOS POR FASE — Split 5 (Teste)")
    print("=" * 80)

    for (exp, loss, phase), grp in df.groupby(["experiment", "loss", "phase"]):
        eers = grp["eer"].values * 100
        print(f"\n  [{exp}] loss={loss} phase={phase}  (n={len(eers)} splits)")
        print(f"    Média:   {eers.mean():.2f}%")
        print(f"    Std:     {eers.std():.2f} pp")
        print(f"    Mediana: {np.median(eers):.2f}%")
        print(f"    Mín:     {eers.min():.2f}%")
        print(f"    Máx:     {eers.max():.2f}%")

    # Accumulated best (same logic as HTML report)
    df_accum = _compute_accumulated(df)

    if not df_accum.empty:
        print("\n" + "=" * 80)
        print("ACUMULADO MELHOR (min fase1, fase2) — Split 5 (Teste)")
        print("=" * 80)

        for (exp, loss, mode), grp in df_accum.groupby(["experiment", "loss", "accum_mode"]):
            eers = grp["eer"].values * 100
            print(f"\n  [{exp}] loss={loss} {mode}  (n={len(eers)} splits)")
            print(f"    Média:   {eers.mean():.2f}%")
            print(f"    Std:     {eers.std():.2f} pp")
            print(f"    Mediana: {np.median(eers):.2f}%")
            print(f"    Mín:     {eers.min():.2f}%")
            print(f"    Máx:     {eers.max():.2f}%")

    print("\n" + "=" * 80)

    if not log_wandb:
        return

    try:
        import wandb

        # Per-phase aggregates
        for (exp, loss, phase), grp in df.groupby(["experiment", "loss", "phase"]):
            eers = grp["eer"].values
            agg_run = wandb.init(
                entity=wandb_entity,
                project=wandb_project,
                name=f"Agg_{exp}_{loss}_{phase}",
                config={"type": "aggregate", "experiment": exp,
                        "loss": loss, "phase": phase, "n_splits": len(eers)},
                reinit=True,
            )
            wandb.log({
                "agg/eer_mean":   float(eers.mean()),
                "agg/eer_std":    float(eers.std()) if len(eers) > 1 else 0.0,
                "agg/eer_median": float(np.median(eers)),
                "agg/eer_min":    float(eers.min()),
                "agg/eer_max":    float(eers.max()),
                "agg/n_splits":   len(eers),
            })
            agg_run.finish()

        # Accumulated best aggregates
        for (exp, loss, mode), grp in df_accum.groupby(["experiment", "loss", "accum_mode"]):
            eers = grp["eer"].values
            agg_run = wandb.init(
                entity=wandb_entity,
                project=wandb_project,
                name=f"Accum_{exp}_{loss}_{mode}",
                config={"type": "accumulated", "experiment": exp,
                        "loss": loss, "accum_mode": mode, "n_splits": len(eers)},
                reinit=True,
            )
            wandb.log({
                "accum/eer_mean":   float(eers.mean()),
                "accum/eer_std":    float(eers.std()) if len(eers) > 1 else 0.0,
                "accum/eer_median": float(np.median(eers)),
                "accum/eer_min":    float(eers.min()),
                "accum/eer_max":    float(eers.max()),
                "accum/n_splits":   len(eers),
            })
            agg_run.finish()

    except Exception as e:
        print(f"⚠️  W&B aggregate log falhou: {e}")


def _print_stats_accum(results: List[dict], wandb_entity: str, wandb_project: str,
                       log_wandb: bool = True) -> None:
    """Print and optionally log aggregated accumulated-best EER stats."""
    if not results:
        print("Nenhum resultado para agregar.")
        return

    import pandas as pd
    df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("ACUMULADO MELHOR — Split 5 (Teste)")
    print("=" * 80)

    for (exp, loss, mode), grp in df.groupby(["experiment", "loss", "accum_mode"]):
        eers = grp["eer"].values * 100
        print(f"\n  [{exp}] loss={loss} {mode}  (n={len(eers)} splits)")
        print(f"    Média:   {eers.mean():.2f}%")
        print(f"    Std:     {eers.std():.2f} pp")
        print(f"    Mediana: {np.median(eers):.2f}%")
        print(f"    Mín:     {eers.min():.2f}%")
        print(f"    Máx:     {eers.max():.2f}%")

    print("\n" + "=" * 80)

    if not log_wandb:
        return

    try:
        import wandb
        for (exp, loss, mode), grp in df.groupby(["experiment", "loss", "accum_mode"]):
            eers = grp["eer"].values
            agg_run = wandb.init(
                entity=wandb_entity,
                project=wandb_project,
                name=f"Agg_{exp}_{loss}_{mode}",
                config={"type": "accumulated_aggregate", "experiment": exp,
                        "loss": loss, "accum_mode": mode, "n_splits": len(eers)},
                reinit=True,
            )
            wandb.log({
                "accum/eer_mean":   float(eers.mean()),
                "accum/eer_std":    float(eers.std()) if len(eers) > 1 else 0.0,
                "accum/eer_median": float(np.median(eers)),
                "accum/eer_min":    float(eers.min()),
                "accum/eer_max":    float(eers.max()),
                "accum/n_splits":   len(eers),
            })
            agg_run.finish()
    except Exception as e:
        print(f"⚠️  W&B aggregate log falhou: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Avalia modelos treinados no split de teste 5."
    )
    p.add_argument("--data-root",       required=True,
                   help="Raiz do dataset LA-CDIP (contém protocol/)")
    p.add_argument("--base-image-dir",  required=True,
                   help="Diretório base das imagens LA-CDIP")
    p.add_argument("--checkpoint-root", default=None,
                   help="Raiz dos checkpoints (default: /mnt/large/checkpoints ou ./checkpoints)")
    p.add_argument("--filter",          default="Sprint3",
                   help="Filtro de nome para os checkpoints (default: Sprint3)")
    p.add_argument("--wandb-entity",         default=WANDB_ENTITY)
    p.add_argument("--wandb-project",        default=WANDB_PROJECT)
    p.add_argument("--train-wandb-project",  default=WANDB_TRAIN_PROJECT,
                   help="Projeto W&B do treino Sprint3 (para buscar EERs de treino)")
    p.add_argument("--no-wandb",        action="store_true")
    p.add_argument("--gpu-id",          type=int, default=None)
    p.add_argument("--dry-run",         action="store_true")
    args = p.parse_args()

    # GPU
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Checkpoint root
    if args.checkpoint_root:
        ckpt_root = Path(args.checkpoint_root)
    elif Path("/mnt/large/checkpoints").exists():
        ckpt_root = Path("/mnt/large/checkpoints")
    elif Path("/mnt/nas/joaopaulo/checkpoints").exists():
        ckpt_root = Path("/mnt/nas/joaopaulo/checkpoints")
    else:
        ckpt_root = WORKSPACE_ROOT / "checkpoints"
    print(f"Checkpoint root: {ckpt_root}")

    # Prepare split 5 data
    split_dir = _prepare_test_split(
        args.data_root,
        WORKSPACE_ROOT / "data" / "generated_splits",
    )
    val_csv = split_dir / "validation_pairs.csv"
    print(f"Split 5 val CSV: {val_csv}  ({sum(1 for _ in open(val_csv))-1} pares)")

    # Find checkpoints
    checkpoints = _find_checkpoints(ckpt_root, args.filter)
    print(f"\nCheckpoints encontrados: {len(checkpoints)}")
    for c in checkpoints:
        print(f"  {c.parent.name}")

    if not checkpoints:
        print("Nenhum checkpoint encontrado. Verifique --checkpoint-root e --filter.")
        sys.exit(1)

    if args.dry_run:
        print("\n[DRY-RUN] Encerrando sem rodar inferência.")
        return

    # ------------------------------------------------------------------
    # Fetch training val/best_eer from W&B to select accumulated best
    # ------------------------------------------------------------------
    print(f"\nBuscando EERs de treino no W&B ({args.train_wandb_project})...")
    train_eers: Dict[str, float] = {}
    try:
        import wandb
        api = wandb.Api()
        runs = api.runs(f"{args.wandb_entity}/{args.train_wandb_project}")
        for r in runs:
            eer = None
            for key in ("val/best_eer", "val/eer"):
                v = r.summary.get(key)
                if v is not None:
                    try:
                        eer = float(v)
                        break
                    except (TypeError, ValueError):
                        if hasattr(v, "get"):
                            for sub in ("min", "last"):
                                c = v.get(sub)
                                if c is not None:
                                    try:
                                        eer = float(c)
                                        break
                                    except (TypeError, ValueError):
                                        pass
                        if eer is not None:
                            break
            if eer is not None:
                train_eers[r.name] = eer
        print(f"  {len(train_eers)} runs com EER de treino encontradas.")
    except Exception as e:
        print(f"  ⚠️  Não foi possível buscar EERs de treino: {e}")
        print("  Fallback: usando fase2 como acumulado (sem comparar com fase1).")

    # ------------------------------------------------------------------
    # Group checkpoints by (loss, split) and select accumulated best
    # ------------------------------------------------------------------
    # Index: (loss, split) -> {phase -> ckpt_path}
    groups: Dict[tuple, dict] = {}
    for ckpt_path in checkpoints:
        info = _parse_run_name(ckpt_path.parent.name)
        if None in (info["experiment"], info["loss"], info["split"], info["phase"]):
            continue
        key = (info["loss"], info["split"])
        groups.setdefault(key, {})[info["phase"]] = ckpt_path

    # For each (loss, split): pick best accumulated checkpoint for profOFF and profON
    to_eval: List[dict] = []  # {ckpt_path, loss, split, accum_mode}
    for (loss, split), phases in sorted(groups.items()):
        for accum_mode, p2_phase in [("accum_profOFF", "fase2_profOFF"),
                                      ("accum_profON",  "fase2_profON")]:
            p1_path = phases.get("fase1")
            p2_path = phases.get(p2_phase)

            if p2_path is None:
                if p1_path is not None:
                    chosen, chosen_phase = p1_path, "fase1"
                else:
                    continue
            elif p1_path is None:
                chosen, chosen_phase = p2_path, p2_phase
            else:
                p1_name = p1_path.parent.name
                p2_name = p2_path.parent.name
                p1_eer  = train_eers.get(p1_name)
                p2_eer  = train_eers.get(p2_name)

                if p1_eer is None and p2_eer is None:
                    chosen, chosen_phase = p2_path, p2_phase
                elif p1_eer is None:
                    chosen, chosen_phase = p2_path, p2_phase
                elif p2_eer is None:
                    chosen, chosen_phase = p1_path, "fase1"
                else:
                    if p1_eer <= p2_eer:
                        chosen, chosen_phase = p1_path, "fase1"
                    else:
                        chosen, chosen_phase = p2_path, p2_phase

            to_eval.append({
                "ckpt_path":    chosen,
                "loss":         loss,
                "split":        split,
                "accum_mode":   accum_mode,
                "chosen_phase": chosen_phase,
                "experiment":   "Sprint3",
            })

    print(f"\nCheckpoints selecionados para inferência: {len(to_eval)}")
    for e in to_eval:
        print(f"  [{e['accum_mode']}] loss={e['loss']} split={e['split']} "
              f"→ {e['chosen_phase']}  ({e['ckpt_path'].parent.name})")

    # Load backbone once — reused for every checkpoint
    print("\nCarregando backbone (uma vez)...")
    backbone, tokenizer = _build_backbone(device)
    siam = _build_siam(backbone, tokenizer, device)

    all_results = []
    for entry in to_eval:
        ckpt_path   = entry["ckpt_path"]
        loss_type   = entry["loss"]
        accum_mode  = entry["accum_mode"]

        print(f"\n{'─'*70}")
        print(f"[EVAL] {ckpt_path.parent.name}")
        print(f"       loss={loss_type} split={entry['split']} "
              f"mode={accum_mode} (from {entry['chosen_phase']})")

        try:
            t0 = time.time()
            _load_weights(siam, ckpt_path, device)
            metrics = _run_eval(siam, val_csv, args.base_image_dir, device, loss_type)
            elapsed = time.time() - t0

            print(f"  EER={metrics['eer']*100:.2f}%  "
                  f"R@1={metrics['recall_at_1']*100:.2f}%  "
                  f"({elapsed:.0f}s)")

            result = {
                "experiment":   entry["experiment"],
                "loss":         loss_type,
                "split":        entry["split"],
                "accum_mode":   accum_mode,
                "chosen_phase": entry["chosen_phase"],
                **metrics,
            }
            all_results.append(result)

            if not args.no_wandb:
                try:
                    import wandb
                    run_name = (f"Test5_{entry['experiment']}_S{entry['split']}"
                                f"_{loss_type}_{accum_mode}")
                    run = wandb.init(
                        entity=args.wandb_entity,
                        project=args.wandb_project,
                        name=run_name,
                        config={**entry, "test_split": TEST_SPLIT,
                                "ckpt": ckpt_path.parent.name},
                        reinit=True,
                    )
                    wandb.log({
                        "test/eer":          metrics["eer"],
                        "test/loss":         metrics["loss"],
                        "test/recall_at_1":  metrics["recall_at_1"],
                        "test/threshold":    metrics["threshold"],
                        "test/batch_recall": metrics["batch_recall"],
                    })
                    run.finish()
                    print(f"  W&B logged: {run_name}")
                except Exception as e:
                    print(f"  ⚠️  W&B log falhou: {e}")

        except Exception as e:
            print(f"  ❌ Erro: {e}")
            import traceback; traceback.print_exc()

    _print_stats_accum(all_results, args.wandb_entity, args.wandb_project,
                       log_wandb=not args.no_wandb)


if __name__ == "__main__":
    main()
