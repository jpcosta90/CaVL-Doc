#!/usr/bin/env python3
"""
Avaliação nos splits de validação 0-4 do LA-CDIP usando os CSVs COMPLETOS.

Diferente do loop de treinamento (que usa um subset de ~1.036 pares), este
script usa todos os pares disponíveis em cada split, gerando EERs comparáveis
com os baselines de embedding (eval_embeddings.py).

Uso (servidor local — Sprint3):
    python scripts/evaluation/eval_lacdip_full.py \\
        --checkpoint-root /mnt/large/checkpoints \\
        --run-prefix Sprint3_ \\
        --base-image-dir /mnt/data/la-cdip/data

Uso (gpds2 — Sprint3b):
    python scripts/evaluation/eval_lacdip_full.py \\
        --checkpoint-root /mnt/nas/joaopaulo/CaVL-Doc/checkpoints \\
        --run-prefix Sprint3b_ \\
        --base-image-dir /mnt/nas/joaopaulo/LA-CDIP/data \\
        --wandb-project CaVL-Doc_LA-CDIP_FullEval

Ver checkpoints sem rodar:
    python scripts/evaluation/eval_lacdip_full.py \\
        --checkpoint-root /mnt/large/checkpoints \\
        --run-prefix Sprint3_ \\
        --base-image-dir /mnt/data/la-cdip/data \\
        --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

WANDB_ENTITY   = "jpcosta1990-university-of-brasilia"
WANDB_PROJECT  = "CaVL-Doc_LA-CDIP_FullEval"

# Architecture defaults (all Sprint3/3b models share these)
MODEL_NAME   = "InternVL3-2B"
PROJ_OUT_DIM = 1536
CUT_LAYER    = 27
POOLER_TYPE  = "attention"
HEAD_TYPE    = "mlp"
NUM_QUERIES  = 1

EMBEDDING_PROMPT = "<image> Analyze this document"

KNOWN_LOSSES = [
    "subcenter_cosface", "subcenter_arcface",
    "contrastive", "cosface", "arcface", "triplet", "circle",
]

# Template do diretório de splits de validação gerados localmente
SPLITS_DIR_TEMPLATE = "sprint3_zsl_val_{split}_train_excl_5"


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def _find_checkpoints(checkpoint_root: Path, run_prefix: str) -> List[Path]:
    """Retorna best_model.pt (ou best_siam.pt como fallback) cujo diretório
    começa com run_prefix e contém 'fase' no nome."""
    found: List[Path] = []
    seen_dirs: set = set()
    prefix = run_prefix.lower()
    for ckpt_name in ["best_model.pt", "best_siam.pt"]:
        for ckpt in checkpoint_root.rglob(ckpt_name):
            if ckpt.parent in seen_dirs:
                continue
            name = ckpt.parent.name.lower()
            if name.startswith(prefix) and "fase" in name:
                found.append(ckpt)
                seen_dirs.add(ckpt.parent)
    return sorted(found, key=lambda p: p.parent.name)


def _parse_run_name(name: str) -> dict:
    """Extrai sprint, loss, split, phase do nome do diretório do checkpoint."""
    info: dict = {
        "name": name, "sprint": None, "loss": None,
        "split": None, "phase": None,
    }

    if name.startswith("Sprint3b_"):
        info["sprint"] = "Sprint3b"
    elif name.startswith("Sprint3_"):
        info["sprint"] = "Sprint3"
    else:
        info["sprint"] = name.split("_")[0]

    m = re.search(r"_S(\d+)_", name)
    if m:
        info["split"] = int(m.group(1))

    name_l = name.lower()
    for loss in sorted(KNOWN_LOSSES, key=len, reverse=True):
        if loss in name_l:
            info["loss"] = loss
            break

    if "fase2_profon" in name_l:
        info["phase"] = "fase2_profON"
    elif "fase2_profoff" in name_l:
        info["phase"] = "fase2_profOFF"
    elif "fase1" in name_l:
        info["phase"] = "fase1"
    elif "fase2" in name_l:
        info["phase"] = "fase2"

    return info


# ---------------------------------------------------------------------------
# Validation CSV
# ---------------------------------------------------------------------------

def _get_val_csv(split_idx: int, splits_base: Path) -> Optional[Path]:
    """Retorna o CSV completo de validação para o split informado."""
    split_dir = splits_base / SPLITS_DIR_TEMPLATE.format(split=split_idx)
    csv = split_dir / "validation_pairs.csv"
    if csv.exists():
        return csv
    # Fallback: diretório sem o sufixo train_excl_5
    alt = splits_base / f"sprint3_zsl_val_{split_idx}" / "validation_pairs.csv"
    return alt if alt.exists() else None


# ---------------------------------------------------------------------------
# Model loading (backbone carregado uma vez, reutilizado por todos os ckpts)
# ---------------------------------------------------------------------------

def _build_backbone(device: str):
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


RICHPROMPT_COR = (
    "<image> Analyze the provided document image and give me its visual description"
    " based on: Shapes and Elements: presence of graphical components, tables,"
    " sections, headers, and any other visual elements. Layout Consistency: Evaluate"
    " the spatial arrangement of text blocks, margins, and alignments. Content Type:"
    " Ensure the document types of content (e.g., tables, forms, paragraphs),"
    " regardless of specific wording."
)

# Maps substring in run name → prompt override (for old checkpoints without saved prompt)
_PROMPT_OVERRIDES: dict[str, str] = {
    "richprompt_cor": RICHPROMPT_COR,
}


def _prompt_for_run(run_name: str, cfg: dict) -> str:
    """Returns the embedding prompt for a given run, using cfg if available."""
    if "embedding_prompt" in cfg:
        return cfg["embedding_prompt"]
    name_lower = run_name.lower()
    for key, prompt in _PROMPT_OVERRIDES.items():
        if key in name_lower:
            return prompt
    return EMBEDDING_PROMPT


def _build_siam(backbone, tokenizer, device: str, cfg: dict):
    from cavl_doc.models.modeling_cavl import build_cavl_model
    from cavl_doc.utils.embedding_utils import prepare_inputs_for_multimodal_embedding

    cut_layer    = cfg.get("cut_layer",   CUT_LAYER)
    pooler_type  = cfg.get("pooler_type", POOLER_TYPE)
    head_type    = cfg.get("head_type",   HEAD_TYPE)
    num_queries  = cfg.get("num_queries", NUM_QUERIES)
    proj_out     = cfg.get("projection_output_dim", PROJ_OUT_DIM)

    # Mutable holder so the prompt can be updated per-checkpoint without rebuilding siam
    _prompt_holder = [cfg.get("embedding_prompt", EMBEDDING_PROMPT)]

    def _encode_fn(backbone, images, cut_layer=cut_layer, **kwargs):
        from cavl_doc.utils.embedding_utils import prepare_inputs_for_multimodal_embedding
        if isinstance(images, torch.Tensor) and images.dim() == 5:
            images_list = [images[i] for i in range(images.shape[0])]
        elif isinstance(images, torch.Tensor):
            images_list = [images]
        else:
            images_list = images

        input_ids_list, pixel_values_list, image_flags_list = [], [], []
        for img in images_list:
            out = prepare_inputs_for_multimodal_embedding(
                backbone, tokenizer, img, _prompt_holder[0]
            )
            input_ids_list.append(out["input_ids"][0])
            pixel_values_list.append(out["pixel_values"])
            image_flags_list.append(out["image_flags"])

        max_len = max(len(ids) for ids in input_ids_list)
        pad_id  = tokenizer.pad_token_id or 0
        padded_ids, padded_mask = [], []
        for ids in input_ids_list:
            pad = max_len - len(ids)
            padded_ids.append(torch.cat([
                ids, torch.full((pad,), pad_id, device=ids.device, dtype=ids.dtype)
            ]))
            padded_mask.append(torch.cat([
                torch.ones_like(ids),
                torch.zeros(pad, device=ids.device, dtype=ids.dtype),
            ]))

        batch_input_ids    = torch.stack(padded_ids).to(device)
        batch_attn_mask    = torch.stack(padded_mask).to(device)
        batch_pixel_values = torch.cat(pixel_values_list).to(device, dtype=torch.bfloat16)
        batch_image_flags  = torch.cat(image_flags_list).to(device)

        out = backbone(
            input_ids=batch_input_ids,
            attention_mask=batch_attn_mask,
            pixel_values=batch_pixel_values,
            image_flags=batch_image_flags,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = out.hidden_states
        lm  = backbone.language_model.model
        idx = cut_layer + 1 if len(hidden) == len(lm.layers) + 1 else cut_layer
        return hidden[idx], batch_attn_mask, batch_input_ids

    siam = build_cavl_model(
        backbone=backbone,
        cut_layer=cut_layer,
        encode_fn=_encode_fn,
        pool_dim=proj_out,
        proj_hidden=4096,
        proj_out=proj_out,
        set_trainable=False,
        tokenizer=tokenizer,
        pooler_type=pooler_type,
        head_type=head_type,
        num_queries=num_queries,
    ).to(device)
    siam._prompt_holder = _prompt_holder
    return siam


def _nq_from_name(run_name: str) -> int:
    """Extrai num_queries do nome do run (ex: 'nq2' → 2). Default 1."""
    m = re.search(r"_nq(\d+)_", run_name.lower())
    return int(m.group(1)) if m else 1


def _load_weights(siam, ckpt_path: Path, device: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt.get("config", {})

    ckpt_pooler  = cfg.get("pooler_type", POOLER_TYPE)
    proj_out     = cfg.get("projection_output_dim", PROJ_OUT_DIM)
    # num_queries: prefere config salvo; fallback: extrai do nome do diretório
    ckpt_nq      = cfg.get("num_queries") or _nq_from_name(ckpt_path.parent.name)

    cur_pooler = getattr(siam.pool, "_pooler_type_tag", POOLER_TYPE)
    cur_nq     = getattr(siam.pool, "num_queries", NUM_QUERIES)

    # Reconstrói o pool se o tipo OU o num_queries do checkpoint diferir do siam atual
    if "siam_pool" in ckpt and (ckpt_pooler != cur_pooler or ckpt_nq != cur_nq):
        from cavl_doc.modules.poolers import build_pooler
        new_pool = build_pooler(ckpt_pooler, hidden_dim=proj_out, num_queries=ckpt_nq).to(device)
        siam.pool = new_pool
        siam.pool._pooler_type_tag = ckpt_pooler
        print(f"  ⚙️  Pool reconstruído: {ckpt_pooler} nq={ckpt_nq}")

    if "siam_pool" in ckpt:
        siam.pool.load_state_dict(ckpt["siam_pool"])
    if "siam_head" in ckpt:
        siam.head.load_state_dict(ckpt["siam_head"])
    if "backbone_trainable" in ckpt and ckpt["backbone_trainable"]:
        siam.backbone.load_state_dict(ckpt["backbone_trainable"], strict=False)
    siam.eval()
    return cfg


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _make_loader(val_csv: Path, base_image_dir: str, batch_size: int):
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
        return (
            [s["image_a"] for s in batch],
            [s["image_b"] for s in batch],
            torch.tensor([int(s["label"])   for s in batch], dtype=torch.long),
            torch.tensor([int(s["class_a"]) for s in batch], dtype=torch.long),
            torch.tensor([int(s["class_b"]) for s in batch], dtype=torch.long),
        )

    return dataset, DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=_collate,
    )


def _run_eval(siam, val_csv: Path, base_image_dir: str,
              device: str, loss_type: str, batch_size: int) -> dict:
    from cavl_doc.modules.losses import build_loss
    from cavl_doc.trainers.rl_trainer import validate_siam_on_loader

    dataset, loader = _make_loader(val_csv, base_image_dir, batch_size)
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
        "eer":         float(veer),
        "loss":        float(vloss),
        "recall_at_1": float(vr1),
        "threshold":   float(vthr),
        "n_pairs":     len(dataset),
    }


# ---------------------------------------------------------------------------
# W&B
# ---------------------------------------------------------------------------

def _variant_tag(run_name: str) -> str:
    """Extrai a variante do nome do run (entre '_noinit_' e '_fase')."""
    m = re.search(r"_noinit_(.+?)_fase", run_name.lower())
    return m.group(1) if m else ""


def _log_wandb(run_info: dict, metrics: dict,
               wandb_entity: str, wandb_project: str) -> None:
    try:
        import wandb
        variant = _variant_tag(run_info.get("name", ""))
        run_name = (
            f"FullEval_{run_info['sprint']}"
            f"_S{run_info['split']}"
            f"_{run_info['loss']}"
            f"_{variant}"
            f"_{run_info['phase']}"
        )
        run = wandb.init(
            entity=wandb_entity, project=wandb_project,
            name=run_name, config=run_info, reinit=True,
        )
        wandb.log({
            "val/eer":         metrics["eer"],
            "val/loss":        metrics["loss"],
            "val/recall_at_1": metrics["recall_at_1"],
            "val/threshold":   metrics["threshold"],
            "val/n_pairs":     metrics["n_pairs"],
        })
        run.finish()
        print(f"  ✅ W&B: {run_name}")
    except Exception as e:
        print(f"  ⚠️  W&B log falhou: {e}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(results: List[dict]) -> None:
    if not results:
        return
    import pandas as pd
    df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("RESUMO — Avaliação completa LA-CDIP (splits 0-4)")
    print("=" * 80)

    # Melhor EER por (sprint, loss, phase) — média sobre splits
    for (sprint, loss, phase), grp in df.groupby(["sprint", "loss", "phase"]):
        eers = grp["eer"].values * 100
        splits_str = ",".join(map(str, sorted(grp["split"].tolist())))
        print(
            f"\n  [{sprint}] {str(loss):25s}  {str(phase):15s}  splits=[{splits_str}]"
            f"\n    Média={eers.mean():.2f}%  Std={eers.std():.2f} pp"
            f"  Mín={eers.min():.2f}%  Máx={eers.max():.2f}%"
        )

    # Melhor acumulado por (sprint, loss, split) = min(fase1, fase2)
    print("\n" + "=" * 80)
    print("MELHOR ACUMULADO — min(EER_fase1, EER_fase2)")
    print("=" * 80)

    fase1  = df[df["phase"] == "fase1"].set_index(["sprint", "loss", "split"])["eer"]
    p2_on  = df[df["phase"] == "fase2_profON"].set_index(["sprint", "loss", "split"])["eer"]
    p2_off = df[df["phase"] == "fase2_profOFF"].set_index(["sprint", "loss", "split"])["eer"]

    by_sprint_loss: Dict[Tuple, List[float]] = {}
    all_keys = set(fase1.index) | set(p2_on.index) | set(p2_off.index)
    for key in sorted(all_keys):
        sprint, loss, split = key
        candidates = [
            v for v in (
                _scalar(fase1.get(key)),
                _scalar(p2_on.get(key)),
                _scalar(p2_off.get(key)),
            ) if v is not None
        ]
        if not candidates:
            continue
        by_sprint_loss.setdefault((sprint, loss), []).append(min(candidates))

    for (sprint, loss), eers in sorted(by_sprint_loss.items()):
        arr = np.array(eers) * 100
        print(
            f"  [{sprint}] {loss:25s}  n={len(arr)}"
            f"  Média={arr.mean():.2f}%  Std={arr.std():.2f} pp"
        )


def _scalar(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Results cache (skip already-evaluated checkpoints)
# ---------------------------------------------------------------------------

def _load_cache(cache_path: Path) -> set[str]:
    """Returns set of checkpoint_path strings already evaluated."""
    if not cache_path.exists():
        return set()
    import pandas as pd
    try:
        df = pd.read_csv(cache_path)
        if "checkpoint_path" in df.columns:
            return set(df["checkpoint_path"].dropna().tolist())
    except Exception:
        pass
    return set()


def _append_cache(cache_path: Path, row: dict) -> None:
    """Appends one result row to the cache CSV immediately."""
    import pandas as pd
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    df.to_csv(cache_path, mode="a", header=not cache_path.exists(), index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_manifest(manifest_path: Path, target_splits: set[int]) -> list[dict]:
    """Lê o manifesto JSON e filtra pelos splits desejados e paths existentes."""
    import json
    with open(manifest_path) as f:
        entries = json.load(f)

    result = []
    for e in entries:
        if e.get("split") not in target_splits:
            continue
        ckpt = Path(e["checkpoint_path"])
        if not ckpt.exists():
            print(f"  ⚠️  Checkpoint não encontrado (pulando): {ckpt}")
            continue
        result.append(e)
    return result


def main() -> None:
    p = argparse.ArgumentParser(
        description="Avaliação completa (sem subset) nos splits 0-4 do LA-CDIP."
    )
    # Fonte dos checkpoints: manifesto JSON ou varredura de diretório
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest",       metavar="JSON",
                     help="Manifesto JSON gerado por gen_eval_manifest.py.")
    src.add_argument("--checkpoint-root", metavar="DIR",
                     help="Raiz onde buscar checkpoints recursivamente.")

    p.add_argument("--run-prefix",      default="Sprint3_",
                   help="Prefixo dos runs — usado apenas com --checkpoint-root.")
    p.add_argument("--base-image-dir",  required=True,
                   help="Diretório base das imagens LA-CDIP.")
    p.add_argument("--splits-base",     default=None,
                   help="Diretório com os splits gerados (default: <workspace>/data/generated_splits).")
    p.add_argument("--splits",          default="0,1,2,3,4",
                   help="Splits a avaliar (default: 0,1,2,3,4).")
    p.add_argument("--gpu-id",          type=int, default=None)
    p.add_argument("--batch-size",      type=int, default=4)
    p.add_argument("--results-cache",   default="data/eval_lacdip_cache.csv",
                   help="CSV onde os resultados são acumulados; checkpoints já presentes são pulados.")
    p.add_argument("--wandb-project",   default=WANDB_PROJECT)
    p.add_argument("--wandb-entity",    default=WANDB_ENTITY)
    p.add_argument("--no-wandb",        action="store_true")
    p.add_argument("--dry-run",         action="store_true",
                   help="Lista os checkpoints encontrados sem rodar avaliação.")
    args = p.parse_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    splits_base   = Path(args.splits_base) if args.splits_base \
                    else WORKSPACE_ROOT / "data" / "generated_splits"
    target_splits = {int(s.strip()) for s in args.splits.split(",") if s.strip()}

    print(f"Device      : {device}")
    print(f"Splits base : {splits_base}")
    print(f"Splits      : {sorted(target_splits)}")

    # --- Resolve lista de checkpoints ---
    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            print(f"❌ Manifesto não encontrado: {manifest_path}")
            return
        print(f"Manifesto   : {manifest_path}")
        entries = _load_manifest(manifest_path, target_splits)
        ckpts_info = [
            (Path(e["checkpoint_path"]), {
                "sprint": e["sprint"], "split": e["split"],
                "loss": e["loss"], "phase": e["phase"],
                "name": e.get("run_name", Path(e["checkpoint_path"]).parent.name),
            })
            for e in entries
        ]
    else:
        checkpoint_root = Path(args.checkpoint_root)
        print(f"Ckpt root   : {checkpoint_root}")
        print(f"Run prefix  : {args.run_prefix}")
        raw = _find_checkpoints(checkpoint_root, args.run_prefix)
        raw = [c for c in raw if _parse_run_name(c.parent.name)["split"] in target_splits]
        ckpts_info = [(c, {**_parse_run_name(c.parent.name), "name": c.parent.name}) for c in raw]

    if not ckpts_info:
        print("❌ Nenhum checkpoint disponível para avaliar.")
        return

    print(f"\nCheckpoints a avaliar: {len(ckpts_info)}")
    for _, info in ckpts_info:
        print(f"  [{info['sprint']}] S{info['split']} | {info['loss']} | {info['phase']}")

    if args.dry_run:
        return

    # --- Cache de resultados já processados ---
    cache_path = Path(args.results_cache)
    done_paths = _load_cache(cache_path)
    if done_paths:
        print(f"\nCache: {len(done_paths)} checkpoint(s) já avaliados em '{cache_path}' — serão pulados.")

    # --- Load backbone (uma vez) ---
    print("\nCarregando backbone (uma vez para todos os checkpoints)...")
    backbone, tokenizer = _build_backbone(device)
    backbone = backbone.to(device)
    backbone.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")

    # Constrói siam shell (pesos carregados por checkpoint)
    siam = _build_siam(backbone, tokenizer, device, cfg={})
    siam.pool._pooler_type_tag = POOLER_TYPE  # marca o tipo atual para detecção de mudança

    results: List[dict] = []

    for ckpt_path, info in ckpts_info:
        split = info["split"]
        loss  = info["loss"] or "unknown"

        # Pula se já avaliado
        if str(ckpt_path) in done_paths:
            print(f"\n  ⏭️  Pulando (já avaliado): {ckpt_path.parent.name}")
            continue

        val_csv = _get_val_csv(split, splits_base)
        if val_csv is None:
            print(f"\n⚠️  CSV não encontrado para split {split}. Pulando {ckpt_path.parent.name}.")
            continue

        print(f"\n{'='*70}")
        print(f"[{info['sprint']}] split={split} | {loss} | {info['phase']}")
        print(f"  Checkpoint : {ckpt_path.parent.name}")
        print(f"  Val CSV    : {val_csv.name}  ({val_csv.stat().st_size // 1024} KB)")  # type: ignore[union-attr]

        # Carrega pesos deste checkpoint no siam existente
        cfg  = _load_weights(siam, ckpt_path, device)
        info = {**info, **{k: cfg.get(k) for k in
                           ["cut_layer", "pooler_type", "head_type", "num_queries",
                            "projection_output_dim", "margin", "scale"]
                           if k in cfg}}

        # Atualiza prompt sem reconstruir o siam (richprompt_cor e futuros checkpoints)
        prompt = _prompt_for_run(ckpt_path.parent.name, cfg)
        siam._prompt_holder[0] = prompt
        print(f"  Prompt     : {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

        t0 = time.time()
        try:
            metrics = _run_eval(siam, val_csv, args.base_image_dir,
                                device, loss, args.batch_size)
        except Exception as e:
            print(f"  ❌ Erro: {e}")
            continue
        elapsed = time.time() - t0

        print(
            f"  EER={metrics['eer']*100:.2f}%  "
            f"R@1={metrics['recall_at_1']*100:.1f}%  "
            f"pairs={metrics['n_pairs']}  "
            f"({elapsed/60:.1f} min)"
        )

        row = {
            "checkpoint_path": str(ckpt_path),
            "run_name":        ckpt_path.parent.name,
            "sprint":          info["sprint"],
            "loss":            loss,
            "split":           split,
            "phase":           info["phase"],
            **metrics,
        }
        _append_cache(cache_path, row)
        done_paths.add(str(ckpt_path))

        results.append({
            "sprint": info["sprint"],
            "loss":   loss,
            "split":  split,
            "phase":  info["phase"],
            **metrics,
        })

        if not args.no_wandb:
            _log_wandb(info, metrics, args.wandb_entity, args.wandb_project)

    _print_summary(results)


if __name__ == "__main__":
    main()
