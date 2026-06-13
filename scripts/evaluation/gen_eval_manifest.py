#!/usr/bin/env python3
"""
Gera (ou atualiza) o manifesto JSON de checkpoints para avaliação completa.

Escaneia --checkpoint-root e registra todos os best_model.pt que correspondam
ao prefixo informado. O JSON pode ser editado manualmente e passado ao
eval_lacdip_full.py via --manifest.

Uso:
    # Servidor local (Sprint3)
    python scripts/evaluation/gen_eval_manifest.py \\
        --checkpoint-root /mnt/large/checkpoints \\
        --run-prefix Sprint3_ \\
        --output data/eval_manifest.json

    # gpds2 (Sprint3b) — merge no JSON existente
    python scripts/evaluation/gen_eval_manifest.py \\
        --checkpoint-root /mnt/nas/joaopaulo/CaVL-Doc/checkpoints \\
        --run-prefix Sprint3b_ \\
        --output data/eval_manifest.json \\
        --merge
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

KNOWN_LOSSES = [
    "subcenter_cosface", "subcenter_arcface",
    "contrastive", "cosface", "arcface", "triplet", "circle",
]


def _find_checkpoints(checkpoint_root: Path, run_prefix: str, name_filter: str | None = None) -> list[Path]:
    found, seen_dirs = [], set()
    prefix = run_prefix.lower()
    nf = name_filter.lower() if name_filter else None
    for ckpt_name in ["best_model.pt", "best_siam.pt"]:
        for ckpt in checkpoint_root.rglob(ckpt_name):
            if ckpt.parent in seen_dirs:
                continue
            name = ckpt.parent.name.lower()
            if name.startswith(prefix) and "fase" in name and (nf is None or nf in name):
                found.append(ckpt)
                seen_dirs.add(ckpt.parent)
    return sorted(found, key=lambda p: p.parent.name)


def _parse(name: str) -> dict:
    info = {"sprint": None, "split": None, "loss": None, "phase": None}

    if name.lower().startswith("sprint3b_"):
        info["sprint"] = "Sprint3b"
    elif name.lower().startswith("sprint3_"):
        info["sprint"] = "Sprint3"
    else:
        info["sprint"] = name.split("_")[0]

    m = re.search(r"_S(\d+)_", name)
    if m:
        info["split"] = int(m.group(1))

    nl = name.lower()
    for loss in sorted(KNOWN_LOSSES, key=len, reverse=True):
        if loss in nl:
            info["loss"] = loss
            break

    if "fase2_profon" in nl:
        info["phase"] = "fase2_profON"
    elif "fase2_profoff" in nl:
        info["phase"] = "fase2_profOFF"
    elif "fase1" in nl:
        info["phase"] = "fase1"
    elif "fase2" in nl:
        info["phase"] = "fase2"

    return info


def main() -> None:
    p = argparse.ArgumentParser(description="Gera manifesto JSON de checkpoints.")
    p.add_argument("--checkpoint-root", required=True)
    p.add_argument("--run-prefix", default="Sprint3_",
                   help="Prefixo dos runs (Sprint3_ ou Sprint3b_)")
    p.add_argument("--name-filter", default=None,
                   help="Substring obrigatória no nome do diretório (ex: s32k3)")
    p.add_argument("--variant-override", default=None,
                   help="Força variant para todas as entradas (ex: mean). "
                        "Útil quando o suffix do run não contém _noinit_<variant>_fase.")
    p.add_argument("--loss-filter", default=None,
                   help="Só inclui estas losses (vírgula). Ex: subcenter_arcface,triplet")
    p.add_argument("--loss-exclude", default=None,
                   help="Exclui estas losses (vírgula). Ex: subcenter_arcface")
    p.add_argument("--output", default="data/eval_manifest.json")
    p.add_argument("--merge", action="store_true",
                   help="Faz merge com o JSON existente (evita duplicatas por checkpoint_path)")
    args = p.parse_args()

    checkpoint_root = Path(args.checkpoint_root)
    if not checkpoint_root.exists():
        print(f"❌ Diretório não encontrado: {checkpoint_root}")
        sys.exit(1)

    ckpts = _find_checkpoints(checkpoint_root, args.run_prefix, args.name_filter)
    if not ckpts:
        print(f"Nenhum checkpoint encontrado em '{checkpoint_root}' com prefixo '{args.run_prefix}'.")
        sys.exit(0)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Carrega JSON existente se --merge
    existing: list[dict] = []
    existing_paths: set[str] = set()
    if args.merge and output.exists():
        with open(output) as f:
            existing = json.load(f)
        existing_paths = {e["checkpoint_path"] for e in existing}

    loss_include = {l.strip() for l in args.loss_filter.split(",")} if args.loss_filter else None
    loss_exclude = {l.strip() for l in args.loss_exclude.split(",")} if args.loss_exclude else set()

    new_entries = []
    for ckpt in ckpts:
        ckpt_str = str(ckpt)
        if ckpt_str in existing_paths:
            continue
        info = _parse(ckpt.parent.name)
        loss = info["loss"] or "unknown"
        if loss_include is not None and loss not in loss_include:
            continue
        if loss in loss_exclude:
            continue
        entry = {
            "sprint":          info["sprint"],
            "split":           info["split"],
            "loss":            info["loss"] or "unknown",
            "phase":           info["phase"] or "unknown",
            "run_name":        ckpt.parent.name,
            "checkpoint_path": ckpt_str,
        }
        if args.variant_override is not None:
            entry["variant"] = args.variant_override
        new_entries.append(entry)

    all_entries = existing + new_entries
    with open(output, "w") as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)

    print(f"{'Merged' if args.merge else 'Criado'}: {output}")
    print(f"  Entradas existentes : {len(existing)}")
    print(f"  Novas adicionadas   : {len(new_entries)}")
    print(f"  Total               : {len(all_entries)}")

    if new_entries:
        print(f"\n{'sprint':10} {'split':5} {'loss':25} {'phase':15} run_name")
        print("-" * 90)
        for e in new_entries:
            print(
                f"{e['sprint']:10} {str(e['split']):5} "
                f"{e['loss']:25} {e['phase']:15} "
                f"{e['run_name'][:50]}"
            )


if __name__ == "__main__":
    main()
