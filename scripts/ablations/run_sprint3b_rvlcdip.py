#!/usr/bin/env python3
"""
Sprint 3b — RVL-CDIP: mesmo protocolo do Sprint3b (fase1 + fase2 opcional com professor)
aplicado ao dataset RVL-CDIP com splits ZSL, suportando múltiplos poolers.

Uso:
    # Attention pooler (q=1) — splits 0-3
    python scripts/ablations/run_sprint3b_rvlcdip.py \\
        --losses subcenter_arcface \\
        --pooler-type attention \\
        --rvl-data-root /mnt/nas/joaopaulo/RVL-CDIP \\
        --base-image-dir /mnt/nas/joaopaulo/RVL-CDIP/data \\
        --checkpoint-root /mnt/nas/joaopaulo/CaVL-Doc/checkpoints \\
        --wandb-project CaVL-Doc_RVL_Sprint3b_PoolerAblation \\
        --num-workers 0 --gpu-id 0

    # MeanPool
    python scripts/ablations/run_sprint3b_rvlcdip.py \\
        --losses subcenter_arcface \\
        --pooler-type mean \\
        --run-suffix mean \\
        --checkpoint-root /mnt/nas/joaopaulo/CaVL-Doc/checkpoints_ablation_rvl_meanpool \\
        --wandb-project CaVL-Doc_RVL_Sprint3b_PoolerAblation \\
        ...
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT   = WORKSPACE_ROOT / "scripts" / "training" / "run_cavl_training.py"
PREP_SPLITS    = WORKSPACE_ROOT / "scripts" / "utils" / "prepare_splits.py"


# ---------------------------------------------------------------------------
# GPU selection (from Sprint4)
# ---------------------------------------------------------------------------

def _select_gpu(gpu_id: Optional[int], min_free_mib: int, wait_secs: float):
    try:
        import subprocess as sp
        out = sp.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            stderr=sp.DEVNULL,
        ).decode()
        gpus = []
        for line in out.strip().splitlines():
            idx, free = line.split(",")
            gpus.append((int(idx.strip()), int(free.strip())))
        if gpu_id is not None:
            match = [g for g in gpus if g[0] == gpu_id]
            return match[0] if match else (gpu_id, 0)
        gpus.sort(key=lambda g: g[1], reverse=True)
        best = gpus[0] if gpus else None
        if best and best[1] >= min_free_mib:
            return best
        if wait_secs > 0:
            time.sleep(wait_secs)
            return _select_gpu(gpu_id, min_free_mib, 0)
        return best
    except Exception:
        return (gpu_id or 0, 0)


# ---------------------------------------------------------------------------
# Split preparation
# ---------------------------------------------------------------------------

def _ensure_split(split_idx: int, protocol: str, data_root: str, pairs_per_class: int) -> Path:
    split_dir = WORKSPACE_ROOT / "data" / "generated_splits" / f"RVL-CDIP_{protocol}_split_{split_idx}"
    if (split_dir / "train_pairs.csv").exists() and (split_dir / "validation_pairs.csv").exists():
        return split_dir
    cmd = [
        sys.executable, str(PREP_SPLITS),
        "--data-root",       data_root,
        "--output-dir",      str(split_dir),
        "--split-idx",       str(split_idx),
        "--protocol",        protocol,
        "--pairs-per-class", str(pairs_per_class),
    ]
    print(f"[SPLIT] Gerando split {split_idx}...")
    subprocess.run(cmd, check=True)
    if not (split_dir / "train_pairs.csv").exists():
        raise FileNotFoundError(f"train_pairs.csv não gerado em {split_dir}")
    return split_dir


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------

def _build_cmd(
    run_name: str,
    pairs_csv: Path,
    output_dir: Path,
    args: argparse.Namespace,
    loss_type: str,
    epochs: int,
    professor_enabled: bool,
    professor_warmup_steps: int,
) -> List[str]:
    professor_lr    = args.professor_lr if professor_enabled else 0.0
    warmup_steps    = professor_warmup_steps if professor_enabled else 999_999
    easy_steps      = professor_warmup_steps if professor_enabled else 999_999

    return [
        args.python_bin, str(TRAIN_SCRIPT),
        "--use-wandb",
        "--wandb-project",               args.wandb_project,
        "--wandb-run-name",              run_name,
        "--dataset-name",                "RVL-CDIP",
        "--model-name",                  "InternVL3-2B",
        "--pairs-csv",                   str(pairs_csv),
        "--base-image-dir",              args.base_image_dir,
        "--loss-type",                   loss_type,
        "--optimizer-type",              "adamw",
        "--scheduler-type",              args.scheduler_type,
        "--student-lr",                  str(args.student_lr),
        "--professor-lr",                str(professor_lr),
        "--margin",                      str(args.margin),
        "--scale",                       str(args.scale),
        "--num-sub-centers",             str(args.num_sub_centers),
        "--epochs",                      str(epochs),
        "--max-steps-per-epoch",         str(args.max_steps_per_epoch),
        "--student-batch-size",          str(args.student_batch_size),
        "--candidate-pool-size",         str(args.candidate_pool_size),
        "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
        "--num-workers",                 str(args.num_workers),
        "--val-subset-size",             str(args.val_subset_size),
        "--patience",                    str(args.patience),
        "--lr-reduce-factor",            str(args.lr_reduce_factor),
        "--projection-output-dim",       "1536",
        "--max-num-image-tokens",        "12",
        "--cut-layer",                   "27",
        "--pooler-type",                 args.pooler_type,
        "--head-type",                   "mlp",
        "--num-queries",                 str(args.num_queries),
        "--baseline-alpha",              str(args.baseline_alpha),
        "--entropy-coeff",               str(args.entropy_coeff),
        "--professor-warmup-steps",      str(warmup_steps),
        "--easy-mining-steps",           str(easy_steps),
        "--seed",                        str(args.seed),
        "--output-dir",                  str(output_dir),
    ]


# ---------------------------------------------------------------------------
# Skip helpers
# ---------------------------------------------------------------------------

def _should_skip(output_dir: Path, target_epochs: int) -> bool:
    best   = output_dir / "best_model.pt"
    last   = output_dir / "last_checkpoint.pt"
    if best.exists():
        print(f"[SKIP] best_model.pt encontrado — {output_dir.name}")
        return True
    if last.exists():
        try:
            import torch
            ckpt = torch.load(last, map_location="cpu", weights_only=False)
            done = int(ckpt.get("epoch", -1))
            if done >= target_epochs - 1:
                print(f"[SKIP] Completo (época {done+1}/{target_epochs}) — {output_dir.name}")
                return True
            print(f"[RESUME] Época {done+1}/{target_epochs} — {output_dir.name}")
        except Exception:
            pass
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Sprint 3b no RVL-CDIP: fase1 + fase2 (professor opcional), múltiplos poolers."
    )

    p.add_argument("--python-bin",      default=sys.executable)
    p.add_argument("--wandb-project",   default="CaVL-Doc_RVL_Sprint3b_PoolerAblation")

    # Losses
    p.add_argument("--losses",          default="subcenter_arcface,subcenter_cosface,contrastive,triplet")
    p.add_argument("--splits",          default="0,1,2,3")
    p.add_argument("--protocol",        default="zsl", choices=["zsl", "gzsl"])
    p.add_argument("--pairs-per-class", type=int, default=100)
    p.add_argument("--run-suffix",      default="",
                   help="Sufixo adicionado ao nome do run (ex: 'mean', 's32k3').")

    # Dados
    p.add_argument("--rvl-data-root",   default="/mnt/nas/joaopaulo/RVL-CDIP")
    p.add_argument("--base-image-dir",  default="/mnt/nas/joaopaulo/RVL-CDIP/data")
    p.add_argument("--checkpoint-root", default=None)

    # Pooler
    p.add_argument("--pooler-type",     default="attention", choices=["attention", "mean", "cross_modal"])
    p.add_argument("--num-queries",     type=int, default=1)

    # Hiperparâmetros (padrões Sprint3b s32k3)
    p.add_argument("--student-lr",                  type=float, default=5e-5)
    p.add_argument("--professor-lr",                type=float, default=9.24e-5)
    p.add_argument("--scheduler-type",              default="plateau")
    p.add_argument("--margin",                      type=float, default=0.35)
    p.add_argument("--scale",                       type=float, default=32.0)
    p.add_argument("--num-sub-centers",             type=int,   default=3)
    p.add_argument("--student-batch-size",          type=int,   default=8)
    p.add_argument("--candidate-pool-size",         type=int,   default=8)
    p.add_argument("--gradient-accumulation-steps", type=int,   default=2)
    p.add_argument("--num-workers",                 type=int,   default=0)
    p.add_argument("--val-subset-size",             type=int,   default=1200)
    p.add_argument("--patience",                    type=int,   default=5)
    p.add_argument("--lr-reduce-factor",            type=float, default=0.5)
    p.add_argument("--baseline-alpha",              type=float, default=0.029)
    p.add_argument("--entropy-coeff",               type=float, default=0.018)
    p.add_argument("--seed",                        type=int,   default=42)

    # Fases
    p.add_argument("--phase1-epochs",           type=int, default=10)
    p.add_argument("--max-steps-per-epoch",     type=int, default=140)
    p.add_argument("--phase2-epochs",           type=int, default=0,
                   help="Épocas da fase2 com professor (0 = skip fase2).")
    p.add_argument("--professor-warmup-epochs", type=int, default=1)

    # Infra
    p.add_argument("--gpu-id",       type=int,   default=None)
    p.add_argument("--min-free-mib", type=int,   default=10_000)
    p.add_argument("--gpu-wait",     type=float, default=0.0)
    p.add_argument("--sleep",        type=float, default=3.0)
    p.add_argument("--dry-run",      action="store_true")

    args = p.parse_args()

    losses = [l.strip() for l in args.losses.split(",") if l.strip()]
    splits = [int(s.strip()) for s in args.splits.split(",") if s.strip()]
    suffix = f"_{args.run_suffix}" if args.run_suffix else ""

    # Checkpoint root
    if args.checkpoint_root:
        ckpt_root = Path(args.checkpoint_root)
    elif Path("/mnt/nas/joaopaulo/CaVL-Doc/checkpoints").exists():
        ckpt_root = Path("/mnt/nas/joaopaulo/CaVL-Doc/checkpoints")
    elif Path("/mnt/large/checkpoints").exists():
        ckpt_root = Path("/mnt/large/checkpoints")
    else:
        ckpt_root = WORKSPACE_ROOT / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    professor_warmup_steps = args.professor_warmup_epochs * args.max_steps_per_epoch

    print("=" * 80)
    print(f"Sprint 3b RVL-CDIP | pooler={args.pooler_type} q={args.num_queries}")
    print(f"fase1={args.phase1_epochs} épocas | fase2={args.phase2_epochs} épocas")
    print(f"splits={splits} | losses={losses}")
    print(f"checkpoint-root: {ckpt_root}")
    print(f"W&B: {args.wandb_project}")
    print("=" * 80)

    selected_gpu = (args.gpu_id or 0, 0) if args.dry_run else _select_gpu(args.gpu_id, args.min_free_mib, args.gpu_wait)
    if selected_gpu is None:
        print("❌ Nenhuma GPU disponível.")
        sys.exit(1)
    print(f"GPU: {selected_gpu[0]}")

    gpu_env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(selected_gpu[0]), "TMPDIR": "/tmp"}

    for loss_type in losses:
        for split_idx in splits:
            split_dir = _ensure_split(split_idx, args.protocol, args.rvl_data_root, args.pairs_per_class)
            pairs_csv = split_dir / "train_pairs.csv"

            run_p1     = f"Sprint3b_RVL_S{split_idx}_{loss_type}{suffix}_fase1_E{args.phase1_epochs}"
            out_p1     = ckpt_root / run_p1
            cmd_p1     = _build_cmd(run_p1, pairs_csv, out_p1, args, loss_type,
                                    args.phase1_epochs, False, professor_warmup_steps)

            print(f"\n{'─'*80}")
            print(f"[{loss_type} | Split {split_idx}]  fase1: {run_p1}")

            if not args.dry_run and not _should_skip(out_p1, args.phase1_epochs):
                subprocess.run(cmd_p1, check=True, env=gpu_env)
                time.sleep(args.sleep)

            if args.phase2_epochs > 0:
                ckpt_p1    = out_p1 / "last_checkpoint.pt"
                best_p1    = out_p1 / "best_model.pt"
                init_ckpt  = ckpt_p1 if ckpt_p1.exists() else (best_p1 if best_p1.exists() else None)

                run_p2on   = f"Sprint3b_RVL_S{split_idx}_{loss_type}{suffix}_fase2_profON_E{args.phase2_epochs}"
                run_p2off  = f"Sprint3b_RVL_S{split_idx}_{loss_type}{suffix}_fase2_profOFF_E{args.phase2_epochs}"
                out_p2on   = ckpt_root / run_p2on
                out_p2off  = ckpt_root / run_p2off

                cmd_p2on  = _build_cmd(run_p2on,  pairs_csv, out_p2on,  args, loss_type,
                                       args.phase2_epochs, True,  professor_warmup_steps)
                cmd_p2off = _build_cmd(run_p2off, pairs_csv, out_p2off, args, loss_type,
                                       args.phase2_epochs, False, professor_warmup_steps)

                if init_ckpt:
                    cmd_p2on  += ["--init-from-checkpoint", str(init_ckpt)]
                    cmd_p2off += ["--init-from-checkpoint", str(init_ckpt)]

                print(f"  fase2 ON : {run_p2on}")
                print(f"  fase2 OFF: {run_p2off}")

                if not args.dry_run:
                    if not _should_skip(out_p2on, args.phase2_epochs):
                        subprocess.run(cmd_p2on,  check=True, env=gpu_env)
                        time.sleep(args.sleep)
                    if not _should_skip(out_p2off, args.phase2_epochs):
                        subprocess.run(cmd_p2off, check=True, env=gpu_env)
                        time.sleep(args.sleep)

    print("\n✅ Sprint 3b RVL-CDIP concluída.")


if __name__ == "__main__":
    main()
