#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = WORKSPACE_ROOT / "scripts" / "training" / "run_cavl_training.py"
PREP_SPLITS_SCRIPT = WORKSPACE_ROOT / "scripts" / "utils" / "prepare_splits.py"

DEFAULT_RVL_DATA_ROOT = "/mnt/data/zs_rvl_cdip"
DEFAULT_RVL_IMAGES_DIR = "/mnt/data/zs_rvl_cdip/data"
DEFAULT_PROJECT = "CaVL-Transfer-LACDIP-to-RVL-ZSL-SubcenterCosface"
DEFAULT_LOSS = "subcenter_cosface"


def _resolve_checkpoint_root(user_value: str | None) -> Path:
    if user_value:
        return Path(user_value).expanduser().resolve()
    if Path("/mnt/large/checkpoints").exists():
        return Path("/mnt/large/checkpoints")
    return (WORKSPACE_ROOT / "checkpoints").resolve()


def _find_latest_lacdip_source_checkpoint(checkpoint_root: Path, name_filter: str) -> Path | None:
    candidates: list[Path] = []
    lower_filter = name_filter.lower()

    for ckpt_name in ["best_siam.pt", "last_checkpoint.pt"]:
        for candidate in checkpoint_root.rglob(ckpt_name):
            parent = candidate.parent.name.lower()
            if lower_filter not in parent:
                continue
            candidates.append(candidate)

    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _ensure_split_exists(split_idx: int, protocol: str, data_root: str, pairs_per_class: int) -> Path:
    split_dir = WORKSPACE_ROOT / "data" / "generated_splits" / f"RVL-CDIP_{protocol}_split_{split_idx}"
    train_csv = split_dir / "train_pairs.csv"
    val_csv = split_dir / "validation_pairs.csv"

    if train_csv.exists() and val_csv.exists():
        return split_dir

    cmd = [
        sys.executable,
        str(PREP_SPLITS_SCRIPT),
        "--data-root",
        data_root,
        "--output-dir",
        str(split_dir),
        "--split-idx",
        str(split_idx),
        "--protocol",
        protocol,
        "--pairs-per-class",
        str(pairs_per_class),
    ]
    print(f"[INFO] Split {split_idx} não encontrado. Gerando com prepare_splits...")
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if not train_csv.exists():
        raise FileNotFoundError(f"Split gerado sem train_pairs.csv: {train_csv}")
    return split_dir


def _build_command(
    python_bin: str,
    split_idx: int,
    pairs_csv: Path,
    base_image_dir: str,
    wandb_project: str,
    run_suffix: str,
    init_checkpoint: Path,
    args: argparse.Namespace,
) -> list[str]:
    run_name = f"Transfer_LA2RVL_S{split_idx}_{DEFAULT_LOSS}_{run_suffix}".rstrip("_")
    professor_lr_value = 0.0 if args.disable_teacher else args.professor_lr

    cmd = [
        python_bin,
        str(TRAIN_SCRIPT),
        "--use-wandb",
        "--wandb-project",
        wandb_project,
        "--wandb-run-name",
        run_name,
        "--dataset-name",
        "RVL-CDIP",
        "--model-name",
        "InternVL3-2B",
        "--pairs-csv",
        str(pairs_csv),
        "--base-image-dir",
        base_image_dir,
        "--loss-type",
        DEFAULT_LOSS,
        "--optimizer-type",
        "adamw",
        "--scheduler-type",
        "constant",
        "--student-lr",
        str(args.student_lr),
        "--professor-lr",
        str(professor_lr_value),
        "--margin",
        str(args.margin),
        "--scale",
        str(args.scale),
        "--num-sub-centers",
        str(args.num_sub_centers),
        "--epochs",
        str(args.epochs),
        "--student-batch-size",
        str(args.student_batch_size),
        "--candidate-pool-size",
        str(args.candidate_pool_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--val-subset-size",
        str(args.val_subset_size),
        "--num-workers",
        str(args.num_workers),
        "--patience",
        str(args.patience),
        "--lr-reduce-factor",
        str(args.lr_reduce_factor),
        "--projection-output-dim",
        "1536",
        "--max-num-image-tokens",
        "12",
        "--cut-layer",
        "27",
        "--pooler-type",
        "attention",
        "--head-type",
        "mlp",
        "--num-queries",
        str(args.num_queries),
        "--baseline-alpha",
        str(args.baseline_alpha),
        "--entropy-coeff",
        str(args.entropy_coeff),
        "--seed",
        str(args.seed),
        "--init-from-checkpoint",
        str(init_checkpoint),
    ]

    if args.init_load_professor:
        cmd.append("--init-load-professor")

    if args.disable_teacher:
        cmd.extend(
            [
                "--professor-warmup-steps",
                str(args.professor_warmup_steps),
                "--easy-mining-steps",
                str(args.easy_mining_steps),
            ]
        )

    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tuning de transferência LA-CDIP -> RVL-CDIP (ZSL) usando subcenter_cosface"
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--splits", default="0,1,2,3", help="Splits RVL ZSL separados por vírgula")
    parser.add_argument("--protocol", default="zsl", choices=["zsl", "gzsl"])
    parser.add_argument("--wandb-project", default=DEFAULT_PROJECT)
    parser.add_argument("--run-suffix", default="main")

    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument(
        "--source-checkpoint",
        default=None,
        help="Checkpoint base LA-CDIP explícito (best_siam.pt / last_checkpoint.pt). Se vazio, faz busca automática.",
    )
    parser.add_argument(
        "--source-filter",
        default="sprint1_subcenter_cosface",
        help="Filtro no nome da pasta para encontrar o checkpoint fonte LA-CDIP automaticamente",
    )

    parser.add_argument("--rvl-data-root", default=DEFAULT_RVL_DATA_ROOT)
    parser.add_argument("--base-image-dir", default=DEFAULT_RVL_IMAGES_DIR)
    parser.add_argument("--pairs-per-class", type=int, default=100)

    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--student-lr", type=float, default=5e-5)
    parser.add_argument("--professor-lr", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.35)
    parser.add_argument("--scale", type=float, default=24.0)
    parser.add_argument("--num-sub-centers", type=int, default=3)

    parser.add_argument("--student-batch-size", type=int, default=6)
    parser.add_argument("--candidate-pool-size", type=int, default=10)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=3)
    parser.add_argument("--val-subset-size", type=int, default=1200)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr-reduce-factor", type=float, default=0.5)
    parser.add_argument("--num-queries", type=int, default=1)
    parser.add_argument("--baseline-alpha", type=float, default=0.05)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--disable-teacher", action="store_true", help="Desativa dinâmica do teacher via warmup longo")
    parser.add_argument("--professor-warmup-steps", type=int, default=999999)
    parser.add_argument("--easy-mining-steps", type=int, default=999999)
    parser.add_argument("--init-load-professor", action="store_true")

    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    splits = [int(item.strip()) for item in args.splits.split(",") if item.strip()]
    if not splits:
        raise ValueError("Nenhum split informado em --splits")

    checkpoint_root = _resolve_checkpoint_root(args.checkpoint_root)
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"checkpoint_root não encontrado: {checkpoint_root}")

    if args.source_checkpoint:
        source_checkpoint = Path(args.source_checkpoint).expanduser().resolve()
    else:
        source_checkpoint = _find_latest_lacdip_source_checkpoint(
            checkpoint_root=checkpoint_root,
            name_filter=args.source_filter,
        )

    if source_checkpoint is None or not source_checkpoint.exists():
        raise FileNotFoundError(
            "Não foi possível encontrar checkpoint fonte LA-CDIP. "
            "Use --source-checkpoint ou ajuste --source-filter."
        )

    print("=" * 90)
    print("Transfer Fine-Tuning | LA-CDIP -> RVL-CDIP | Loss=subcenter_cosface | Zero-shot por split")
    print("=" * 90)
    print(f"Checkpoint fonte: {source_checkpoint}")
    print(f"Splits: {splits}")
    print(f"Projeto WandB: {args.wandb_project}")

    for split_idx in splits:
        split_dir = _ensure_split_exists(
            split_idx=split_idx,
            protocol=args.protocol,
            data_root=args.rvl_data_root,
            pairs_per_class=args.pairs_per_class,
        )
        pairs_csv = split_dir / "train_pairs.csv"

        cmd = _build_command(
            python_bin=args.python_bin,
            split_idx=split_idx,
            pairs_csv=pairs_csv,
            base_image_dir=args.base_image_dir,
            wandb_project=args.wandb_project,
            run_suffix=args.run_suffix,
            init_checkpoint=source_checkpoint,
            args=args,
        )

        print("-" * 90)
        print(f"[Split {split_idx}] CMD:")
        print(" ".join(cmd))

        if args.dry_run:
            continue

        subprocess.run(cmd, check=True)
        time.sleep(3)

    print("\n✅ Execução de transferência concluída.")


if __name__ == "__main__":
    main()
