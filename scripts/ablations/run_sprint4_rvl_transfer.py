#!/usr/bin/env python3
"""
Sprint 4 — Transfer Learning LA-CDIP → RVL-CDIP

Para cada split, roda dois treinos sequencialmente:
  - transfer : inicializado com o modelo do HF Hub (LA-CDIP fine-tuned)
  - direct   : inicializado do zero no RVL-CDIP

O modelo fonte é baixado automaticamente do HF Hub na primeira execução
e cacheado localmente. A loss é lida do config embutido no checkpoint.

Uso:
    python scripts/ablations/run_sprint4_rvl_transfer.py \
        --hf-model-id Jpcosta90/cavl-doc-lacdip \
        --rvl-data-root /mnt/data/zs_rvl_cdip \
        --base-image-dir /mnt/data/zs_rvl_cdip/data

    # GPU específica:
    python scripts/ablations/run_sprint4_rvl_transfer.py \
        --hf-model-id Jpcosta90/cavl-doc-lacdip \
        --gpu-id 2

    # Dry-run (imprime comandos sem executar):
    python scripts/ablations/run_sprint4_rvl_transfer.py \
        --hf-model-id Jpcosta90/cavl-doc-lacdip \
        --dry-run
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT   = WORKSPACE_ROOT / "scripts" / "training" / "run_cavl_training.py"
PREP_SPLITS    = WORKSPACE_ROOT / "scripts" / "utils" / "prepare_splits.py"


# ---------------------------------------------------------------------------
# GPU selection
# ---------------------------------------------------------------------------

def _parse_nvidia_smi_free_memory() -> List[Tuple[int, int]]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode()
        result = []
        for line in output.strip().splitlines():
            idx, free = line.split(",")
            result.append((int(idx.strip()), int(free.strip())))
        return result
    except Exception:
        return []


def _select_gpu(
    gpu_id: Optional[int] = None,
    min_free_mib: int = 10_000,
    wait_seconds: float = 0.0,
) -> Optional[Tuple[int, int]]:
    """Seleciona a GPU com mais memória livre. Aguarda até wait_seconds se necessário."""
    if gpu_id is not None:
        gpus = _parse_nvidia_smi_free_memory()
        match = next((g for g in gpus if g[0] == gpu_id), None)
        return match

    deadline = time.time() + max(0.0, wait_seconds)
    while True:
        gpus = _parse_nvidia_smi_free_memory()
        candidates = [(idx, free) for idx, free in gpus if free >= min_free_mib]
        if candidates:
            return max(candidates, key=lambda x: (x[1], -x[0]))
        if time.time() >= deadline:
            if gpus:
                return max(gpus, key=lambda x: (x[1], -x[0]))
            return None
        print(f"  Aguardando GPU com ≥{min_free_mib} MiB livres...")
        time.sleep(30)


# ---------------------------------------------------------------------------
# HF Hub download
# ---------------------------------------------------------------------------

def _download_hf_checkpoint(hub_id: str, cache_dir: Path) -> Tuple[Path, dict]:
    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / "best_siam.pt"

    if local_path.exists():
        print(f"[HF] Checkpoint em cache: {local_path}")
    else:
        print(f"[HF] Baixando '{hub_id}' → {local_path} ...")
        downloaded = hf_hub_download(
            repo_id=hub_id,
            filename="best_siam.pt",
            local_dir=str(cache_dir),
        )
        if Path(downloaded) != local_path:
            Path(downloaded).rename(local_path)
        print(f"[HF] Download concluído: {local_path}")

    ckpt = torch.load(local_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    eer    = ckpt.get("metrics", {}).get("eer")
    epoch  = ckpt.get("epoch")
    print(f"[HF] Modelo: EER={eer*100:.2f}% (epoch {epoch})" if eer else "[HF] Modelo carregado.")
    return local_path, config


# ---------------------------------------------------------------------------
# Split helpers
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
    args: argparse.Namespace,
    loss_type: str,
    init_checkpoint: Optional[Path],
    professor_lr: float,
    warmup_steps: int,
) -> List[str]:
    cmd = [
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
        "--epochs",                      str(args.epochs),
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
        "--pooler-type",                 "attention",
        "--head-type",                   "mlp",
        "--num-queries",                 str(args.num_queries),
        "--baseline-alpha",              str(args.baseline_alpha),
        "--entropy-coeff",               str(args.entropy_coeff),
        "--professor-warmup-steps",      str(warmup_steps),
        "--seed",                        str(args.seed),
    ]

    if init_checkpoint is not None:
        cmd += ["--init-from-checkpoint", str(init_checkpoint)]

    return cmd


# ---------------------------------------------------------------------------
# Skip helper
# ---------------------------------------------------------------------------

def _should_skip(checkpoint_root: Path, run_name: str) -> bool:
    ckpt = checkpoint_root / run_name / "last_checkpoint.pt"
    if ckpt.exists():
        print(f"[SKIP] {run_name} — last_checkpoint.pt já existe.")
        return True
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Sprint 4: Transfer LA-CDIP→RVL-CDIP vs. treino direto no RVL-CDIP"
    )

    # Fonte do modelo
    p.add_argument("--hf-model-id", required=True,
                   help="ID do modelo no HF Hub (ex: Jpcosta90/cavl-doc-lacdip)")
    p.add_argument("--hf-cache-dir", default=None,
                   help="Diretório local para cache do modelo HF.")
    p.add_argument("--loss-type", default=None,
                   help="Loss para o RVL-CDIP. Se vazio, lido do config do modelo HF.")

    # Dados RVL-CDIP
    p.add_argument("--rvl-data-root",   default="/mnt/data/zs_rvl_cdip")
    p.add_argument("--base-image-dir",  default="/mnt/data/zs_rvl_cdip/data")
    p.add_argument("--splits",          default="0,1,2,3")
    p.add_argument("--protocol",        default="zsl", choices=["zsl", "gzsl"])
    p.add_argument("--pairs-per-class", type=int, default=100)

    # W&B
    p.add_argument("--wandb-project", default="CaVL-Doc_RVL_Sprint4_Transfer")

    # Hiperparâmetros de treino
    p.add_argument("--epochs",                     type=int,   default=10)
    p.add_argument("--max-steps-per-epoch",        type=int,   default=140)
    p.add_argument("--student-lr",                 type=float, default=5e-5)
    p.add_argument("--professor-lr",               type=float, default=5e-5)
    p.add_argument("--scheduler-type",             default="plateau",
                   choices=["step", "cosine", "plateau", "constant"])
    p.add_argument("--margin",                     type=float, default=0.35)
    p.add_argument("--scale",                      type=float, default=24.0)
    p.add_argument("--num-sub-centers",            type=int,   default=2)
    p.add_argument("--student-batch-size",         type=int,   default=4)
    p.add_argument("--candidate-pool-size",        type=int,   default=8)
    p.add_argument("--gradient-accumulation-steps", type=int,  default=3)
    p.add_argument("--num-workers",                type=int,   default=0)
    p.add_argument("--val-subset-size",            type=int,   default=1200)
    p.add_argument("--patience",                   type=int,   default=5)
    p.add_argument("--lr-reduce-factor",           type=float, default=0.5)
    p.add_argument("--num-queries",                type=int,   default=1)
    p.add_argument("--baseline-alpha",             type=float, default=0.05)
    p.add_argument("--entropy-coeff",              type=float, default=0.01)
    p.add_argument("--seed",                       type=int,   default=42)

    # Warmup do professor
    p.add_argument("--professor-warmup-epochs", type=int, default=1,
                   help="Épocas de shadow warmup do professor antes de aplicar seleção.")

    # GPU
    p.add_argument("--gpu-id",       type=int, default=None,
                   help="ID físico da GPU a usar (default: seleção automática por memória livre).")
    p.add_argument("--min-free-mib", type=int, default=10_000,
                   help="Memória mínima livre (MiB) para seleção automática de GPU.")
    p.add_argument("--gpu-wait",     type=float, default=0.0,
                   help="Segundos para aguardar GPU com memória suficiente.")

    # Infra
    p.add_argument("--python-bin",      default=sys.executable)
    p.add_argument("--checkpoint-root", default=None)
    p.add_argument("--sleep",           type=float, default=3.0)
    p.add_argument("--dry-run",         action="store_true")

    args = p.parse_args()

    # Resolve paths
    if args.checkpoint_root:
        ckpt_root = Path(args.checkpoint_root)
    elif Path("/mnt/large/checkpoints").exists():
        ckpt_root = Path("/mnt/large/checkpoints")
    else:
        ckpt_root = WORKSPACE_ROOT / "checkpoints"

    if args.hf_cache_dir:
        hf_cache = Path(args.hf_cache_dir)
    else:
        safe_id = args.hf_model_id.replace("/", "_")
        hf_cache = ckpt_root / "hf_cache" / safe_id

    splits = [int(s.strip()) for s in args.splits.split(",") if s.strip()]
    professor_warmup_steps = args.professor_warmup_epochs * args.max_steps_per_epoch

    # Seleção de GPU
    print("=" * 80)
    print(f"Sprint 4 | RVL-CDIP | Transfer vs. Direto | modelo: {args.hf_model_id}")
    print("=" * 80)

    if args.dry_run:
        selected_gpu = (args.gpu_id or 0, 0)
        print(f"[DRY-RUN] GPU: {selected_gpu[0]}")
    else:
        selected_gpu = _select_gpu(args.gpu_id, args.min_free_mib, args.gpu_wait)
        if selected_gpu is None:
            print("❌ Nenhuma GPU disponível.")
            sys.exit(1)
        if args.gpu_id is not None:
            print(f"GPU fixa selecionada: física {selected_gpu[0]}")
        else:
            print(f"GPU selecionada automaticamente: física {selected_gpu[0]} com {selected_gpu[1]} MiB livres")

    gpu_env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(selected_gpu[0])}

    # Download modelo HF
    if not args.dry_run:
        hf_ckpt_path, hf_config = _download_hf_checkpoint(args.hf_model_id, hf_cache)
    else:
        hf_ckpt_path = hf_cache / "best_siam.pt"
        hf_config = {}
        print(f"[DRY-RUN] Checkpoint HF seria baixado para: {hf_ckpt_path}")

    # Loss: argumento > config do modelo > fallback
    loss_type = args.loss_type or hf_config.get("loss_type") or "subcenter_cosface"
    print(f"Loss: {loss_type}")
    print(f"Splits: {splits} | Épocas: {args.epochs} | Warmup professor: {args.professor_warmup_epochs} épocas")
    print(f"Checkpoint root: {ckpt_root}")
    print("=" * 80)

    for split_idx in splits:
        split_dir = _ensure_split(split_idx, args.protocol, args.rvl_data_root, args.pairs_per_class)
        pairs_csv = split_dir / "train_pairs.csv"

        run_transfer = f"Sprint4_S{split_idx}_{loss_type}_transfer_E{args.epochs}"
        run_direct   = f"Sprint4_S{split_idx}_{loss_type}_direct_E{args.epochs}"

        cmd_transfer = _build_cmd(
            run_name=run_transfer,
            pairs_csv=pairs_csv,
            args=args,
            loss_type=loss_type,
            init_checkpoint=hf_ckpt_path,
            professor_lr=args.professor_lr,
            warmup_steps=professor_warmup_steps,
        )
        cmd_direct = _build_cmd(
            run_name=run_direct,
            pairs_csv=pairs_csv,
            args=args,
            loss_type=loss_type,
            init_checkpoint=None,
            professor_lr=args.professor_lr,
            warmup_steps=professor_warmup_steps,
        )

        print("-" * 80)
        print(f"[Split {split_idx}] GPU {selected_gpu[0]}")
        print(f"  [TRANSFER] {run_transfer}")
        print(f"  [DIRECT  ] {run_direct}")

        if args.dry_run:
            print("  CMD transfer:", " ".join(cmd_transfer))
            print("  CMD direct  :", " ".join(cmd_direct))
            continue

        if not _should_skip(ckpt_root, run_transfer):
            subprocess.run(cmd_transfer, check=True, env=gpu_env)
            time.sleep(args.sleep)

        if not _should_skip(ckpt_root, run_direct):
            subprocess.run(cmd_direct, check=True, env=gpu_env)
            time.sleep(args.sleep)

    print("\n✅ Sprint 4 concluída.")


if __name__ == "__main__":
    main()
