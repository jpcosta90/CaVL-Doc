#!/usr/bin/env python3
"""
ArcDoc: treino final com todos os splits combinados.

Mesma estrutura da Sprint 6 (dois estágios + CosineAnnealingWarmRestarts + ArcFace),
mas usando um único conjunto de dados com todos os splits combinados e augmentation
offline (gerado por scripts/utils/prepare_arcdoc_training.py).

Projeto W&B: ArcDoc_Final
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT   = WORKSPACE_ROOT / "scripts" / "training" / "run_cavl_training.py"


# ---------------------------------------------------------------------------
# Helpers (idênticos ao Sprint 6)
# ---------------------------------------------------------------------------

def _resolve_checkpoint_root(user_value: Optional[str]) -> Path:
    if user_value:
        return Path(user_value).expanduser().resolve()
    if Path("/mnt/nas/joaopaulo/CaVL-Doc/checkpoints").exists():
        return Path("/mnt/nas/joaopaulo/CaVL-Doc/checkpoints")
    if Path("/mnt/large/checkpoints").exists():
        return Path("/mnt/large/checkpoints")
    return (WORKSPACE_ROOT / "checkpoints").resolve()


def _checkpoint_epoch(ckpt_path: Path) -> int:
    try:
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return int(ckpt.get("epoch", -1))
    except Exception:
        return -1


def _checkpoint_done(ckpt_path: Path, max_epochs: int, patience: int) -> bool:
    if not ckpt_path.exists():
        return False
    try:
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        epoch      = int(ckpt.get("epoch", -1))
        no_improve = int(ckpt.get("no_improve", 0))
        return epoch >= max_epochs - 1 or no_improve >= patience
    except Exception:
        return False


def _parse_nvidia_smi_free_memory() -> List[Tuple[int, int]]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode()
        return [(int(a.strip()), int(b.strip())) for a, b in (l.split(",") for l in out.strip().splitlines())]
    except Exception:
        return []


def _select_gpu(gpu_id: Optional[int], min_free_mib: int, wait_seconds: float) -> Optional[Tuple[int, int]]:
    if gpu_id is not None:
        gpus = _parse_nvidia_smi_free_memory()
        return next((g for g in gpus if g[0] == gpu_id), None)
    deadline = time.time() + max(0.0, wait_seconds)
    while True:
        gpus = _parse_nvidia_smi_free_memory()
        candidates = [(i, f) for i, f in gpus if f >= min_free_mib]
        if candidates:
            return max(candidates, key=lambda x: (x[1], -x[0]))
        if time.time() >= deadline:
            return max(gpus, key=lambda x: (x[1], -x[0])) if gpus else None
        print(f"  Aguardando GPU com ≥{min_free_mib} MiB livres...")
        time.sleep(30)


def _resolve_best_model(run_dir: Path) -> Optional[Path]:
    for candidate in [run_dir / "best_model.pt", run_dir / "best_siam.pt"]:
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------

def _build_cmd(
    python_bin: str,
    run_name: str,
    pairs_csv: Path,
    base_image_dir: str,
    wandb_project: str,
    args: argparse.Namespace,
    stage_epochs: int,
    teacher_on: bool,
    batch_size: int,
    grad_accum: int,
    num_workers: int,
    resume_from: Optional[Path] = None,
    init_from: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> List[str]:
    professor_lr = args.professor_lr if teacher_on else 0.0
    warmup_steps = args.teacher_warmup_steps if teacher_on else 999_999
    easy_steps   = 0 if teacher_on else 999_999

    val_base = args.val_base_image_dir or args.base_image_dir

    cmd = [
        python_bin, str(TRAIN_SCRIPT),
        "--use-wandb",
        "--wandb-project",            wandb_project,
        "--wandb-run-name",           run_name,
        "--dataset-name",             "LA-CDIP",
        "--model-name",               "InternVL3-2B",
        "--pairs-csv",                str(pairs_csv),
        "--base-image-dir",           base_image_dir,
        "--val-base-image-dir",       val_base,
        "--loss-type",                args.loss_type,
        "--optimizer-type",           "adamw",
        "--scheduler-type",           "cosine_warm",
        "--student-lr",               str(args.student_lr),
        "--professor-lr",             str(professor_lr),
        "--margin",                   str(args.margin),
        "--scale",                    str(args.scale),
        "--num-sub-centers",          str(args.num_sub_centers),
        "--epochs",                   str(stage_epochs),
        "--max-steps-per-epoch",      str(args.max_steps_per_epoch),
        "--student-batch-size",       str(batch_size),
        "--candidate-pool-size",      str(args.candidate_pool_size),
        "--gradient-accumulation-steps", str(grad_accum),
        "--num-workers",              str(num_workers),
        "--patience",                 str(args.patience),
        "--lr-t0",                    str(args.lr_t0),
        "--lr-t-mult",                str(args.lr_t_mult),
        "--projection-output-dim",    "1536",
        "--max-num-image-tokens",     str(args.max_num_image_tokens),
        "--cut-layer",                str(args.cut_layer),
        "--pooler-type",              "attention",
        "--head-type",                "mlp",
        "--num-queries",              str(args.num_queries),
        "--baseline-alpha",           str(args.baseline_alpha),
        "--entropy-coeff",            str(args.entropy_coeff),
        "--professor-warmup-steps",   str(warmup_steps),
        "--easy-mining-steps",        str(easy_steps),
        "--val-subset-size",          str(args.val_subset_size),
        "--seed",                     str(args.seed),
    ]
    if output_dir is not None:
        cmd += ["--output-dir", str(output_dir)]
    if resume_from is not None:
        cmd += ["--resume-from", str(resume_from)]
    elif init_from is not None:
        cmd += ["--init-from-checkpoint", str(init_from)]
    return cmd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _read_hf_token_file() -> Optional[str]:
    for candidate in [
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token",
    ]:
        if candidate.exists():
            token = candidate.read_text().strip()
            return token if token else None
    return None


def main() -> None:
    p = argparse.ArgumentParser(
        description="ArcDoc: treino final em todos os splits com augmentation."
    )
    p.add_argument("--python-bin",       default=sys.executable)
    p.add_argument("--wandb-project",    default="ArcDoc_SubArcFace_s32k3")
    p.add_argument("--run-name",         default="ArcDoc_SubArcFace_s32k3",
                   help="Nome base do run (sufixo _fase1/_fase2 adicionado automaticamente)")

    _default_data = WORKSPACE_ROOT / "data" / "generated_splits" / "final_split3"
    _hf_dataset   = "Jpcosta90/cavl-doc-lacdip-split3"

    # Dados
    p.add_argument("--from-hf", action="store_true",
                   help=f"Baixa o dataset do HF Hub ({_hf_dataset}) antes de treinar.")
    p.add_argument("--hf-dataset-repo", default=_hf_dataset,
                   help=f"Repo HF do dataset (default: {_hf_dataset})")
    p.add_argument("--hf-local-dir", default=None,
                   help="Diretório local para salvar o dataset baixado do HF. "
                        "Sobrescreve --train-csv, --base-image-dir e --val-base-image-dir "
                        "automaticamente. Ex: /mnt/nas/joaopaulo/CaVL-Doc/final_split3")
    p.add_argument("--train-csv",
                   default=str(_default_data / "train_pairs.csv"),
                   help="CSV de treino gerado por prepare_arcdoc_training.py")
    p.add_argument("--base-image-dir",
                   default=str(_default_data / "images_train"),
                   help="Diretório base das imagens de treino augmentadas (images_train/)")
    p.add_argument("--val-base-image-dir",
                   default=str(_default_data / "images_val"),
                   help="Diretório base das imagens de validação augmentadas (images_val/).")
    p.add_argument("--checkpoint-root",    default=None)

    # Fases — mesmos defaults da Sprint 6
    p.add_argument("--phase1-epochs",    type=int, default=50)
    p.add_argument("--phase2-epochs",    type=int, default=30)
    p.add_argument("--teacher-warmup-steps", type=int, default=140)

    # Early stopping / LR
    p.add_argument("--patience",         type=int,   default=9999)
    p.add_argument("--lr-t0",            type=int,   default=10)
    p.add_argument("--lr-t-mult",        type=int,   default=2)

    # Loss
    p.add_argument("--loss-type",        default="subcenter_arcface")
    p.add_argument("--num-sub-centers",  type=int,   default=3)
    p.add_argument("--margin",           type=float, default=0.35)
    p.add_argument("--scale",            type=float, default=32.0)

    # Otimização
    p.add_argument("--student-lr",       type=float, default=5e-5)
    p.add_argument("--max-steps-per-epoch", type=int, default=500)
    p.add_argument("--val-subset-size",  type=int,   default=2000)
    p.add_argument("--seed",             type=int,   default=42)

    # Batch — fase 1
    p.add_argument("--phase1-batch-size", type=int,  default=8)
    p.add_argument("--phase1-grad-accum", type=int,  default=2)
    p.add_argument("--phase1-num-workers", type=int, default=2)

    # Batch — fase 2
    p.add_argument("--phase2-batch-size", type=int,  default=4)
    p.add_argument("--phase2-grad-accum", type=int,  default=3)
    p.add_argument("--phase2-num-workers", type=int, default=0)

    # Teacher
    p.add_argument("--professor-lr",     type=float, default=5e-5)
    p.add_argument("--baseline-alpha",   type=float, default=0.05)
    p.add_argument("--entropy-coeff",    type=float, default=0.01)
    p.add_argument("--candidate-pool-size", type=int, default=8)

    # Arquitetura
    p.add_argument("--max-num-image-tokens", type=int, default=12)
    p.add_argument("--cut-layer",        type=int,   default=27)
    p.add_argument("--num-queries",      type=int,   default=1)

    # GPU
    p.add_argument("--gpu-id",           type=int,   default=3)
    p.add_argument("--min-free-mib",     type=int,   default=10_000)
    p.add_argument("--gpu-wait",         type=float, default=0.0)
    p.add_argument("--sleep",            type=float, default=3.0)
    p.add_argument("--dry-run",          action="store_true")

    args = p.parse_args()

    checkpoint_root = _resolve_checkpoint_root(args.checkpoint_root)
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"checkpoint_root não encontrado: {checkpoint_root}")

    # ── Download do HF Hub (opcional) ────────────────────────────────────────
    if args.from_hf:
        local_dir = Path(args.hf_local_dir) if args.hf_local_dir else Path(args.base_image_dir).parent
        print(f"Baixando dataset do HF Hub: {args.hf_dataset_repo}")
        print(f"  Destino: {local_dir}")
        try:
            from huggingface_hub import snapshot_download
            import os
            _hf_token = (
                os.environ.get("HF_TOKEN")
                or os.environ.get("HUGGINGFACE_TOKEN")
                or _read_hf_token_file()
            )
            snapshot_download(
                repo_id=args.hf_dataset_repo,
                repo_type="dataset",
                local_dir=str(local_dir),
                token=_hf_token or None,
            )
            print(f"  ✅ Dataset baixado em: {local_dir}")
        except Exception as e:
            print(f"  ❌ Falha ao baixar dataset: {e}")
            sys.exit(1)
        # Redireciona todos os paths para o diretório local especificado
        if args.hf_local_dir:
            args.train_csv        = str(local_dir / "train_pairs.csv")
            args.base_image_dir   = str(local_dir / "images_train")
            args.val_base_image_dir = str(local_dir / "images_val")

    pairs_csv = Path(args.train_csv)
    if not pairs_csv.exists():
        raise FileNotFoundError(f"train-csv não encontrado: {pairs_csv}")

    # GPU
    if args.dry_run:
        selected_gpu = (args.gpu_id or 0, 0)
    else:
        selected_gpu = _select_gpu(args.gpu_id, args.min_free_mib, args.gpu_wait)
        if selected_gpu is None:
            print("❌ Nenhuma GPU disponível.")
            sys.exit(1)
    gpu_env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(selected_gpu[0])}
    print(f"GPU: {selected_gpu[0]}  ({selected_gpu[1]} MiB livres)")

    run_p1  = f"{args.run_name}_fase1_E{args.phase1_epochs}"
    run_p2  = f"{args.run_name}_fase2_teacher_E{args.phase2_epochs}"
    ckpt_p1 = checkpoint_root / run_p1 / "last_checkpoint.pt"
    ckpt_p2 = checkpoint_root / run_p2 / "last_checkpoint.pt"

    print("=" * 80)
    print(f"ArcDoc — Treino Final")
    print(f"Dados:   {pairs_csv}")
    print(f"Fase 1:  max {args.phase1_epochs} épocas  |  Fase 2: max {args.phase2_epochs} épocas")
    print(f"LR: {args.student_lr}  |  Cosine T_0: {args.lr_t0}  T_mult: {args.lr_t_mult}")
    print(f"Projeto W&B: {args.wandb_project}")
    print("=" * 80)

    # ── Fase 1: student only ──────────────────────────────────────────────────
    cmd_p1 = _build_cmd(
        python_bin=args.python_bin, run_name=run_p1, pairs_csv=pairs_csv,
        base_image_dir=args.base_image_dir, wandb_project=args.wandb_project,
        args=args, stage_epochs=args.phase1_epochs, teacher_on=False,
        batch_size=args.phase1_batch_size, grad_accum=args.phase1_grad_accum,
        num_workers=args.phase1_num_workers, output_dir=checkpoint_root / run_p1,
    )
    print(f"\n[FASE 1] {run_p1}")
    if not args.dry_run:
        print(" ".join(cmd_p1))

    if _checkpoint_done(ckpt_p1, args.phase1_epochs, args.patience):
        print(f"  → Fase 1 completa (época {_checkpoint_epoch(ckpt_p1) + 1}). Pulando.")
    elif ckpt_p1.exists():
        epoch_done = _checkpoint_epoch(ckpt_p1)
        print(f"  → Retomando fase 1 da época {epoch_done + 1}...")
        cmd_p1_resume = _build_cmd(
            python_bin=args.python_bin, run_name=run_p1, pairs_csv=pairs_csv,
            base_image_dir=args.base_image_dir, wandb_project=args.wandb_project,
            args=args, stage_epochs=args.phase1_epochs, teacher_on=False,
            batch_size=args.phase1_batch_size, grad_accum=args.phase1_grad_accum,
            num_workers=args.phase1_num_workers, resume_from=ckpt_p1,
            output_dir=checkpoint_root / run_p1,
        )
        if not args.dry_run:
            subprocess.run(cmd_p1_resume, check=True, env=gpu_env)
            time.sleep(args.sleep)
    else:
        if not args.dry_run:
            subprocess.run(cmd_p1, check=True, env=gpu_env)
            time.sleep(args.sleep)

    # ── Resolve best_model após fase 1 ───────────────────────────────────────
    best_p1 = _resolve_best_model(checkpoint_root / run_p1)

    # ── Fase 2: com teacher ───────────────────────────────────────────────────
    if best_p1 is None:
        print(f"  ⚠️  best_model.pt da fase 1 não encontrado. Pulando fase 2.")
        sys.exit(0)

    print(f"  ✅ Iniciando fase 2 com: {best_p1.name}")

    cmd_p2 = _build_cmd(
        python_bin=args.python_bin, run_name=run_p2, pairs_csv=pairs_csv,
        base_image_dir=args.base_image_dir, wandb_project=args.wandb_project,
        args=args, stage_epochs=args.phase2_epochs, teacher_on=True,
        batch_size=args.phase2_batch_size, grad_accum=args.phase2_grad_accum,
        num_workers=args.phase2_num_workers, init_from=best_p1,
        output_dir=checkpoint_root / run_p2,
    )
    print(f"\n[FASE 2] {run_p2}")
    if not args.dry_run:
        print(" ".join(cmd_p2))

    if _checkpoint_done(ckpt_p2, args.phase2_epochs, args.patience):
        print(f"  → Fase 2 completa (época {_checkpoint_epoch(ckpt_p2) + 1}). Pulando.")
    elif ckpt_p2.exists():
        epoch_done = _checkpoint_epoch(ckpt_p2)
        print(f"  → Retomando fase 2 da época {epoch_done + 1}...")
        cmd_p2_resume = _build_cmd(
            python_bin=args.python_bin, run_name=run_p2, pairs_csv=pairs_csv,
            base_image_dir=args.base_image_dir, wandb_project=args.wandb_project,
            args=args, stage_epochs=args.phase2_epochs, teacher_on=True,
            batch_size=args.phase2_batch_size, grad_accum=args.phase2_grad_accum,
            num_workers=args.phase2_num_workers, resume_from=ckpt_p2,
            output_dir=checkpoint_root / run_p2,
        )
        if not args.dry_run:
            subprocess.run(cmd_p2_resume, check=True, env=gpu_env)
            time.sleep(args.sleep)
    else:
        if not args.dry_run:
            subprocess.run(cmd_p2, check=True, env=gpu_env)
            time.sleep(args.sleep)

    print("\n✅ ArcDoc concluído.")


if __name__ == "__main__":
    main()
