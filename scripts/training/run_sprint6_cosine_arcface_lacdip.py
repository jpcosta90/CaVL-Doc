#!/usr/bin/env python3
"""
Sprint 6 CaVL-Doc: ArcFace com CosineAnnealingWarmRestarts.

Mesma estrutura do run_final_arcface_lacdip.py (dois estágios + early stopping),
mas troca ReduceLROnPlateau por CosineAnnealingWarmRestarts para evitar mínimos locais.

O scheduler reinicia a LR ciclicamente (T_0 épocas no primeiro ciclo, multiplicado por
T_mult nos ciclos seguintes), permitindo que o modelo escape de mínimos locais ao longo
do treino. O early stopping ainda controla quando parar.

Projeto W&B: CaVL-Doc_LA-CDIP_Sprint6_CosineWarm
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
PREP_SCRIPT    = WORKSPACE_ROOT / "scripts" / "utils" / "prepare_protocol_split.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _resolve_checkpoint_root(user_value: Optional[str]) -> Path:
    if user_value:
        return Path(user_value).expanduser().resolve()
    if Path("/mnt/large/checkpoints").exists():
        return Path("/mnt/large/checkpoints")
    return (WORKSPACE_ROOT / "checkpoints").resolve()


def _checkpoint_epoch(ckpt_path: Path) -> int:
    """Retorna a última época completa (0-indexed) do checkpoint, ou -1 em falha."""
    try:
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return int(ckpt.get("epoch", -1))
    except Exception:
        return -1


def _checkpoint_done(ckpt_path: Path, max_epochs: int, patience: int) -> bool:
    """True se o checkpoint atingiu max_epochs OU o early stopping disparou."""
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


def _prepare_split(
    data_root: str,
    metadata_root: str,
    split_idx: int,
    exclude: List[int],
    force: bool,
    prebuilt_base: Optional[str],
) -> Path:
    excl_tag = "none" if not exclude else "-".join(str(s) for s in sorted(exclude))

    # 1. Tenta diretório pré-construído (Sprint 3 ou outro passado pelo usuário)
    if prebuilt_base:
        candidates = [
            Path(prebuilt_base) / f"sprint3_zsl_val_{split_idx}_train_excl_{excl_tag}",
            Path(prebuilt_base) / f"final_val_{split_idx}_excl_{excl_tag}",
        ]
        for candidate in candidates:
            if (candidate / "train_pairs.csv").exists():
                print(f"  [SPLIT] Reutilizando {candidate.name}")
                return candidate

    split_dir = WORKSPACE_ROOT / "data" / "generated_splits" / f"final_val_{split_idx}_excl_{excl_tag}"
    if not force and (split_dir / "train_pairs.csv").exists() and (split_dir / "validation_pairs.csv").exists():
        print(f"  [SPLIT] {split_dir.name} já existe, pulando geração.")
        return split_dir
    cmd = [
        sys.executable, str(PREP_SCRIPT),
        "--data-root", metadata_root,
        "--output-dir", str(split_dir),
        "--val-split-idx", str(split_idx),
        "--exclude-train-splits", ",".join(str(s) for s in exclude),
    ]
    print(f"  [PREP] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return split_dir


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
) -> List[str]:
    professor_lr    = args.professor_lr if teacher_on else 0.0
    warmup_steps    = args.teacher_warmup_steps if teacher_on else 999_999
    easy_steps      = 0 if teacher_on else 999_999

    cmd = [
        python_bin, str(TRAIN_SCRIPT),
        "--use-wandb",
        "--wandb-project",       wandb_project,
        "--wandb-run-name",      run_name,
        "--dataset-name",        "LA-CDIP",
        "--model-name",          "InternVL3-2B",
        "--pairs-csv",           str(pairs_csv),
        "--base-image-dir",      base_image_dir,
        "--loss-type",           args.loss_type,
        "--optimizer-type",      "adamw",
        "--scheduler-type",      "cosine_warm",
        "--student-lr",          str(args.student_lr),
        "--professor-lr",        str(professor_lr),
        "--margin",              str(args.margin),
        "--scale",               str(args.scale),
        "--epochs",              str(stage_epochs),
        "--max-steps-per-epoch", str(args.max_steps_per_epoch),
        "--student-batch-size",  str(batch_size),
        "--candidate-pool-size", str(args.candidate_pool_size),
        "--gradient-accumulation-steps", str(grad_accum),
        "--num-workers",         str(num_workers),
        "--patience",            str(args.patience),
        "--lr-t0",               str(args.lr_t0),
        "--lr-t-mult",           str(args.lr_t_mult),
        "--projection-output-dim", "1536",
        "--max-num-image-tokens", str(args.max_num_image_tokens),
        "--cut-layer",           str(args.cut_layer),
        "--pooler-type",         "attention",
        "--head-type",           "mlp",
        "--num-queries",         str(args.num_queries),
        "--baseline-alpha",      str(args.baseline_alpha),
        "--entropy-coeff",       str(args.entropy_coeff),
        "--professor-warmup-steps", str(warmup_steps),
        "--easy-mining-steps",   str(easy_steps),
        "--val-subset-size",     str(args.val_subset_size),
        "--seed",                str(args.seed),
    ]
    if resume_from is not None:
        cmd += ["--resume-from", str(resume_from)]
    elif init_from is not None:
        cmd += ["--init-from-checkpoint", str(init_from)]
    return cmd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Treinamento final CaVL-Doc: ArcFace com early stopping em dois estágios."
    )
    p.add_argument("--python-bin",        default=sys.executable)
    p.add_argument("--wandb-project",     default="CaVL-Doc_LA-CDIP_Sprint6_CosineWarm")
    p.add_argument("--loss-type",         default="arcface",
                   help="Função de perda (default: arcface)")
    p.add_argument("--splits",            default="0,1,2,3,4")
    p.add_argument("--lacdip-data-root",  required=True,
                   help="Raiz do dataset LA-CDIP (onde ficam splits.csv e protocol.csv)")
    p.add_argument("--metadata-root",     default=None,
                   help="Diretório com splits.csv e protocol.csv, se diferente de --lacdip-data-root")
    p.add_argument("--base-image-dir",    required=True)
    p.add_argument("--prebuilt-splits-base", default=None,
                   help="Base dos splits já gerados (ex: data/generated_splits). "
                        "Procura sprint3_zsl_val_{N}_train_excl_5 e final_val_{N}_excl_5 antes de gerar novos.")
    p.add_argument("--exclude-train-splits", default="5",
                   help="Splits excluídos do treino (default: 5 para manter holdout final)")
    p.add_argument("--rebuild-splits",    action="store_true",
                   help="Força reconstrução dos CSVs de split mesmo que já existam")
    p.add_argument("--checkpoint-root",   default=None)
    p.add_argument("--run-suffix",        default="")

    # Fases
    p.add_argument("--phase1-epochs",     type=int,   default=50,
                   help="Máximo de épocas da fase 1 (student only); early stopping pode parar antes")
    p.add_argument("--phase2-epochs",     type=int,   default=30,
                   help="Máximo de épocas da fase 2 (com teacher); early stopping pode parar antes")
    p.add_argument("--teacher-warmup-steps", type=int, default=140,
                   help="Steps iniciais onde o professor aprende mas não influencia o student")

    # Early stopping / LR
    p.add_argument("--patience",          type=int,   default=10,
                   help="Early stopping: épocas sem melhora para encerrar o treino")
    p.add_argument("--lr-t0",             type=int,   default=10,
                   help="CosineWarmRestarts: épocas do primeiro ciclo (T_0)")
    p.add_argument("--lr-t-mult",         type=int,   default=2,
                   help="CosineWarmRestarts: fator multiplicador dos ciclos (T_mult)")

    # Otimização
    p.add_argument("--student-lr",        type=float, default=1e-5)
    p.add_argument("--max-steps-per-epoch", type=int, default=140)
    p.add_argument("--val-subset-size",   type=int,   default=1036)
    p.add_argument("--seed",              type=int,   default=42)

    # Loss hyperparams
    p.add_argument("--margin",            type=float, default=0.35)
    p.add_argument("--scale",             type=float, default=24.0)

    # Batch / accum — fase 1 (sem teacher, pode usar batch maior)
    p.add_argument("--phase1-batch-size", type=int,   default=8)
    p.add_argument("--phase1-grad-accum", type=int,   default=2)
    p.add_argument("--phase1-num-workers", type=int,  default=2)

    # Batch / accum — fase 2 (com teacher, mais memória)
    p.add_argument("--phase2-batch-size", type=int,   default=4)
    p.add_argument("--phase2-grad-accum", type=int,   default=3)
    p.add_argument("--phase2-num-workers", type=int,  default=0)

    # Teacher config
    p.add_argument("--professor-lr",      type=float, default=5e-5)
    p.add_argument("--baseline-alpha",    type=float, default=0.05)
    p.add_argument("--entropy-coeff",     type=float, default=0.01)
    p.add_argument("--candidate-pool-size", type=int, default=8)

    # Arquitetura
    p.add_argument("--max-num-image-tokens", type=int, default=12)
    p.add_argument("--cut-layer",         type=int,   default=27)
    p.add_argument("--num-queries",       type=int,   default=1)

    # GPU
    p.add_argument("--gpu-id",            type=int,   default=None)
    p.add_argument("--min-free-mib",      type=int,   default=10_000)
    p.add_argument("--gpu-wait",          type=float, default=0.0)
    p.add_argument("--sleep",             type=float, default=3.0)
    p.add_argument("--dry-run",           action="store_true")

    args = p.parse_args()

    splits          = [int(s) for s in _parse_csv_list(args.splits)]
    exclude_splits  = [int(s) for s in _parse_csv_list(args.exclude_train_splits)]
    checkpoint_root = _resolve_checkpoint_root(args.checkpoint_root)

    if not checkpoint_root.exists():
        raise FileNotFoundError(f"checkpoint_root não encontrado: {checkpoint_root}")

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

    print("=" * 80)
    print(f"Loss: {args.loss_type}  |  Splits: {splits}  |  Excluídos: {exclude_splits}")
    print(f"Fase 1: max {args.phase1_epochs} épocas  |  Fase 2: max {args.phase2_epochs} épocas")
    print(f"Early stop patience: {args.patience}  |  Cosine T_0: {args.lr_t0}  T_mult: {args.lr_t_mult}")
    print(f"Projeto W&B: {args.wandb_project}")
    print("=" * 80)

    for split_idx in splits:
        # Prepara CSV de treino/validação
        split_dir = _prepare_split(
            data_root=args.lacdip_data_root,
            metadata_root=args.metadata_root or args.lacdip_data_root,
            split_idx=split_idx,
            exclude=exclude_splits,
            force=args.rebuild_splits,
            prebuilt_base=args.prebuilt_splits_base,
        )
        pairs_csv = split_dir / "train_pairs.csv"

        suffix    = f"_{args.run_suffix}" if args.run_suffix else ""
        run_p1    = f"Sprint6_S{split_idx}_{args.loss_type}_fase1_E{args.phase1_epochs}{suffix}"
        run_p2    = f"Sprint6_S{split_idx}_{args.loss_type}_fase2_teacher_E{args.phase2_epochs}{suffix}"
        ckpt_p1   = checkpoint_root / run_p1  / "last_checkpoint.pt"
        ckpt_p2   = checkpoint_root / run_p2  / "last_checkpoint.pt"
        best_p1   = checkpoint_root / run_p1  / "best_model.pt"
        if not best_p1.exists():
            best_p1 = checkpoint_root / run_p1 / "best_siam.pt"  # fallback legado

        print(f"\n{'='*80}")
        print(f"Split {split_idx}")
        print(f"{'='*80}")

        # ── Fase 1: student only ──────────────────────────────────────────
        cmd_p1 = _build_cmd(
            python_bin=args.python_bin, run_name=run_p1, pairs_csv=pairs_csv,
            base_image_dir=args.base_image_dir, wandb_project=args.wandb_project,
            args=args, stage_epochs=args.phase1_epochs, teacher_on=False,
            batch_size=args.phase1_batch_size, grad_accum=args.phase1_grad_accum,
            num_workers=args.phase1_num_workers,
        )
        print(f"\n[FASE 1] {run_p1}")
        if not args.dry_run:
            print(" ".join(cmd_p1))

        if _checkpoint_done(ckpt_p1, args.phase1_epochs, args.patience):
            epoch_done = _checkpoint_epoch(ckpt_p1)
            print(f"  → Fase 1 completa (época {epoch_done + 1}). Pulando.")
        elif ckpt_p1.exists():
            epoch_done = _checkpoint_epoch(ckpt_p1)
            print(f"  → Retomando fase 1 da época {epoch_done + 1}...")
            cmd_p1_resume = _build_cmd(
                python_bin=args.python_bin, run_name=run_p1, pairs_csv=pairs_csv,
                base_image_dir=args.base_image_dir, wandb_project=args.wandb_project,
                args=args, stage_epochs=args.phase1_epochs, teacher_on=False,
                batch_size=args.phase1_batch_size, grad_accum=args.phase1_grad_accum,
                num_workers=args.phase1_num_workers, resume_from=ckpt_p1,
            )
            if not args.dry_run:
                subprocess.run(cmd_p1_resume, check=True, env=gpu_env)
                time.sleep(args.sleep)
        else:
            if not args.dry_run:
                subprocess.run(cmd_p1, check=True, env=gpu_env)
                time.sleep(args.sleep)

        # ── Fase 2: com teacher ───────────────────────────────────────────
        if not best_p1.exists():
            print(f"  ⚠️  best_model.pt da fase 1 não encontrado em {best_p1}. Pulando fase 2.")
            continue

        cmd_p2 = _build_cmd(
            python_bin=args.python_bin, run_name=run_p2, pairs_csv=pairs_csv,
            base_image_dir=args.base_image_dir, wandb_project=args.wandb_project,
            args=args, stage_epochs=args.phase2_epochs, teacher_on=True,
            batch_size=args.phase2_batch_size, grad_accum=args.phase2_grad_accum,
            num_workers=args.phase2_num_workers, init_from=best_p1,
        )
        print(f"\n[FASE 2] {run_p2}")
        if not args.dry_run:
            print(" ".join(cmd_p2))

        if _checkpoint_done(ckpt_p2, args.phase2_epochs, args.patience):
            epoch_done = _checkpoint_epoch(ckpt_p2)
            print(f"  → Fase 2 completa (época {epoch_done + 1}). Pulando.")
        elif ckpt_p2.exists():
            epoch_done = _checkpoint_epoch(ckpt_p2)
            print(f"  → Retomando fase 2 da época {epoch_done + 1}...")
            cmd_p2_resume = _build_cmd(
                python_bin=args.python_bin, run_name=run_p2, pairs_csv=pairs_csv,
                base_image_dir=args.base_image_dir, wandb_project=args.wandb_project,
                args=args, stage_epochs=args.phase2_epochs, teacher_on=True,
                batch_size=args.phase2_batch_size, grad_accum=args.phase2_grad_accum,
                num_workers=args.phase2_num_workers, resume_from=ckpt_p2,
            )
            if not args.dry_run:
                subprocess.run(cmd_p2_resume, check=True, env=gpu_env)
                time.sleep(args.sleep)
        else:
            if not args.dry_run:
                subprocess.run(cmd_p2, check=True, env=gpu_env)
                time.sleep(args.sleep)

    print("\n✅ Concluído.")


if __name__ == "__main__":
    main()
