#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import math
import os
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional


TOP5_LOSSES = [
    "subcenter_cosface",
    "subcenter_arcface",
    "triplet",
    "contrastive",
    "circle",
]


def setup_runtime_env(runtime_root: Optional[str]) -> Dict[str, str]:
    env = os.environ.copy()
    if not runtime_root:
        return env

    root = os.path.abspath(runtime_root)
    home = os.path.expanduser("~")
    paths = {
        "TMPDIR": os.path.join(root, "tmp"),
        "XDG_CACHE_HOME": os.path.join(root, "xdg", "cache"),
        "XDG_CONFIG_HOME": os.path.join(root, "xdg", "config"),
        "XDG_DATA_HOME": os.path.join(root, "xdg", "data"),
        "WANDB_DIR": os.path.join(root, "wandb", "runs"),
        "WANDB_CACHE_DIR": os.path.join(root, "wandb", "cache"),
        "WANDB_CONFIG_DIR": os.path.join(root, "wandb", "config"),
        # Reutiliza cache padrão do HF (onde InternVL já foi baixado)
        "HF_HOME": os.path.join(home, ".cache", "huggingface"),
        "HUGGINGFACE_HUB_CACHE": os.path.join(home, ".cache", "huggingface", "hub"),
        "TRANSFORMERS_CACHE": os.path.join(home, ".cache", "huggingface", "hub"),
        "TORCH_HOME": os.path.join(root, "torch"),
        "MPLCONFIGDIR": os.path.join(root, "mpl"),
        "PIP_CACHE_DIR": os.path.join(root, "pip"),
        "PYTHONPYCACHEPREFIX": os.path.join(root, "pycache"),
    }

    for _, path in paths.items():
        os.makedirs(path, exist_ok=True)

    env.update(paths)
    return env


def _parse_nvidia_smi_free_memory() -> List[tuple[int, int]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    gpus: List[tuple[int, int]] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            gpu_index = int(parts[0])
            free_mib = int(float(re.sub(r"[^0-9.]", "", parts[1]) or 0))
        except ValueError:
            continue
        gpus.append((gpu_index, free_mib))
    return gpus


def _select_gpu(min_free_mib: int, wait_seconds: float, poll_interval: float) -> Optional[tuple[int, int]]:
    deadline = time.time() + max(0.0, wait_seconds)
    while True:
        gpus = _parse_nvidia_smi_free_memory()
        if gpus:
            selected = max(gpus, key=lambda item: (item[1], -item[0]))
            if selected[1] >= min_free_mib:
                return selected

        if wait_seconds <= 0.0 or time.time() >= deadline:
            return None

        time.sleep(max(1.0, poll_interval))


def setup_gpu_env(
    env: Dict[str, str],
    gpu_id: Optional[int],
    min_free_mib: int,
    wait_seconds: float,
    poll_interval: float,
) -> Optional[tuple[int, int]]:
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        return gpu_id, -1

    selected = _select_gpu(min_free_mib=min_free_mib, wait_seconds=wait_seconds, poll_interval=poll_interval)
    if selected is None:
        return None

    selected_gpu, free_mib = selected
    env["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
    return selected_gpu, free_mib


def _to_float(value: str, default: float) -> float:
    try:
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return default
        return numeric
    except (TypeError, ValueError):
        return default


def _to_int(value: str, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _latest_existing_file(pattern: str) -> Optional[str]:
    candidates = [path for path in glob.glob(pattern) if os.path.isfile(path)]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _resolve_pairs_csv(workspace_root: str, provided: str) -> str:
    checked: List[str] = []

    if provided:
        candidate_abs = provided if os.path.isabs(provided) else os.path.join(workspace_root, provided)
        checked.append(candidate_abs)
        if os.path.isfile(candidate_abs):
            return candidate_abs

    fixed_candidates = [
        os.path.join(workspace_root, "data", "LA-CDIP", "train_pairs.csv"),
        os.path.join(workspace_root, "data", "generated_splits", "LA-CDIP_zsl_split_1_5pairs", "train_pairs.csv"),
    ]
    for candidate in fixed_candidates:
        checked.append(candidate)
        if os.path.isfile(candidate):
            return candidate

    dynamic_patterns = [
        os.path.join(workspace_root, "data", "generated_splits", "LA-CDIP_*", "train_pairs.csv"),
        os.path.join(workspace_root, "data", "generated_splits", "LA-CDIP_*pairs", "train_pairs.csv"),
    ]
    for pattern in dynamic_patterns:
        latest = _latest_existing_file(pattern)
        if latest:
            return latest

    checked_preview = "\n - ".join(checked[:10])
    raise FileNotFoundError(
        "Não foi possível localizar train_pairs.csv automaticamente. Caminhos testados:\n"
        f" - {checked_preview}\n"
        "Use --pairs-csv com caminho absoluto ou relativo à raiz do repositório."
    )


def _resolve_base_image_dir(workspace_root: str, provided: str) -> str:
    checked: List[str] = []

    if provided:
        candidate = provided if os.path.isabs(provided) else os.path.join(workspace_root, provided)
        checked.append(candidate)
        if os.path.isdir(candidate):
            return candidate

    fallback_candidates = [
        "/mnt/data/la-cdip/data",
        "/mnt/nas/joaopaulo/la-cdip/data",
        os.path.join(workspace_root, "data", "LA-CDIP", "data"),
        os.path.join(workspace_root, "data", "LA-CDIP"),
    ]

    for candidate in fallback_candidates:
        checked.append(candidate)
        if os.path.isdir(candidate):
            return candidate

    checked_preview = "\n - ".join(checked)
    raise FileNotFoundError(
        "Não foi possível localizar base_image_dir automaticamente. Caminhos testados:\n"
        f" - {checked_preview}\n"
        "Use --base-image-dir com caminho absoluto válido."
    )


def load_best_configs(runs_csv_path: str, losses: List[str]) -> Dict[str, Dict[str, float]]:
    if not os.path.exists(runs_csv_path):
        raise FileNotFoundError(f"runs_raw.csv não encontrado: {runs_csv_path}")

    by_loss: Dict[str, List[Dict[str, str]]] = {loss: [] for loss in losses}
    with open(runs_csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            loss_name = (row.get("loss_type") or "").strip()
            if loss_name in by_loss:
                by_loss[loss_name].append(row)

    selected: Dict[str, Dict[str, float]] = {}
    for loss_name in losses:
        candidates = by_loss.get(loss_name, [])
        if not candidates:
            raise ValueError(f"Sem runs para loss '{loss_name}' em {runs_csv_path}")

        non_diverged = [row for row in candidates if not _to_bool(row.get("diverged", "False"))]
        filtered = non_diverged if non_diverged else candidates

        def _rank_key(row: Dict[str, str]):
            return (
                _to_float(row.get("eer_final", "inf"), float("inf")),
                _to_float(row.get("eer_stability", "inf"), float("inf")),
            )

        best_row = sorted(filtered, key=_rank_key)[0]
        selected[loss_name] = {
            "student_lr": _to_float(best_row.get("lr", "1e-5"), 1e-5),
            "margin": _to_float(best_row.get("margin", "0.5"), 0.5),
            "scale": _to_float(best_row.get("scale", "64.0"), 64.0),
            "num_sub_centers": _to_int(best_row.get("k", "3"), 3),
            "eer_final": _to_float(best_row.get("eer_final", "inf"), float("inf")),
            "eer_stability": _to_float(best_row.get("eer_stability", "inf"), float("inf")),
            "run_id": best_row.get("run_id", "unknown"),
        }

    return selected


def _is_resume_compatible(resume_path: str, expected: Dict[str, object]) -> tuple[bool, List[str]]:
    cfg_path = os.path.join(os.path.dirname(resume_path), "training_config.json")
    if not os.path.exists(cfg_path):
        return False, ["training_config.json ausente"]

    try:
        with open(cfg_path, "r", encoding="utf-8") as handle:
            cfg = json.load(handle)
    except Exception as exc:
        return False, [f"falha ao ler training_config.json: {exc}"]

    mismatches: List[str] = []
    for key, exp_value in expected.items():
        got_value = cfg.get(key)
        if isinstance(exp_value, float):
            try:
                if abs(float(got_value) - float(exp_value)) > 1e-12:
                    mismatches.append(f"{key}={got_value} (esperado {exp_value})")
            except Exception:
                mismatches.append(f"{key}={got_value} (esperado {exp_value})")
        else:
            if str(got_value) != str(exp_value):
                mismatches.append(f"{key}={got_value} (esperado {exp_value})")

    return len(mismatches) == 0, mismatches


def build_command(
    script_path: str,
    pairs_csv: str,
    base_image_dir: str,
    wandb_project: str,
    loss_name: str,
    config: Dict[str, float],
    epochs: int,
    max_steps_per_epoch: int,
    seed: int,
    with_professor_last5: bool,
    run_suffix: str,
    scheduler_type: str,
    patience: int,
    student_batch_size: int,
    gradient_accumulation_steps: int,
    candidate_pool_size: int,
    val_subset_size: int,
    val_samples_per_class: int,
    num_workers: int,
    weight_decay: float,
    max_num_image_tokens: int,
    checkpoint_root: str,
    auto_resume: bool,
) -> List[str]:
    warmup_steps = 5 * max_steps_per_epoch
    if not with_professor_last5:
        warmup_steps = (epochs + 1) * max_steps_per_epoch

    variant = "prof_last5_on" if with_professor_last5 else "prof_last5_off"
    run_name = (
        f"Sprint1_{loss_name}_k{int(config['num_sub_centers'])}"
        f"_E{epochs}_SPE{max_steps_per_epoch}_{variant}_seed{seed}"
    )
    if run_suffix:
        run_name = f"{run_name}_{run_suffix}"

    cmd = [
        sys.executable,
        script_path,
        "--use-wandb",
        "--wandb-project",
        wandb_project,
        "--wandb-run-name",
        run_name,
        "--dataset-name",
        "LA-CDIP",
        "--model-name",
        "InternVL3-2B",
        "--pairs-csv",
        pairs_csv,
        "--base-image-dir",
        base_image_dir,
        "--loss-type",
        loss_name,
        "--optimizer-type",
        "adamw",
        "--student-lr",
        f"{config['student_lr']:.12g}",
        "--margin",
        f"{config['margin']:.6f}",
        "--scale",
        f"{config['scale']:.6f}",
        "--num-sub-centers",
        str(int(config["num_sub_centers"])),
        "--professor-warmup-steps",
        str(warmup_steps),
        "--easy-mining-steps",
        str(warmup_steps),
        "--epochs",
        str(epochs),
        "--max-steps-per-epoch",
        str(max_steps_per_epoch),
        "--weight-decay",
        str(weight_decay),
        "--num-workers",
        str(num_workers),
        "--student-batch-size",
        str(student_batch_size),
        "--gradient-accumulation-steps",
        str(gradient_accumulation_steps),
        "--candidate-pool-size",
        str(candidate_pool_size),
        "--scheduler-type",
        scheduler_type,
        "--patience",
        str(patience),
        "--projection-output-dim",
        "1536",
        "--max-num-image-tokens",
        str(max_num_image_tokens),
        "--seed",
        str(seed),
    ]

    # Regra: val_subset_size > 0 limita validação; <= 0 usa validação completa (sem limite por subset)
    if val_subset_size > 0:
        cmd.extend(["--val-subset-size", str(val_subset_size)])
    else:
        cmd.extend(["--val-samples-per-class", str(val_samples_per_class)])

    resume_path = os.path.join(checkpoint_root, run_name, "last_checkpoint.pt")

    # Resume automático: continua do último checkpoint da mesma run se existir.
    if auto_resume and os.path.exists(resume_path):
        expected_cfg = {
            "dataset_name": "LA-CDIP",
            "model_name": "InternVL3-2B",
            "pairs_csv": pairs_csv,
            "base_image_dir": base_image_dir,
            "loss_type": loss_name,
            # Compara com os valores efetivamente enviados na CLI para evitar
            # falso negativo por diferença de arredondamento serializado.
            "student_lr": float(f"{config['student_lr']:.12g}"),
            "margin": float(f"{config['margin']:.6f}"),
            "scale": float(f"{config['scale']:.6f}"),
            "num_sub_centers": int(config["num_sub_centers"]),
            "epochs": epochs,
            "max_steps_per_epoch": max_steps_per_epoch,
            "scheduler_type": scheduler_type,
            "patience": patience,
            "student_batch_size": student_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "candidate_pool_size": candidate_pool_size,
            "weight_decay": weight_decay,
            "seed": seed,
        }
        ok_resume, reasons = _is_resume_compatible(resume_path, expected_cfg)
        if ok_resume:
            cmd.extend(["--resume-from", resume_path])
        else:
            print(f"⚠️ Auto-resume ignorado para '{run_name}' (config incompatível):")
            for reason in reasons[:8]:
                print(f"   - {reason}")

    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sprint 1 LA-CDIP: validação Top-5 (10 épocas) focada em losses"
    )
    parser.add_argument(
        "--runs-csv",
        default="scripts/optimization/coarse_search/configs/lacdip/fine_search/runs_raw.csv",
        help="CSV base para escolher melhor config de cada loss",
    )
    parser.add_argument("--pairs-csv", default="data/LA-CDIP/train_pairs.csv")
    parser.add_argument("--base-image-dir", default="/mnt/data/la-cdip/data")
    parser.add_argument("--wandb-project", default="CaVL-Doc_LA-CDIP_Sprint1_Top5Validation")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps-per-epoch", type=int, default=140)
    parser.add_argument("--student-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--candidate-pool-size", type=int, default=8)
    parser.add_argument("--val-subset-size", type=int, default=1036)
    parser.add_argument(
        "--val-samples-per-class",
        type=int,
        default=1000000,
        help="Usado quando --val-subset-size <= 0 para evitar downsampling e usar validação completa",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument(
        "--max-num-image-tokens",
        type=int,
        default=12,
        help="Quantidade máxima de image tokens por amostra (reduzir ajuda a baixar uso de VRAM).",
    )
    parser.add_argument(
        "--scheduler-type",
        default="plateau",
        choices=["step", "cosine", "plateau", "constant"],
        help="Scheduler usado no fine-tuning da Sprint 1",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Paciência de early stopping do treino",
    )
    parser.add_argument(
        "--losses",
        default=",".join(TOP5_LOSSES),
        help="Lista de losses separadas por vírgula",
    )
    parser.add_argument(
        "--seeds",
        default="42",
        help="Seeds separadas por vírgula. Ex.: 42,43,44",
    )
    parser.add_argument(
        "--professor-mode",
        default="off",
        choices=["off", "on", "both"],
        help="Controle do professor na Sprint 1: off (padrão), on ou both (ablação).",
    )
    parser.add_argument("--sleep", type=float, default=3.0, help="Pausa entre runs")
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Número total de shards/servidores em paralelo",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Índice deste shard (0..num_shards-1)",
    )
    parser.add_argument(
        "--run-suffix",
        default="",
        help="Sufixo opcional para diferenciar runs (ex.: unb, srvA)",
    )
    parser.add_argument(
        "--runtime-root",
        default=None,
        help="Diretório raiz para caches/logs temporários fora do $HOME",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="GPU física fixa para usar. Se omitido, o launcher escolhe a GPU com mais memória livre.",
    )
    parser.add_argument(
        "--gpu-min-free-mib",
        type=int,
        default=12000,
        help="Memória livre mínima em MiB para iniciar. Se nenhuma GPU atingir isso, o launcher espera.",
    )
    parser.add_argument(
        "--gpu-wait-seconds",
        type=float,
        default=0.0,
        help="Tempo máximo para esperar por uma GPU com memória livre suficiente. 0 = tenta uma vez.",
    )
    parser.add_argument(
        "--gpu-poll-seconds",
        type=float,
        default=15.0,
        help="Intervalo entre verificações de GPU quando estiver aguardando.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Apenas imprime comandos")
    parser.add_argument(
        "--checkpoint-root",
        default=None,
        help="Raiz de checkpoints. Se vazio, usa /mnt/large/checkpoints quando existir, senão checkpoints/",
    )
    parser.add_argument(
        "--auto-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retoma automaticamente runs compatíveis com last_checkpoint.pt existente",
    )
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num-shards deve ser >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index deve estar entre 0 e num_shards-1")

    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    script_path = os.path.join(workspace_root, "scripts", "training", "run_cavl_training.py")
    runs_csv_path = os.path.join(workspace_root, args.runs_csv)
    checkpoint_root = args.checkpoint_root
    if not checkpoint_root:
        checkpoint_root = "/mnt/large/checkpoints" if os.path.exists("/mnt/large/checkpoints") else "checkpoints"

    pairs_csv_path = _resolve_pairs_csv(workspace_root, args.pairs_csv)
    base_image_dir_path = _resolve_base_image_dir(workspace_root, args.base_image_dir)

    losses = [loss.strip() for loss in args.losses.split(",") if loss.strip()]
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]

    best_configs = load_best_configs(runs_csv_path, losses)
    run_env = setup_runtime_env(args.runtime_root)
    selected_gpu = setup_gpu_env(
        env=run_env,
        gpu_id=args.gpu_id,
        min_free_mib=args.gpu_min_free_mib,
        wait_seconds=args.gpu_wait_seconds,
        poll_interval=args.gpu_poll_seconds,
    )

    print("=" * 80)
    print("Sprint 1 | LA-CDIP | Top-5 validação quase final")
    if args.professor_mode == "off":
        professor_regime = "SEM professor nas 5 finais"
    elif args.professor_mode == "on":
        professor_regime = "COM professor nas 5 finais"
    else:
        professor_regime = "ablação com/sem professor nas 5 finais"
    print(f"Regime: 10 épocas, steps/época maior, {professor_regime}")
    print("=" * 80)
    print(f"pairs_csv: {pairs_csv_path}")
    print(f"base_image_dir: {base_image_dir_path}")
    if args.runtime_root:
        print(f"Runtime isolado fora do HOME: {os.path.abspath(args.runtime_root)}")
    if selected_gpu is None:
        print("GPU: nenhuma seleção automática aplicada (sem CUDA_VISIBLE_DEVICES explícito)")
    elif args.gpu_id is not None:
        print(f"GPU fixa selecionada: física {args.gpu_id}")
    else:
        print(f"GPU selecionada automaticamente: física {selected_gpu[0]} com {selected_gpu[1]} MiB livres")
    for loss_name in losses:
        config = best_configs[loss_name]
        print(
            f"- {loss_name}: run_id={config['run_id']} | "
            f"eer={config['eer_final']:.6f} | std={config['eer_stability']:.6f} | "
            f"lr={config['student_lr']:.6g} margin={config['margin']:.4f} "
            f"scale={config['scale']:.2f} k={int(config['num_sub_centers'])}"
        )

    all_jobs = []
    if args.professor_mode == "off":
        professor_variants = [False]
    elif args.professor_mode == "on":
        professor_variants = [True]
    else:
        professor_variants = [True, False]

    for loss_name in losses:
        for seed in seeds:
            for with_professor_last5 in professor_variants:
                all_jobs.append((loss_name, seed, with_professor_last5))

    shard_jobs = [
        job
        for index, job in enumerate(all_jobs)
        if (index % args.num_shards) == args.shard_index
    ]

    print(
        f"Jobs totais: {len(all_jobs)} | shard {args.shard_index}/{args.num_shards - 1}: {len(shard_jobs)} jobs"
    )

    for loss_name, seed, with_professor_last5 in shard_jobs:
        config = best_configs[loss_name]
        cmd = build_command(
            script_path=script_path,
            pairs_csv=pairs_csv_path,
            base_image_dir=base_image_dir_path,
            wandb_project=args.wandb_project,
            loss_name=loss_name,
            config=config,
            epochs=args.epochs,
            max_steps_per_epoch=args.max_steps_per_epoch,
            seed=seed,
            with_professor_last5=with_professor_last5,
            run_suffix=args.run_suffix,
            scheduler_type=args.scheduler_type,
            patience=args.patience,
            student_batch_size=args.student_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            candidate_pool_size=args.candidate_pool_size,
            val_subset_size=args.val_subset_size,
            val_samples_per_class=args.val_samples_per_class,
            num_workers=args.num_workers,
            weight_decay=args.weight_decay,
            max_num_image_tokens=args.max_num_image_tokens,
            checkpoint_root=checkpoint_root,
            auto_resume=args.auto_resume,
        )

        variant = "COM professor nas últimas 5" if with_professor_last5 else "SEM professor nas últimas 5"
        print("\n" + "-" * 80)
        print(f"Loss={loss_name} | Seed={seed} | Variante={variant}")
        print("CMD:", " ".join(cmd))

        if args.dry_run:
            continue

        subprocess.run(cmd, check=True, env=run_env)
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
