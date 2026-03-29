#!/usr/bin/env python3
import argparse
import csv
import glob
import math
import os
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
    paths = {
        "TMPDIR": os.path.join(root, "tmp"),
        "XDG_CACHE_HOME": os.path.join(root, "xdg", "cache"),
        "XDG_CONFIG_HOME": os.path.join(root, "xdg", "config"),
        "XDG_DATA_HOME": os.path.join(root, "xdg", "data"),
        "WANDB_DIR": os.path.join(root, "wandb", "runs"),
        "WANDB_CACHE_DIR": os.path.join(root, "wandb", "cache"),
        "WANDB_CONFIG_DIR": os.path.join(root, "wandb", "config"),
        "HF_HOME": os.path.join(root, "hf"),
        "HUGGINGFACE_HUB_CACHE": os.path.join(root, "hf", "hub"),
        "TRANSFORMERS_CACHE": os.path.join(root, "hf", "transformers"),
        "TORCH_HOME": os.path.join(root, "torch"),
        "MPLCONFIGDIR": os.path.join(root, "mpl"),
        "PIP_CACHE_DIR": os.path.join(root, "pip"),
        "PYTHONPYCACHEPREFIX": os.path.join(root, "pycache"),
    }

    for _, path in paths.items():
        os.makedirs(path, exist_ok=True)

    env.update(paths)
    return env


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

    return [
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
        "0.05",
        "--num-workers",
        "2",
        "--student-batch-size",
        "8",
        "--gradient-accumulation-steps",
        "2",
        "--candidate-pool-size",
        "8",
        "--val-subset-size",
        "512",
        "--scheduler-type",
        "constant",
        "--projection-output-dim",
        "1536",
        "--max-num-image-tokens",
        "12",
        "--seed",
        str(seed),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sprint 1 LA-CDIP: validação Top-5 (10 épocas) com ablação do professor nas 5 últimas épocas"
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
    parser.add_argument("--dry-run", action="store_true", help="Apenas imprime comandos")
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num-shards deve ser >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index deve estar entre 0 e num_shards-1")

    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    script_path = os.path.join(workspace_root, "scripts", "training", "run_cavl_training.py")
    runs_csv_path = os.path.join(workspace_root, args.runs_csv)

    pairs_csv_path = _resolve_pairs_csv(workspace_root, args.pairs_csv)
    base_image_dir_path = _resolve_base_image_dir(workspace_root, args.base_image_dir)

    losses = [loss.strip() for loss in args.losses.split(",") if loss.strip()]
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]

    best_configs = load_best_configs(runs_csv_path, losses)
    run_env = setup_runtime_env(args.runtime_root)

    print("=" * 80)
    print("Sprint 1 | LA-CDIP | Top-5 validação quase final")
    print("Regime: 10 épocas, steps/época maior, ablação com/sem professor nas 5 finais")
    print("=" * 80)
    print(f"pairs_csv: {pairs_csv_path}")
    print(f"base_image_dir: {base_image_dir_path}")
    if args.runtime_root:
        print(f"Runtime isolado fora do HOME: {os.path.abspath(args.runtime_root)}")
    for loss_name in losses:
        config = best_configs[loss_name]
        print(
            f"- {loss_name}: run_id={config['run_id']} | "
            f"eer={config['eer_final']:.6f} | std={config['eer_stability']:.6f} | "
            f"lr={config['student_lr']:.6g} margin={config['margin']:.4f} "
            f"scale={config['scale']:.2f} k={int(config['num_sub_centers'])}"
        )

    all_jobs = []
    for loss_name in losses:
        for seed in seeds:
            for with_professor_last5 in (True, False):
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
