#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple


DEFAULT_FALLBACK_LOSSES = ["subcenter_cosface", "subcenter_arcface", "contrastive"]


@dataclass
class Sprint1BestRun:
    loss_type: str
    run_name: str
    run_id: str
    created_at: str
    best_eer: float
    k: int


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


def _parse_nvidia_smi_free_memory() -> List[Tuple[int, int]]:
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

    gpus: List[Tuple[int, int]] = []
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


def _select_gpu(min_free_mib: int, wait_seconds: float, poll_interval: float) -> Optional[Tuple[int, int]]:
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
) -> Optional[Tuple[int, int]]:
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        return gpu_id, -1

    selected = _select_gpu(min_free_mib=min_free_mib, wait_seconds=wait_seconds, poll_interval=poll_interval)
    if selected is None:
        return None

    selected_gpu, free_mib = selected
    env["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
    return selected_gpu, free_mib


def _run_interruptible(cmd: List[str], env: Dict[str, str], cwd: Optional[str] = None) -> int:
    """Run command in its own process group so Ctrl+C propagates to child processes."""
    proc = subprocess.Popen(
        cmd,
        env=env,
        cwd=cwd,
        preexec_fn=os.setsid,
    )
    try:
        return proc.wait()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupção detectada. Encerrando subprocessos...", flush=True)
        try:
            os.killpg(proc.pid, signal.SIGINT)
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)
        raise


def _run_interruptible_check(cmd: List[str], env: Dict[str, str], cwd: Optional[str] = None) -> None:
    rc = _run_interruptible(cmd=cmd, env=env, cwd=cwd)
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


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


def _parse_csv_str_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_csv_float_list(raw: str) -> List[float]:
    values: List[float] = []
    for item in _parse_csv_str_list(raw):
        values.append(float(item))
    return values


def _parse_csv_int_list(raw: str) -> List[int]:
    values: List[int] = []
    for item in _parse_csv_str_list(raw):
        values.append(int(float(item)))
    return values


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


def _resolve_checkpoint_root(user_value: Optional[str], workspace_root: str) -> str:
    if user_value:
        return os.path.abspath(user_value)
    if os.path.exists("/mnt/large/checkpoints"):
        return "/mnt/large/checkpoints"
    return os.path.join(workspace_root, "checkpoints")


def _resolve_sprint1_best_csv(workspace_root: str, provided: str) -> str:
    candidate = provided if os.path.isabs(provided) else os.path.join(workspace_root, provided)
    if not os.path.exists(candidate):
        raise FileNotFoundError(f"CSV do Sprint 1 não encontrado: {candidate}")
    return candidate


def _parse_dt(value: str) -> datetime:
    raw = (value or "").strip()
    if not raw:
        return datetime.min
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min


def _infer_professor_mode_from_name(run_name: str) -> str:
    name = (run_name or "").lower()
    if "prof_last5_off" in name:
        return "off"
    if "prof_last5_on" in name:
        return "on"
    return ""


def _infer_loss_from_name(run_name: str) -> str:
    name = (run_name or "").strip()
    match = re.match(r"^Sprint1_(.+?)_k\d+_E\d+_SPE\d+_prof_last5_(?:on|off)_seed\d+", name)
    if match:
        return match.group(1)
    return ""


def _infer_k_from_name(run_name: str, default: int = 3) -> int:
    name = (run_name or "").strip()
    match = re.search(r"_k(\d+)_E", name)
    if not match:
        return default
    try:
        return int(match.group(1))
    except ValueError:
        return default


def _load_rows_from_sprint1_csv(sprint1_runs_csv: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(sprint1_runs_csv, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def _load_rows_from_wandb(entity: str, project: str) -> List[Dict[str, str]]:
    import wandb

    api = wandb.Api()
    runs = list(api.runs(f"{entity}/{project}"))

    rows: List[Dict[str, str]] = []
    for run in runs:
        attrs = run._attrs if hasattr(run, "_attrs") else {}
        updated_like = attrs.get("heartbeatAt") or attrs.get("updatedAt") or attrs.get("createdAt") or ""
        created_at = attrs.get("createdAt") or ""
        summary = dict(run.summary) if run.summary else {}

        best_eer_value = None
        for key in ["best_eer", "val/best_eer", "metrics/best_eer", "eer_best", "best_val_eer"]:
            if key in summary:
                best_eer_value = summary[key]
                break

        run_name = run.name or ""
        inferred_loss = _infer_loss_from_name(run_name)
        inferred_k = _infer_k_from_name(run_name, default=3)
        professor_label = _infer_professor_mode_from_name(run_name)

        rows.append(
            {
                "run_id": run.id or "",
                "run_name": run_name,
                "state": run.state or "",
                "created_at": str(updated_like),
                "raw_created_at": str(created_at),
                "loss_type": (
                    (summary.get("loss_type") or "").strip()
                    if isinstance(summary.get("loss_type"), str)
                    else inferred_loss
                ),
                "k": str(summary.get("num_sub_centers", summary.get("k", inferred_k))),
                "seed": str(summary.get("seed", "")),
                "professor_last5_label": professor_label,
                "best_eer": "" if best_eer_value is None else str(best_eer_value),
            }
        )

    return rows


def _read_json(path: str) -> Optional[Dict[str, object]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _find_training_config_for_run(checkpoint_root: str, run_name: str) -> Optional[Dict[str, object]]:
    direct_cfg = os.path.join(checkpoint_root, run_name, "training_config.json")
    cfg = _read_json(direct_cfg)
    if cfg is not None:
        return cfg

    candidate_dirs = [path for path in glob.glob(os.path.join(checkpoint_root, f"{run_name}*")) if os.path.isdir(path)]
    for run_dir in sorted(candidate_dirs, key=os.path.getmtime, reverse=True):
        cfg_path = os.path.join(run_dir, "training_config.json")
        cfg = _read_json(cfg_path)
        if cfg is not None:
            return cfg

    return None


def _matches_professor_policy(row: Dict[str, str], require_mode: str) -> bool:
    if require_mode == "any":
        return True
    label = (row.get("professor_last5_label") or "").strip().lower()
    if not label:
        label = _infer_professor_mode_from_name((row.get("run_name") or "").strip())
    if not label:
        flag = (row.get("professor_last5") or "").strip().lower()
        label = "on" if flag in {"1", "true", "yes", "y"} else "off"
    return label == require_mode


def _config_matches_reference(
    cfg: Optional[Dict[str, object]],
    expected_epochs: Optional[int],
    expected_scheduler: Optional[str],
    expected_patience: Optional[int],
    expected_seed: Optional[int],
    require_training_config: bool,
) -> bool:
    if cfg is None:
        return not require_training_config

    if expected_epochs is not None and _to_int(str(cfg.get("epochs", "-1")), -1) != expected_epochs:
        return False

    if expected_scheduler is not None:
        scheduler_value = str(cfg.get("scheduler_type", "")).strip().lower()
        if scheduler_value != expected_scheduler.strip().lower():
            return False

    if expected_patience is not None and _to_int(str(cfg.get("patience", "-1")), -1) != expected_patience:
        return False

    if expected_seed is not None and _to_int(str(cfg.get("seed", "-1")), -1) != expected_seed:
        return False

    return True


def load_sprint1_best_runs(
    sprint1_runs_csv: str,
    checkpoint_root: str,
    selected_losses: Optional[List[str]] = None,
    source: str = "wandb",
    wandb_entity: str = "jpcosta1990-university-of-brasilia",
    wandb_project: str = "CaVL-Doc_LA-CDIP_Sprint1_Top5Validation",
    only_last_n: int = 7,
    allowed_run_ids: Optional[List[str]] = None,
    require_professor_mode: str = "off",
    selection_mode: str = "all-unique",
    expected_epochs: Optional[int] = 10,
    expected_scheduler: Optional[str] = "plateau",
    expected_patience: Optional[int] = 5,
    expected_seed: Optional[int] = None,
    require_training_config: bool = False,
) -> Dict[str, Sprint1BestRun]:
    grouped: Dict[str, List[Sprint1BestRun]] = {}
    if source == "wandb":
        all_rows = _load_rows_from_wandb(entity=wandb_entity, project=wandb_project)
    else:
        all_rows = _load_rows_from_sprint1_csv(sprint1_runs_csv)

    all_rows = [row for row in all_rows if (row.get("state") or "").strip().lower() == "finished"]

    if allowed_run_ids:
        allowed_set = set(allowed_run_ids)
        all_rows = [row for row in all_rows if (row.get("run_id") or "").strip() in allowed_set]

    all_rows = sorted(all_rows, key=lambda row: _parse_dt(row.get("created_at", "")), reverse=True)

    if only_last_n > 0:
        all_rows = all_rows[:only_last_n]

    filtered_rows: List[Dict[str, str]] = []
    for row in all_rows:
        if not _matches_professor_policy(row, require_professor_mode):
            continue

        run_name = (row.get("run_name") or "").strip()
        if not run_name:
            continue

        cfg = _find_training_config_for_run(checkpoint_root=checkpoint_root, run_name=run_name)
        if not _config_matches_reference(
            cfg=cfg,
            expected_epochs=expected_epochs,
            expected_scheduler=expected_scheduler,
            expected_patience=expected_patience,
            expected_seed=expected_seed,
            require_training_config=require_training_config,
        ):
            continue

        filtered_rows.append(row)

    for row in filtered_rows:
        loss_type = (row.get("loss_type") or "").strip()
        if not loss_type:
            continue

        run_name = (row.get("run_name") or "").strip()
        run_id = (row.get("run_id") or "").strip()
        created_at = (row.get("created_at") or "").strip()
        best_eer = _to_float(row.get("best_eer", "inf"), float("inf"))
        k_value = _to_int(row.get("k", "3"), 3)

        entry = Sprint1BestRun(
            loss_type=loss_type,
            run_name=run_name,
            run_id=run_id,
            created_at=created_at,
            best_eer=best_eer,
            k=k_value,
        )
        grouped.setdefault(loss_type, []).append(entry)

    if not grouped:
        raise ValueError(
            "Nenhuma run válida encontrada no subconjunto do Sprint 1. "
            "Verifique filtros (--sprint1-*) ou atualize o CSV consolidado."
        )

    best_by_loss: Dict[str, Sprint1BestRun] = {}
    for loss_type, rows in grouped.items():
        ordered_rows = sorted(rows, key=lambda item: item.best_eer)
        chosen = ordered_rows[0]
        for candidate in ordered_rows:
            if _find_checkpoint_for_run(checkpoint_root, candidate.run_name):
                chosen = candidate
                break
        best_by_loss[loss_type] = chosen

    if selected_losses:
        missing = [loss for loss in selected_losses if loss not in best_by_loss]
        if missing:
            raise ValueError(f"Losses não encontradas no Sprint 1: {missing}")
        return {loss: best_by_loss[loss] for loss in selected_losses}

    ordered = sorted(best_by_loss.values(), key=lambda item: item.best_eer)
    if selection_mode == "top2-plus-contrastive":
        top2 = [item.loss_type for item in ordered[:2]]
        if "contrastive" in best_by_loss and "contrastive" not in top2:
            top2.append("contrastive")
        if not top2:
            top2 = [item.loss_type for item in ordered[:3]]
        return {loss: best_by_loss[loss] for loss in top2}

    return {item.loss_type: best_by_loss[item.loss_type] for item in ordered}


def _find_checkpoint_for_run(checkpoint_root: str, run_name: str) -> Optional[str]:
    run_dir = os.path.join(checkpoint_root, run_name)
    preferred = [
        os.path.join(run_dir, "best_siam.pt"),
        os.path.join(run_dir, "last_checkpoint.pt"),
    ]
    for candidate in preferred:
        if os.path.exists(candidate):
            return candidate

    for ckpt_name in ["best_siam.pt", "last_checkpoint.pt"]:
        pattern = os.path.join(checkpoint_root, f"{run_name}*", ckpt_name)
        matches = glob.glob(pattern)
        if matches:
            return max(matches, key=os.path.getmtime)

    return None


def _is_resume_compatible(resume_path: str, expected: Dict[str, object]) -> Tuple[bool, List[str]]:
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


def _build_jobs(
    selected: Dict[str, Sprint1BestRun],
    professor_lrs: List[float],
    candidate_pool_sizes: List[int],
    baseline_alphas: List[float],
    entropy_coeffs: List[float],
    seeds: List[int],
    max_jobs_per_loss: int,
) -> List[Dict[str, object]]:
    jobs: List[Dict[str, object]] = []

    for loss_type, best_run in selected.items():
        loss_jobs: List[Dict[str, object]] = []
        for seed in seeds:
            for professor_lr in professor_lrs:
                for pool_size in candidate_pool_sizes:
                    for baseline_alpha in baseline_alphas:
                        for entropy_coeff in entropy_coeffs:
                            loss_jobs.append(
                                {
                                    "loss_type": loss_type,
                                    "seed": seed,
                                    "professor_lr": professor_lr,
                                    "candidate_pool_size": pool_size,
                                    "baseline_alpha": baseline_alpha,
                                    "entropy_coeff": entropy_coeff,
                                    "sprint1_run": best_run,
                                }
                            )

        if max_jobs_per_loss > 0:
            loss_jobs = loss_jobs[:max_jobs_per_loss]

        jobs.extend(loss_jobs)

    return jobs


def build_command(
    script_path: str,
    pairs_csv: str,
    base_image_dir: str,
    wandb_project: str,
    run_suffix: str,
    checkpoint_root: str,
    auto_resume: bool,
    epochs: int,
    max_steps_per_epoch: int,
    student_lr: float,
    scheduler_type: str,
    patience: int,
    student_batch_size: int,
    gradient_accumulation_steps: int,
    val_subset_size: int,
    val_samples_per_class: int,
    weight_decay: float,
    num_workers: int,
    max_num_image_tokens: int,
    init_load_professor: bool,
    job: Dict[str, object],
) -> List[str]:
    sprint1_run: Sprint1BestRun = job["sprint1_run"]  # type: ignore[assignment]
    loss_type = str(job["loss_type"])
    seed = int(job["seed"])
    professor_lr = float(job["professor_lr"])
    candidate_pool_size = int(job["candidate_pool_size"])
    baseline_alpha = float(job["baseline_alpha"])
    entropy_coeff = float(job["entropy_coeff"])

    init_ckpt = _find_checkpoint_for_run(checkpoint_root, sprint1_run.run_name)
    if not init_ckpt:
        raise FileNotFoundError(
            f"Checkpoint base do Sprint 1 não encontrado para run '{sprint1_run.run_name}' em {checkpoint_root}"
        )

    run_name = (
        f"Sprint2_{loss_type}_from_{sprint1_run.run_id}"
        f"_plr{professor_lr:.0e}_pool{candidate_pool_size}"
        f"_ba{baseline_alpha:.3g}_ent{entropy_coeff:.3g}"
        f"_E{epochs}_SPE{max_steps_per_epoch}_seed{seed}"
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
        loss_type,
        "--optimizer-type",
        "adamw",
        "--scheduler-type",
        scheduler_type,
        "--student-lr",
        f"{student_lr:.12g}",
        "--professor-lr",
        f"{professor_lr:.12g}",
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
        "--patience",
        str(patience),
        "--projection-output-dim",
        "1536",
        "--max-num-image-tokens",
        str(max_num_image_tokens),
        "--seed",
        str(seed),
        "--baseline-alpha",
        f"{baseline_alpha:.12g}",
        "--entropy-coeff",
        f"{entropy_coeff:.12g}",
        "--init-from-checkpoint",
        init_ckpt,
    ]

    if init_load_professor:
        cmd.append("--init-load-professor")

    if val_subset_size > 0:
        cmd.extend(["--val-subset-size", str(val_subset_size)])
    else:
        cmd.extend(["--val-samples-per-class", str(val_samples_per_class)])

    resume_path = os.path.join(checkpoint_root, run_name, "last_checkpoint.pt")
    if auto_resume and os.path.exists(resume_path):
        expected_cfg = {
            "dataset_name": "LA-CDIP",
            "model_name": "InternVL3-2B",
            "pairs_csv": pairs_csv,
            "base_image_dir": base_image_dir,
            "loss_type": loss_type,
            "student_lr": float(f"{student_lr:.12g}"),
            "professor_lr": float(f"{professor_lr:.12g}"),
            "candidate_pool_size": candidate_pool_size,
            "baseline_alpha": float(f"{baseline_alpha:.12g}"),
            "entropy_coeff": float(f"{entropy_coeff:.12g}"),
            "epochs": epochs,
            "max_steps_per_epoch": max_steps_per_epoch,
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


def build_baseline_off_command(
    script_path: str,
    pairs_csv: str,
    base_image_dir: str,
    wandb_project: str,
    run_suffix: str,
    checkpoint_root: str,
    epochs: int,
    max_steps_per_epoch: int,
    candidate_pool_size: int,
    student_lr: float,
    scheduler_type: str,
    patience: int,
    student_batch_size: int,
    gradient_accumulation_steps: int,
    val_subset_size: int,
    val_samples_per_class: int,
    weight_decay: float,
    num_workers: int,
    max_num_image_tokens: int,
    loss_type: str,
    seed: int,
    sprint1_run: Sprint1BestRun,
) -> List[str]:
    init_ckpt = _find_checkpoint_for_run(checkpoint_root, sprint1_run.run_name)
    if not init_ckpt:
        raise FileNotFoundError(
            f"Checkpoint base do Sprint 1 não encontrado para run '{sprint1_run.run_name}' em {checkpoint_root}"
        )

    run_name = f"Sprint2_{loss_type}_from_{sprint1_run.run_id}_prof_off_E{epochs}_SPE{max_steps_per_epoch}_seed{seed}"
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
        loss_type,
        "--optimizer-type",
        "adamw",
        "--scheduler-type",
        scheduler_type,
        "--student-lr",
        f"{student_lr:.12g}",
        "--professor-lr",
        "0",
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
        "--patience",
        str(patience),
        "--projection-output-dim",
        "1536",
        "--max-num-image-tokens",
        str(max_num_image_tokens),
        "--seed",
        str(seed),
        "--init-from-checkpoint",
        init_ckpt,
    ]

    if val_subset_size > 0:
        cmd.extend(["--val-subset-size", str(val_subset_size)])
    else:
        cmd.extend(["--val-samples-per-class", str(val_samples_per_class)])

    return cmd


def _write_bayes_sweep_yaml(
    yaml_path: str,
    script_path: str,
    pairs_csv: str,
    base_image_dir: str,
    checkpoint_path: str,
    loss_type: str,
    epochs: int,
    max_steps_per_epoch: int,
    candidate_pool_size: int,
    student_lr: float,
    scheduler_type: str,
    patience: int,
    student_batch_size: int,
    gradient_accumulation_steps: int,
    val_subset_size: int,
    val_samples_per_class: int,
    weight_decay: float,
    num_workers: int,
    max_num_image_tokens: int,
    professor_lr_min: float,
    professor_lr_max: float,
    baseline_alpha_min: float,
    baseline_alpha_max: float,
    entropy_coeff_min: float,
    entropy_coeff_max: float,
    seed: int,
) -> None:
    val_arg = [
        "  - \"--val-subset-size\"",
        f"  - \"{val_subset_size}\"",
    ] if val_subset_size > 0 else [
        "  - \"--val-samples-per-class\"",
        f"  - \"{val_samples_per_class}\"",
    ]

    content = [
        "program: scripts/training/run_cavl_training.py",
        "method: bayes",
        "metric:",
        "  name: val/eer",
        "  goal: minimize",
        "parameters:",
        "  professor-lr:",
        "    distribution: log_uniform_values",
        f"    min: {professor_lr_min}",
        f"    max: {professor_lr_max}",
        "  baseline-alpha:",
        "    distribution: uniform",
        f"    min: {baseline_alpha_min}",
        f"    max: {baseline_alpha_max}",
        "  entropy-coeff:",
        "    distribution: uniform",
        f"    min: {entropy_coeff_min}",
        f"    max: {entropy_coeff_max}",
        "command:",
        "  - ${env}",
        "  - ${interpreter}",
        "  - ${program}",
        "  - \"--use-wandb\"",
        "  - \"--optimizer-type\"",
        "  - \"adamw\"",
        "  - \"--dataset-name\"",
        "  - \"LA-CDIP\"",
        "  - \"--model-name\"",
        "  - \"InternVL3-2B\"",
        "  - \"--pairs-csv\"",
        f"  - \"{pairs_csv}\"",
        "  - \"--base-image-dir\"",
        f"  - \"{base_image_dir}\"",
        "  - \"--loss-type\"",
        f"  - \"{loss_type}\"",
        "  - \"--student-lr\"",
        f"  - \"{student_lr}\"",
        "  - \"--scheduler-type\"",
        f"  - \"{scheduler_type}\"",
        "  - \"--patience\"",
        f"  - \"{patience}\"",
        "  - \"--epochs\"",
        f"  - \"{epochs}\"",
        "  - \"--max-steps-per-epoch\"",
        f"  - \"{max_steps_per_epoch}\"",
        "  - \"--weight-decay\"",
        f"  - \"{weight_decay}\"",
        "  - \"--num-workers\"",
        f"  - \"{num_workers}\"",
        "  - \"--student-batch-size\"",
        f"  - \"{student_batch_size}\"",
        "  - \"--gradient-accumulation-steps\"",
        f"  - \"{gradient_accumulation_steps}\"",
        "  - \"--candidate-pool-size\"",
        f"  - \"{candidate_pool_size}\"",
        "  - \"--projection-output-dim\"",
        "  - \"1536\"",
        "  - \"--max-num-image-tokens\"",
        f"  - \"{max_num_image_tokens}\"",
        "  - \"--seed\"",
        f"  - \"{seed}\"",
        "  - \"--init-from-checkpoint\"",
        f"  - \"{checkpoint_path}\"",
        *val_arg,
        "  - ${args}",
    ]

    with open(yaml_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(content) + "\n")


def _run_bayes_sweep_for_loss(
    *,
    workspace_root: str,
    run_env: Dict[str, str],
    wandb_project: str,
    wandb_entity: str,
    runs_per_loss: int,
    dry_run: bool,
    script_path: str,
    pairs_csv: str,
    base_image_dir: str,
    checkpoint_root: str,
    loss_type: str,
    sprint1_run: Sprint1BestRun,
    epochs: int,
    max_steps_per_epoch: int,
    candidate_pool_size: int,
    student_lr: float,
    scheduler_type: str,
    patience: int,
    student_batch_size: int,
    gradient_accumulation_steps: int,
    val_subset_size: int,
    val_samples_per_class: int,
    weight_decay: float,
    num_workers: int,
    max_num_image_tokens: int,
    professor_lr_min: float,
    professor_lr_max: float,
    baseline_alpha_min: float,
    baseline_alpha_max: float,
    entropy_coeff_min: float,
    entropy_coeff_max: float,
    seed: int,
) -> None:
    init_ckpt = _find_checkpoint_for_run(checkpoint_root, sprint1_run.run_name)
    if not init_ckpt:
        raise FileNotFoundError(
            f"Checkpoint base do Sprint 1 não encontrado para run '{sprint1_run.run_name}' em {checkpoint_root}"
        )

    run_env = run_env.copy()
    run_env["WANDB_RUN_GROUP"] = f"sprint2-{loss_type}"

    with tempfile.TemporaryDirectory(prefix=f"sprint2_bayes_{loss_type}_", dir=workspace_root) as tmp_dir:
        yaml_path = os.path.join(tmp_dir, f"sweep_{loss_type}.yaml")
        _write_bayes_sweep_yaml(
            yaml_path=yaml_path,
            script_path=script_path,
            pairs_csv=pairs_csv,
            base_image_dir=base_image_dir,
            checkpoint_path=init_ckpt,
            loss_type=loss_type,
            epochs=epochs,
            max_steps_per_epoch=max_steps_per_epoch,
            candidate_pool_size=candidate_pool_size,
            student_lr=student_lr,
            scheduler_type=scheduler_type,
            patience=patience,
            student_batch_size=student_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            val_subset_size=val_subset_size,
            val_samples_per_class=val_samples_per_class,
            weight_decay=weight_decay,
            num_workers=num_workers,
            max_num_image_tokens=max_num_image_tokens,
            professor_lr_min=professor_lr_min,
            professor_lr_max=professor_lr_max,
            baseline_alpha_min=baseline_alpha_min,
            baseline_alpha_max=baseline_alpha_max,
            entropy_coeff_min=entropy_coeff_min,
            entropy_coeff_max=entropy_coeff_max,
            seed=seed,
        )

        sweep_cmd = ["wandb", "sweep", yaml_path, "--project", wandb_project, "--entity", wandb_entity]
        print("SWEEP CMD:", " ".join(sweep_cmd))
        if dry_run:
            return

        sweep_proc = subprocess.run(
            sweep_cmd,
            check=True,
            text=True,
            capture_output=True,
            env=run_env,
            cwd=workspace_root,
        )
        sweep_output = (sweep_proc.stdout or "") + "\n" + (sweep_proc.stderr or "")
        match = re.search(r"wandb agent\s+([^\s]+)", sweep_output)
        if not match:
            raise RuntimeError(f"Não foi possível extrair o SWEEP_ID do output do wandb sweep.\n{sweep_output}")
        sweep_id = match.group(1).strip()

        agent_cmd = ["wandb", "agent", sweep_id, "--count", str(runs_per_loss)]
        print("AGENT CMD:", " ".join(agent_cmd))
        _run_interruptible_check(agent_cmd, env=run_env, cwd=workspace_root)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sprint 2 LA-CDIP: teacher sweep a partir dos melhores checkpoints do Sprint 1"
    )
    parser.add_argument(
        "--sprint1-runs-csv",
        default="analysis/sprint1_report/sprint1_runs_raw.csv",
        help="CSV consolidado de runs do Sprint 1",
    )
    parser.add_argument(
        "--sprint1-source",
        default="wandb",
        choices=["wandb", "csv"],
        help="Fonte das runs do Sprint 1 para seleção da base do Sprint 2",
    )
    parser.add_argument(
        "--sprint1-wandb-entity",
        default="jpcosta1990-university-of-brasilia",
        help="Entity W&B do Sprint 1",
    )
    parser.add_argument(
        "--sprint1-wandb-project",
        default="CaVL-Doc_LA-CDIP_Sprint1_Top5Validation",
        help="Projeto W&B do Sprint 1",
    )
    parser.add_argument(
        "--sprint1-only-last-n",
        type=int,
        default=7,
        help="Considera apenas as N runs mais recentes do Sprint 1 antes de selecionar losses (0 = todas)",
    )
    parser.add_argument(
        "--sprint1-allowed-run-ids",
        default="",
        help="Lista explícita de run_ids do Sprint 1 (separados por vírgula). Quando preenchido, ignora as demais runs.",
    )
    parser.add_argument(
        "--sprint1-require-professor-mode",
        default="off",
        choices=["any", "off", "on"],
        help="Filtro do regime professor no Sprint 1 para definir a base do Sprint 2.",
    )
    parser.add_argument(
        "--sprint1-selection-mode",
        default="all-unique",
        choices=["all-unique", "top2-plus-contrastive"],
        help="Como transformar as últimas runs válidas em losses-base do Sprint 2.",
    )
    parser.add_argument(
        "--sprint1-reference-epochs",
        type=int,
        default=10,
        help="Fixa o número de épocas esperado no Sprint 1 (0 = ignora)",
    )
    parser.add_argument(
        "--sprint1-reference-scheduler",
        default="plateau",
        help="Scheduler esperado no Sprint 1 (vazio = ignora)",
    )
    parser.add_argument(
        "--sprint1-reference-patience",
        type=int,
        default=5,
        help="Patience esperado no Sprint 1 (0 = ignora)",
    )
    parser.add_argument(
        "--sprint1-reference-seed",
        type=int,
        default=None,
        help="Seed esperada no Sprint 1 (omitido = ignora)",
    )
    parser.add_argument(
        "--sprint1-require-training-config",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Se true, descarta runs sem training_config.json localizável no checkpoint_root.",
    )
    parser.add_argument(
        "--losses",
        default="",
        help="Losses fixas separadas por vírgula (ex.: subcenter_cosface,triplet,contrastive). Vazio = auto (top-2 + contrastive)",
    )
    parser.add_argument("--pairs-csv", default="data/LA-CDIP/train_pairs.csv")
    parser.add_argument("--base-image-dir", default="/mnt/data/la-cdip/data")
    parser.add_argument("--wandb-project", default="CaVL-Doc_LA-CDIP_Sprint2_TeacherSweep")
    parser.add_argument("--wandb-entity", default="jpcosta1990-university-of-brasilia")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps-per-epoch", type=int, default=50)
    parser.add_argument("--student-lr", type=float, default=1e-5)
    parser.add_argument("--scheduler-type", default="plateau", choices=["step", "cosine", "plateau", "constant"])
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--student-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=3)
    parser.add_argument("--val-subset-size", type=int, default=1036)
    parser.add_argument("--val-samples-per-class", type=int, default=1000000)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-num-image-tokens", type=int, default=12)

    parser.add_argument("--professor-lrs", default="5e-5,1e-4")
    parser.add_argument("--candidate-pool-sizes", default="4")
    parser.add_argument("--baseline-alphas", default="0.01,0.05")
    parser.add_argument("--entropy-coeffs", default="0.005,0.02")
    parser.add_argument("--seeds", default="42")
    parser.add_argument(
        "--max-jobs-per-loss",
        type=int,
        default=2,
        help="Limite de combinações por loss (0 = sem limite)",
    )
    parser.add_argument("--bayes-runs-per-loss", type=int, default=5)
    parser.add_argument("--run-baseline-off", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--sleep", type=float, default=3.0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--run-suffix", default="")
    parser.add_argument("--runtime-root", default=None)

    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--gpu-min-free-mib", type=int, default=12000)
    parser.add_argument("--gpu-wait-seconds", type=float, default=0.0)
    parser.add_argument("--gpu-poll-seconds", type=float, default=15.0)

    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--init-load-professor", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num-shards deve ser >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index deve estar entre 0 e num_shards-1")

    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    script_path = os.path.join(workspace_root, "scripts", "training", "run_cavl_training.py")
    sprint1_runs_csv = _resolve_sprint1_best_csv(workspace_root, args.sprint1_runs_csv)
    checkpoint_root = _resolve_checkpoint_root(args.checkpoint_root, workspace_root)
    pairs_csv_path = _resolve_pairs_csv(workspace_root, args.pairs_csv)
    base_image_dir_path = _resolve_base_image_dir(workspace_root, args.base_image_dir)

    explicit_losses = _parse_csv_str_list(args.losses) if args.losses.strip() else None
    allowed_run_ids = _parse_csv_str_list(args.sprint1_allowed_run_ids) if args.sprint1_allowed_run_ids.strip() else None

    selected = load_sprint1_best_runs(
        sprint1_runs_csv=sprint1_runs_csv,
        checkpoint_root=checkpoint_root,
        selected_losses=explicit_losses,
        source=args.sprint1_source,
        wandb_entity=args.sprint1_wandb_entity,
        wandb_project=args.sprint1_wandb_project,
        only_last_n=args.sprint1_only_last_n,
        allowed_run_ids=allowed_run_ids,
        require_professor_mode=args.sprint1_require_professor_mode,
        selection_mode=args.sprint1_selection_mode,
        expected_epochs=(args.sprint1_reference_epochs if args.sprint1_reference_epochs > 0 else None),
        expected_scheduler=(args.sprint1_reference_scheduler.strip() if args.sprint1_reference_scheduler.strip() else None),
        expected_patience=(args.sprint1_reference_patience if args.sprint1_reference_patience > 0 else None),
        expected_seed=args.sprint1_reference_seed,
        require_training_config=args.sprint1_require_training_config,
    )

    if not selected:
        selected = {
            loss: Sprint1BestRun(loss_type=loss, run_name="", run_id="", created_at="", best_eer=float("inf"), k=3)
            for loss in DEFAULT_FALLBACK_LOSSES
        }

    seeds = _parse_csv_int_list(args.seeds)
    seed = seeds[0] if seeds else 42
    candidate_pool_sizes = _parse_csv_int_list(args.candidate_pool_sizes)
    candidate_pool_size = candidate_pool_sizes[0] if candidate_pool_sizes else 8
    loss_items = list(selected.items())
    shard_loss_items = [item for idx, item in enumerate(loss_items) if (idx % args.num_shards) == args.shard_index]

    professor_lrs = _parse_csv_float_list(args.professor_lrs)
    baseline_alphas = _parse_csv_float_list(args.baseline_alphas)
    entropy_coeffs = _parse_csv_float_list(args.entropy_coeffs)

    professor_lr_min = min(professor_lrs) if professor_lrs else 5e-5
    professor_lr_max = max(professor_lrs) if professor_lrs else 1e-4
    baseline_alpha_min = min(baseline_alphas) if baseline_alphas else 0.01
    baseline_alpha_max = max(baseline_alphas) if baseline_alphas else 0.05
    entropy_coeff_min = min(entropy_coeffs) if entropy_coeffs else 0.005
    entropy_coeff_max = max(entropy_coeffs) if entropy_coeffs else 0.02

    run_env = setup_runtime_env(args.runtime_root)
    selected_gpu = setup_gpu_env(
        env=run_env,
        gpu_id=args.gpu_id,
        min_free_mib=args.gpu_min_free_mib,
        wait_seconds=args.gpu_wait_seconds,
        poll_interval=args.gpu_poll_seconds,
    )

    print("=" * 100)
    print("Sprint 2 | LA-CDIP | Teacher Sweep a partir do Sprint 1")
    print("=" * 100)
    print(f"sprint1_runs_csv: {sprint1_runs_csv}")
    print(f"sprint1_source: {args.sprint1_source}")
    print(f"sprint1_only_last_n: {args.sprint1_only_last_n}")
    print(f"sprint1_require_professor_mode: {args.sprint1_require_professor_mode}")
    print(f"sprint1_selection_mode: {args.sprint1_selection_mode}")
    print(f"sprint1_require_training_config: {args.sprint1_require_training_config}")
    print(f"pairs_csv: {pairs_csv_path}")
    print(f"base_image_dir: {base_image_dir_path}")
    print(f"checkpoint_root: {checkpoint_root}")
    if args.runtime_root:
        print(f"Runtime isolado fora do HOME: {os.path.abspath(args.runtime_root)}")
    if selected_gpu is None:
        print("GPU: nenhuma seleção automática aplicada (sem CUDA_VISIBLE_DEVICES explícito)")
    elif args.gpu_id is not None:
        print(f"GPU fixa selecionada: física {args.gpu_id}")
    else:
        print(f"GPU selecionada automaticamente: física {selected_gpu[0]} com {selected_gpu[1]} MiB livres")

    print("\nLosses e bases do Sprint 1:")
    for loss_type, best in selected.items():
        print(
            f"- {loss_type}: run={best.run_name} | run_id={best.run_id} | created_at={best.created_at} "
            f"| best_eer={best.best_eer:.6f} | k={best.k}"
        )

    print(
        f"\nLosses totais: {len(loss_items)} | shard {args.shard_index}/{args.num_shards - 1}: {len(shard_loss_items)} losses"
    )

    for loss_type, sprint1_run in shard_loss_items:
        print("\n" + "-" * 100)
        print(f"Loss={loss_type} | Sprint1 run_id={sprint1_run.run_id}")

        _run_bayes_sweep_for_loss(
            workspace_root=workspace_root,
            run_env=run_env,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            runs_per_loss=args.bayes_runs_per_loss,
            dry_run=args.dry_run,
            script_path=script_path,
            pairs_csv=pairs_csv_path,
            base_image_dir=base_image_dir_path,
            checkpoint_root=checkpoint_root,
            loss_type=loss_type,
            sprint1_run=sprint1_run,
            epochs=args.epochs,
            max_steps_per_epoch=args.max_steps_per_epoch,
            candidate_pool_size=candidate_pool_size,
            student_lr=args.student_lr,
            scheduler_type=args.scheduler_type,
            patience=args.patience,
            student_batch_size=args.student_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            val_subset_size=args.val_subset_size,
            val_samples_per_class=args.val_samples_per_class,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            max_num_image_tokens=args.max_num_image_tokens,
            professor_lr_min=professor_lr_min,
            professor_lr_max=professor_lr_max,
            baseline_alpha_min=baseline_alpha_min,
            baseline_alpha_max=baseline_alpha_max,
            entropy_coeff_min=entropy_coeff_min,
            entropy_coeff_max=entropy_coeff_max,
            seed=seed,
        )

        if args.run_baseline_off:
            baseline_cmd = build_baseline_off_command(
                script_path=script_path,
                pairs_csv=pairs_csv_path,
                base_image_dir=base_image_dir_path,
                wandb_project=args.wandb_project,
                run_suffix=args.run_suffix,
                checkpoint_root=checkpoint_root,
                epochs=args.epochs,
                max_steps_per_epoch=args.max_steps_per_epoch,
                candidate_pool_size=candidate_pool_size,
                student_lr=args.student_lr,
                scheduler_type=args.scheduler_type,
                patience=args.patience,
                student_batch_size=args.student_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                val_subset_size=args.val_subset_size,
                val_samples_per_class=args.val_samples_per_class,
                weight_decay=args.weight_decay,
                num_workers=args.num_workers,
                max_num_image_tokens=args.max_num_image_tokens,
                loss_type=loss_type,
                seed=seed,
                sprint1_run=sprint1_run,
            )
            print("BASELINE CMD:", " ".join(baseline_cmd))
            if not args.dry_run:
                _run_interruptible_check(baseline_cmd, env=run_env, cwd=workspace_root)
                time.sleep(args.sleep)


if __name__ == "__main__":
    main()
