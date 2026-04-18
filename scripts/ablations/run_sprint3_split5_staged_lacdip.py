#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys
import time
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = WORKSPACE_ROOT / "scripts" / "training" / "run_cavl_training.py"
PREP_PROTOCOL_SPLIT_SCRIPT = WORKSPACE_ROOT / "scripts" / "utils" / "prepare_protocol_split.py"


@dataclass
class TeacherConfig:
    professor_lr: float
    baseline_alpha: float
    entropy_coeff: float
    candidate_pool_size: int
    student_batch_size: int
    source_run_name: str
    source_run_id: str
    source_best_eer: float


def _parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_checkpoint_root(user_value: Optional[str]) -> Path:
    if user_value:
        return Path(user_value).expanduser().resolve()
    if Path("/mnt/large/checkpoints").exists():
        return Path("/mnt/large/checkpoints")
    return (WORKSPACE_ROOT / "checkpoints").resolve()


def _latest_checkpoint_for_name_filter(checkpoint_root: Path, name_filter: str) -> Optional[Path]:
    filter_lower = name_filter.lower()
    candidates: List[Path] = []
    for ckpt_name in ["best_siam.pt", "last_checkpoint.pt"]:
        for candidate in checkpoint_root.rglob(ckpt_name):
            parent_name = candidate.parent.name.lower()
            if filter_lower in parent_name:
                candidates.append(candidate)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _prepare_protocol_based_split(
    val_split_idx: int,
    protocol: str,
    data_root: str,
    exclude_train_splits: List[int],
    force_rebuild: bool,
) -> Path:
    base_dir = WORKSPACE_ROOT / "data" / "generated_splits"
    excl_tag = "none" if not exclude_train_splits else "-".join(str(item) for item in sorted(exclude_train_splits))
    split_dir = base_dir / f"sprint3_zsl_val_{val_split_idx}_train_excl_{excl_tag}"
    train_csv = split_dir / "train_pairs.csv"
    val_csv = split_dir / "validation_pairs.csv"

    if train_csv.exists() and val_csv.exists() and not force_rebuild:
        return split_dir

    cmd = [
        sys.executable,
        str(PREP_PROTOCOL_SPLIT_SCRIPT),
        "--data-root",
        data_root,
        "--output-dir",
        str(split_dir),
        "--val-split-idx",
        str(val_split_idx),
        "--protocol",
        protocol,
        "--exclude-train-splits",
        ",".join(str(item) for item in exclude_train_splits),
    ]
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(
            f"Split protocol-based inválido em {split_dir}: train_pairs.csv/validation_pairs.csv ausentes"
        )
    return split_dir


def _extract_best_eer(summary: Dict[str, object]) -> Optional[float]:
    for key in [
        "val/best_eer",
        "best_eer",
        "metrics/best_eer",
        "eer_best",
        "best_val_eer",
        "val/eer",
        "eer",
        "metrics/eer",
    ]:
        if key in summary:
            raw = summary[key]
            if isinstance(raw, dict) or hasattr(raw, "get"):
                for nested_key in ("min", "value", "best", "latest"):
                    try:
                        nested_value = raw.get(nested_key)
                    except Exception:
                        nested_value = None
                    if nested_value is not None:
                        try:
                            return float(nested_value)
                        except (TypeError, ValueError):
                            continue
                try:
                    raw_as_dict = dict(raw)
                except Exception:
                    raw_as_dict = {}
                for nested_key in ("min", "value", "best", "latest"):
                    if nested_key in raw_as_dict:
                        try:
                            return float(raw_as_dict[nested_key])
                        except (TypeError, ValueError):
                            continue
            try:
                return float(raw)
            except (TypeError, ValueError):
                continue
    return None


def _extract_loss_type(run_name: str, run_config: Dict[str, object], run_summary: Dict[str, object]) -> str:
    for source in (run_config, run_summary):
        value = source.get("loss_type")
        if isinstance(value, str) and value.strip():
            return value.strip()

    name = (run_name or "").strip()
    m = re.match(r"^Sprint2_(.+?)_from_", name)
    if m:
        return m.group(1)
    m = re.match(r"^LA-CDIP_InternVL3-2B_(.+?)_\d{8}-\d{6}$", name)
    if m:
        return m.group(1)

    return ""


def _extract_from_name(run_name: str) -> Dict[str, float]:
    result: Dict[str, float] = {}
    plr = re.search(r"_plr([0-9eE+\-.]+)", run_name)
    pool = re.search(r"_pool(\d+)", run_name)
    ba = re.search(r"_ba([0-9eE+\-.]+)", run_name)
    ent = re.search(r"_ent([0-9eE+\-.]+)", run_name)

    if plr:
        try:
            result["professor_lr"] = float(plr.group(1))
        except ValueError:
            pass
    if pool:
        try:
            result["candidate_pool_size"] = float(pool.group(1))
        except ValueError:
            pass
    if ba:
        try:
            result["baseline_alpha"] = float(ba.group(1))
        except ValueError:
            pass
    if ent:
        try:
            result["entropy_coeff"] = float(ent.group(1))
        except ValueError:
            pass

    return result


def _extract_float(run_config: Dict[str, object], key_variants: List[str]) -> Optional[float]:
    for key in key_variants:
        if key in run_config:
            try:
                return float(run_config[key])
            except (TypeError, ValueError):
                continue
    return None


def _extract_int(run_config: Dict[str, object], key_variants: List[str]) -> Optional[int]:
    for key in key_variants:
        if key in run_config:
            try:
                return int(float(run_config[key]))
            except (TypeError, ValueError):
                continue
    return None


def _normalize_wandb_config_payload(raw: Dict[str, object]) -> Dict[str, object]:
    normalized: Dict[str, object] = {}
    for key, value in raw.items():
        if isinstance(value, dict) and "value" in value:
            normalized[key] = value.get("value")
        else:
            normalized[key] = value
    return normalized


def _load_run_config_from_file(run: object) -> Dict[str, object]:
    try:
        config_file = run.file("config.yaml")
        if not config_file:
            return {}

        downloaded = config_file.download(root=tempfile.mkdtemp(), replace=True)
        path = Path(downloaded.name)
        if not path.exists():
            return {}

        try:
            import yaml
        except ImportError:
            return {}

        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            parsed = yaml.safe_load(handle) or {}
        if not isinstance(parsed, dict):
            return {}
        parsed.pop("_wandb", None)
        return _normalize_wandb_config_payload(parsed)
    except Exception:
        return {}


def _merge_run_configs(primary: Dict[str, object], secondary: Dict[str, object]) -> Dict[str, object]:
    merged = dict(secondary)
    merged.update({k: v for k, v in primary.items() if v is not None})
    return merged


def _load_sprint2_runs(entity: str, project: str) -> List[Dict[str, object]]:
    import wandb

    api = wandb.Api()
    runs = list(api.runs(f"{entity}/{project}"))
    payload: List[Dict[str, object]] = []

    for run in runs:
        run_name = run.name or ""
        if "prof_off" in run_name.lower() or "baseoff" in run_name.lower():
            continue

        api_run_config = dict(run.config) if run.config else {}
        file_run_config = _load_run_config_from_file(run)
        run_config = _merge_run_configs(api_run_config, file_run_config)
        run_summary = dict(run.summary) if run.summary else {}
        best_eer = _extract_best_eer(run_summary)
        if best_eer is None:
            continue

        loss_type = _extract_loss_type(run_name=run_name, run_config=run_config, run_summary=run_summary)
        if not loss_type:
            continue

        parsed_from_name = _extract_from_name(run_name)
        professor_lr = _extract_float(run_config, ["professor-lr", "professor_lr"])
        baseline_alpha = _extract_float(run_config, ["baseline-alpha", "baseline_alpha"])
        entropy_coeff = _extract_float(run_config, ["entropy-coeff", "entropy_coeff"])
        candidate_pool_size = _extract_int(run_config, ["candidate-pool-size", "candidate_pool_size"])
        student_batch_size = _extract_int(run_config, ["student-batch-size", "student_batch_size"])

        if professor_lr is None:
            professor_lr = parsed_from_name.get("professor_lr")
        if baseline_alpha is None:
            baseline_alpha = parsed_from_name.get("baseline_alpha")
        if entropy_coeff is None:
            entropy_coeff = parsed_from_name.get("entropy_coeff")
        if candidate_pool_size is None and "candidate_pool_size" in parsed_from_name:
            candidate_pool_size = int(parsed_from_name["candidate_pool_size"])

        payload.append(
            {
                "run_name": run_name,
                "run_id": run.id,
                "sweep_id": (run.sweep.id if getattr(run, "sweep", None) else None),
                "loss_type": loss_type,
                "best_eer": float(best_eer),
                "professor_lr": (None if professor_lr is None else float(professor_lr)),
                "baseline_alpha": (None if baseline_alpha is None else float(baseline_alpha)),
                "entropy_coeff": (None if entropy_coeff is None else float(entropy_coeff)),
                "candidate_pool_size": (None if candidate_pool_size is None else int(candidate_pool_size)),
                "student_batch_size": (None if student_batch_size is None else int(student_batch_size)),
            }
        )

    return payload


def _select_losses_only(
    sprint2_rows: List[Dict[str, object]],
    loss_mode: str,
    explicit_losses: List[str],
) -> List[str]:
    if loss_mode == "explicit":
        losses = explicit_losses
    else:
        best_by_loss: Dict[str, Dict[str, object]] = {}
        for row in sprint2_rows:
            loss_name = str(row.get("loss_type", "") or "")
            if not loss_name:
                continue
            current_best = best_by_loss.get(loss_name)
            if current_best is None or float(row["best_eer"]) < float(current_best["best_eer"]):
                best_by_loss[loss_name] = row

        ranked = sorted(best_by_loss.items(), key=lambda kv: float(kv[1]["best_eer"]))
        top2 = [name for name, _ in ranked[:2]]
        losses = list(top2)
        if "contrastive" not in losses:
            losses.append("contrastive")

    losses_unique: List[str] = []
    for loss in losses:
        if loss not in losses_unique:
            losses_unique.append(loss)
    return losses_unique


def _select_teacher_and_losses(
    sprint2_rows: List[Dict[str, object]],
    loss_mode: str,
    explicit_losses: List[str],
) -> Tuple[TeacherConfig, List[str]]:
    if not sprint2_rows:
        raise RuntimeError("Nenhuma run válida da Sprint 2 encontrada para seleção automática.")

    losses_unique = _select_losses_only(
        sprint2_rows=sprint2_rows,
        loss_mode=loss_mode,
        explicit_losses=explicit_losses,
    )

    teacher_rows = [
        row for row in sprint2_rows
        if row.get("professor_lr") is not None
        and row.get("baseline_alpha") is not None
        and row.get("entropy_coeff") is not None
        and row.get("candidate_pool_size") is not None
        and row.get("student_batch_size") is not None
    ]
    if not teacher_rows:
        raise RuntimeError(
            "Runs da Sprint 2 encontradas, mas sem config completa de professor no W&B para auto-select. "
            "Para usar parâmetros fixos calibrados, rode com --no-auto-select-teacher."
        )

    global_best = min(teacher_rows, key=lambda row: float(row["best_eer"]))
    teacher = TeacherConfig(
        professor_lr=float(global_best["professor_lr"]),
        baseline_alpha=float(global_best["baseline_alpha"]),
        entropy_coeff=float(global_best["entropy_coeff"]),
        candidate_pool_size=int(global_best["candidate_pool_size"]),
        student_batch_size=int(global_best["student_batch_size"]),
        source_run_name=str(global_best["run_name"]),
        source_run_id=str(global_best["run_id"]),
        source_best_eer=float(global_best["best_eer"]),
    )

    if not losses_unique:
        raise RuntimeError("Nenhuma loss selecionada para Sprint 3.")

    return teacher, losses_unique


def _build_train_cmd(
    python_bin: str,
    run_name: str,
    pairs_csv: Path,
    base_image_dir: str,
    wandb_project: str,
    loss_type: str,
    args: argparse.Namespace,
    candidate_pool_size: int,
    student_batch_size: int,
    gradient_accumulation_steps: int,
    num_workers: int,
    stage_epochs: int,
    teacher_enabled: bool,
    init_from_checkpoint: Optional[Path],
    seed: int,
) -> List[str]:
    professor_lr = args.professor_lr if teacher_enabled else 0.0
    warmup_steps = args.professor_warmup_steps_on if teacher_enabled else args.professor_warmup_steps_off
    easy_steps = args.easy_mining_steps_on if teacher_enabled else args.easy_mining_steps_off

    cmd = [
        python_bin,
        str(TRAIN_SCRIPT),
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
        str(pairs_csv),
        "--base-image-dir",
        base_image_dir,
        "--loss-type",
        loss_type,
        "--optimizer-type",
        "adamw",
        "--scheduler-type",
        args.scheduler_type,
        "--student-lr",
        str(args.student_lr),
        "--professor-lr",
        str(professor_lr),
        "--margin",
        str(args.margin),
        "--scale",
        str(args.scale),
        "--num-sub-centers",
        str(args.num_sub_centers),
        "--epochs",
        str(stage_epochs),
        "--max-steps-per-epoch",
        str(args.max_steps_per_epoch),
        "--student-batch-size",
        str(student_batch_size),
        "--candidate-pool-size",
        str(candidate_pool_size),
        "--gradient-accumulation-steps",
        str(gradient_accumulation_steps),
        "--num-workers",
        str(num_workers),
        "--patience",
        str(args.patience),
        "--lr-reduce-factor",
        str(args.lr_reduce_factor),
        "--projection-output-dim",
        "1536",
        "--max-num-image-tokens",
        str(args.max_num_image_tokens),
        "--cut-layer",
        str(args.cut_layer),
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
        "--professor-warmup-steps",
        str(warmup_steps),
        "--easy-mining-steps",
        str(easy_steps),
        "--seed",
        str(seed),
    ]

    if args.val_subset_size > 0:
        cmd.extend(["--val-subset-size", str(args.val_subset_size)])
    else:
        cmd.extend(["--val-samples-per-class", str(args.val_samples_per_class)])

    if init_from_checkpoint is not None:
        cmd.extend(["--init-from-checkpoint", str(init_from_checkpoint)])

    if args.init_load_professor:
        cmd.append("--init-load-professor")

    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sprint 3 LA-CDIP: top2+contrastive em 5 splits com treino staged (professor ON/OFF)"
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--wandb-project", default="CaVL-Doc_LA-CDIP_Sprint3_Staged5x5")
    parser.add_argument("--wandb-entity", default="jpcosta1990-university-of-brasilia")
    parser.add_argument("--sprint2-project", default="CaVL-Doc_LA-CDIP_Sprint2_TeacherSweep")

    parser.add_argument("--loss-mode", default="top2-plus-contrastive", choices=["top2-plus-contrastive", "explicit"])
    parser.add_argument("--losses", default="subcenter_cosface,subcenter_arcface,contrastive")

    parser.add_argument("--splits", default="0,1,2,3,4")
    parser.add_argument("--protocol", default="zsl", choices=["zsl", "gzsl"])
    parser.add_argument("--lacdip-data-root", default="/mnt/data/la-cdip")
    parser.add_argument("--base-image-dir", default="/mnt/data/la-cdip/data")
    parser.add_argument(
        "--exclude-train-splits",
        default="5",
        help="Splits excluídos do treino em TODOS os cenários (ex.: 5 para manter holdout final).",
    )
    parser.add_argument(
        "--rebuild-protocol-splits",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Força reconstrução dos CSVs protocol-based mesmo que já existam.",
    )

    parser.add_argument("--teacher-epochs", type=int, default=5)
    parser.add_argument("--teacher-warmup-epochs", type=int, default=1,
                        help="Épocas de shadow warmup no início do stage ON (professor aprende sem influenciar student).")
    parser.add_argument("--student-only-epochs", type=int, default=10)
    parser.add_argument("--max-steps-per-epoch", type=int, default=140)

    parser.add_argument("--student-lr", type=float, default=1e-5)
    parser.add_argument("--scheduler-type", default="plateau", choices=["step", "cosine", "plateau", "constant"])
    parser.add_argument("--margin", type=float, default=0.35)
    parser.add_argument("--scale", type=float, default=24.0)
    parser.add_argument("--num-sub-centers", type=int, default=2)

    parser.add_argument("--student-batch-size", type=int, default=4)
    parser.add_argument("--candidate-pool-size", type=int, default=8)
    parser.add_argument("--warmup-student-batch-size", type=int, default=8)
    parser.add_argument("--warmup-gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--warmup-num-workers", type=int, default=2)
    parser.add_argument("--teacher-student-batch-size", type=int, default=4)
    parser.add_argument("--teacher-candidate-pool-size", type=int, default=8)
    parser.add_argument("--teacher-gradient-accumulation-steps", type=int, default=3)
    parser.add_argument("--teacher-num-workers", type=int, default=0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-subset-size", type=int, default=1036)
    parser.add_argument("--val-samples-per-class", type=int, default=1000000)

    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr-reduce-factor", type=float, default=0.5)
    parser.add_argument("--num-queries", type=int, default=1)
    parser.add_argument("--max-num-image-tokens", type=int, default=12)
    parser.add_argument("--cut-layer", type=int, default=27)

    parser.add_argument("--professor-lr", type=float, default=5e-5)
    parser.add_argument("--baseline-alpha", type=float, default=0.05)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--auto-select-teacher", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--professor-warmup-steps-on", type=int, default=0)
    parser.add_argument("--easy-mining-steps-on", type=int, default=0)
    parser.add_argument("--professor-warmup-steps-off", type=int, default=999999)
    parser.add_argument("--easy-mining-steps-off", type=int, default=999999)
    parser.add_argument("--init-load-professor", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument(
        "--source-init-mode",
        default="latest-by-loss",
        choices=["latest-by-loss", "none"],
        help="Checkpoint de inicialização do stage ON: busca último Sprint1 por loss ou inicia do zero.",
    )
    parser.add_argument("--source-filter-prefix", default="sprint1")

    parser.add_argument("--seeds", default="42")
    parser.add_argument("--run-suffix", default="")
    parser.add_argument("--sleep", type=float, default=3.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    losses = [item for item in _parse_csv_list(args.losses)]
    splits = [int(item) for item in _parse_csv_list(args.splits)]
    seeds = [int(item) for item in _parse_csv_list(args.seeds)]
    exclude_train_splits = [int(item) for item in _parse_csv_list(args.exclude_train_splits)]

    if not splits:
        raise ValueError("Nenhum split informado em --splits")
    if not seeds:
        raise ValueError("Nenhuma seed informada em --seeds")

    checkpoint_root = _resolve_checkpoint_root(args.checkpoint_root)
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"checkpoint_root não encontrado: {checkpoint_root}")

    if args.warmup_student_batch_size <= 0:
        raise ValueError("--warmup-student-batch-size precisa ser > 0")
    if args.teacher_student_batch_size <= 0:
        raise ValueError("--teacher-student-batch-size precisa ser > 0")
    if args.teacher_candidate_pool_size <= 0:
        raise ValueError("--teacher-candidate-pool-size precisa ser > 0")
    if args.warmup_gradient_accumulation_steps <= 0:
        raise ValueError("--warmup-gradient-accumulation-steps precisa ser > 0")
    if args.teacher_gradient_accumulation_steps <= 0:
        raise ValueError("--teacher-gradient-accumulation-steps precisa ser > 0")
    if args.warmup_num_workers < 0:
        raise ValueError("--warmup-num-workers não pode ser negativo")
    if args.teacher_num_workers < 0:
        raise ValueError("--teacher-num-workers não pode ser negativo")

    if args.auto_select_teacher:
        sprint2_rows = _load_sprint2_runs(entity=args.wandb_entity, project=args.sprint2_project)
        selected_teacher, selected_losses = _select_teacher_and_losses(
            sprint2_rows=sprint2_rows,
            loss_mode=args.loss_mode,
            explicit_losses=losses,
        )
        args.professor_lr = selected_teacher.professor_lr
        args.baseline_alpha = selected_teacher.baseline_alpha
        args.entropy_coeff = selected_teacher.entropy_coeff
        args.teacher_student_batch_size = selected_teacher.student_batch_size
        args.teacher_candidate_pool_size = selected_teacher.candidate_pool_size
        losses = selected_losses

        print("=" * 90)
        print("[AUTO] Melhor professor config (Sprint 2)")
        print(f"  run_name: {selected_teacher.source_run_name}")
        print(f"  run_id: {selected_teacher.source_run_id}")
        print(f"  best_eer: {selected_teacher.source_best_eer:.6f}")
        print(f"  professor_lr: {selected_teacher.professor_lr}")
        print(f"  baseline_alpha: {selected_teacher.baseline_alpha}")
        print(f"  entropy_coeff: {selected_teacher.entropy_coeff}")
        print(f"  student_batch_size: {selected_teacher.student_batch_size}")
        print(f"  candidate_pool_size: {selected_teacher.candidate_pool_size}")
        print(f"  losses (Sprint 3): {', '.join(losses)}")
        print("=" * 90)

    if args.teacher_warmup_epochs >= args.teacher_epochs:
        raise ValueError(
            f"--teacher-warmup-epochs ({args.teacher_warmup_epochs}) deve ser menor que "
            f"--teacher-epochs ({args.teacher_epochs}), senão o professor nunca é aplicado."
        )
    args.professor_warmup_steps_on = args.teacher_warmup_epochs * args.max_steps_per_epoch

    print("=" * 90)
    total_epochs = args.teacher_epochs + args.student_only_epochs
    active_epochs = args.teacher_epochs - args.teacher_warmup_epochs
    print(f"Sprint 3 | LA-CDIP | warmup={args.student_only_epochs} + shadow={args.teacher_warmup_epochs} + ativo={active_epochs} | total={total_epochs}")
    print("=" * 90)
    print(f"Splits: {splits}")
    print(f"Losses: {losses}")
    print(f"Seeds: {seeds}")
    print(f"Projeto W&B: {args.wandb_project}")
    print(f"Data root: {args.lacdip_data_root}")
    print(f"Split(s) excluído(s) do treino: {exclude_train_splits}")
    print(
        "Batch/pool stage OFF (warmup): "
        f"student_batch={args.warmup_student_batch_size}, candidate_pool={args.warmup_student_batch_size}, "
        f"grad_acc={args.warmup_gradient_accumulation_steps}, workers={args.warmup_num_workers}"
    )
    print(
        "Batch/pool stage ON (teacher): "
        f"student_batch={args.teacher_student_batch_size}, candidate_pool={args.teacher_candidate_pool_size}, "
        f"grad_acc={args.teacher_gradient_accumulation_steps}, workers={args.teacher_num_workers}"
    )

    for split_idx in splits:
        split_dir = _prepare_protocol_based_split(
            val_split_idx=split_idx,
            protocol=args.protocol,
            data_root=args.lacdip_data_root,
            exclude_train_splits=exclude_train_splits,
            force_rebuild=args.rebuild_protocol_splits,
        )
        pairs_csv = split_dir / "train_pairs.csv"

        for loss_name in losses:
            source_ckpt = None
            if args.source_init_mode == "latest-by-loss":
                source_filter = f"{args.source_filter_prefix}_{loss_name}"
                source_ckpt = _latest_checkpoint_for_name_filter(checkpoint_root, source_filter)
                if source_ckpt is None:
                    print(f"[WARN] Sem checkpoint fonte para loss={loss_name} ({source_filter}); stage ON inicia do zero.")

            for seed in seeds:
                suffix = f"_{args.run_suffix}" if args.run_suffix else ""
                base_name = f"Sprint3_S{split_idx}_{loss_name}_seed{seed}{suffix}"
                run_warmup = f"{base_name}_fase1_E{args.student_only_epochs}"
                run_on = f"{base_name}_fase2_profON_E{args.teacher_epochs}"
                run_off_cont = f"{base_name}_fase2_profOFF_E{args.teacher_epochs}"

                cmd_warmup = _build_train_cmd(
                    python_bin=args.python_bin,
                    run_name=run_warmup,
                    pairs_csv=pairs_csv,
                    base_image_dir=args.base_image_dir,
                    wandb_project=args.wandb_project,
                    loss_type=loss_name,
                    args=args,
                    candidate_pool_size=args.warmup_student_batch_size,
                    student_batch_size=args.warmup_student_batch_size,
                    gradient_accumulation_steps=args.warmup_gradient_accumulation_steps,
                    num_workers=args.warmup_num_workers,
                    stage_epochs=args.student_only_epochs,
                    teacher_enabled=False,
                    init_from_checkpoint=source_ckpt,
                    seed=seed,
                )

                ckpt_warmup = checkpoint_root / run_warmup / "last_checkpoint.pt"
                cmd_on = _build_train_cmd(
                    python_bin=args.python_bin,
                    run_name=run_on,
                    pairs_csv=pairs_csv,
                    base_image_dir=args.base_image_dir,
                    wandb_project=args.wandb_project,
                    loss_type=loss_name,
                    args=args,
                    candidate_pool_size=args.teacher_candidate_pool_size,
                    student_batch_size=args.teacher_student_batch_size,
                    gradient_accumulation_steps=args.teacher_gradient_accumulation_steps,
                    num_workers=args.teacher_num_workers,
                    stage_epochs=args.teacher_epochs,
                    teacher_enabled=True,
                    init_from_checkpoint=ckpt_warmup,
                    seed=seed,
                )

                cmd_off_cont = _build_train_cmd(
                    python_bin=args.python_bin,
                    run_name=run_off_cont,
                    pairs_csv=pairs_csv,
                    base_image_dir=args.base_image_dir,
                    wandb_project=args.wandb_project,
                    loss_type=loss_name,
                    args=args,
                    candidate_pool_size=args.teacher_candidate_pool_size,
                    student_batch_size=args.teacher_student_batch_size,
                    gradient_accumulation_steps=args.teacher_gradient_accumulation_steps,
                    num_workers=args.teacher_num_workers,
                    stage_epochs=args.teacher_epochs,
                    teacher_enabled=False,
                    init_from_checkpoint=ckpt_warmup,
                    seed=seed,
                )

                print("-" * 90)
                print(f"[Split {split_idx}] loss={loss_name} seed={seed}")
                print("[WARMUP  ]", " ".join(cmd_warmup))
                print("[ON      ]", " ".join(cmd_on))
                print("[OFF_CONT]", " ".join(cmd_off_cont))

                if args.dry_run:
                    continue

                ckpt_on      = checkpoint_root / run_on      / "last_checkpoint.pt"
                ckpt_off_cont_path = checkpoint_root / run_off_cont / "last_checkpoint.pt"

                if ckpt_warmup.exists():
                    print(f"[SKIP] {run_warmup} — last_checkpoint.pt já existe.")
                else:
                    subprocess.run(cmd_warmup, check=True)
                    if not ckpt_warmup.exists():
                        raise FileNotFoundError(f"Checkpoint warmup não encontrado para branches ON/OFF: {ckpt_warmup}")
                    time.sleep(args.sleep)

                if ckpt_on.exists():
                    print(f"[SKIP] {run_on} — last_checkpoint.pt já existe.")
                else:
                    subprocess.run(cmd_on, check=True)
                    time.sleep(args.sleep)

                if ckpt_off_cont_path.exists():
                    print(f"[SKIP] {run_off_cont} — last_checkpoint.pt já existe.")
                else:
                    subprocess.run(cmd_off_cont, check=True)
                    time.sleep(args.sleep)

    print("\n✅ Sprint 3 concluída.")


if __name__ == "__main__":
    main()
