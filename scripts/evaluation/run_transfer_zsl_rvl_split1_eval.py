#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from cavl_doc.data.dataset import DocumentPairDataset
from cavl_doc.models.backbone_loader import load_model, warm_up_model
from cavl_doc.models.modeling_cavl import build_cavl_model
from cavl_doc.trainers.rl_trainer import build_loss, validate_siam_on_loader
from cavl_doc.utils.embedding_utils import prepare_inputs_for_multimodal_embedding


DEFAULT_TOP5_LOSSES = [
    "subcenter_cosface",
    "subcenter_arcface",
    "triplet",
    "contrastive",
    "circle",
]


def _load_config_from_checkpoint(ckpt_path: Path, ckpt: dict) -> dict:
    config = ckpt.get("config", {})
    if not config and "args" in ckpt:
        args_obj = ckpt["args"]
        config = vars(args_obj) if hasattr(args_obj, "__dict__") else {}

    if not config:
        json_path = ckpt_path.parent / "training_config.json"
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)

    if not config:
        raise ValueError("Não foi possível recuperar config do checkpoint (nem do training_config.json).")
    return config


def _build_encode_fn(backbone, tokenizer, device: str, cut_layer: int):
    def _encode_fn(_backbone, images, **kwargs):
        input_ids_list = []
        pixel_values_list = []
        image_flags_list = []

        if isinstance(images, torch.Tensor):
            if images.dim() == 5:
                images_list = [images[i] for i in range(images.shape[0])]
            else:
                images_list = [images]
        else:
            images_list = images

        for image in images_list:
            prepared = prepare_inputs_for_multimodal_embedding(
                backbone, tokenizer, image, "<image> Analyze this document"
            )
            input_ids_list.append(prepared["input_ids"][0])
            pixel_values_list.append(prepared["pixel_values"])
            image_flags_list.append(prepared["image_flags"])

        max_len = max(len(ids) for ids in input_ids_list)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        padded_ids = []
        padded_masks = []
        for ids in input_ids_list:
            pad_len = max_len - len(ids)
            padded_ids.append(
                torch.cat(
                    [ids, torch.full((pad_len,), pad_id, device=ids.device, dtype=ids.dtype)]
                )
            )
            padded_masks.append(
                torch.cat(
                    [torch.ones_like(ids), torch.zeros((pad_len,), device=ids.device, dtype=ids.dtype)]
                )
            )

        batch_input_ids = torch.stack(padded_ids).to(device)
        batch_attention_mask = torch.stack(padded_masks).to(device)
        batch_pixel_values = torch.cat(pixel_values_list, dim=0).to(device, dtype=torch.bfloat16)
        batch_image_flags = torch.cat(image_flags_list, dim=0).to(device)

        with torch.no_grad():
            outputs = backbone(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                pixel_values=batch_pixel_values,
                image_flags=batch_image_flags,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = outputs.hidden_states
        language_model = backbone.language_model.model
        index = cut_layer + 1 if len(hidden_states) == (len(language_model.layers) + 1) else cut_layer
        return hidden_states[index], None

    return _encode_fn


def _load_siam_from_checkpoint(ckpt_path: Path, backbone, tokenizer, device: str):
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    config = _load_config_from_checkpoint(ckpt_path, ckpt)

    cut_layer = int(config.get("cut_layer", 27))
    projection_output_dim = int(config.get("projection_output_dim", 1536))

    encode_fn = _build_encode_fn(backbone=backbone, tokenizer=tokenizer, device=device, cut_layer=cut_layer)

    siam = build_cavl_model(
        backbone=backbone,
        cut_layer=cut_layer,
        encode_fn=encode_fn,
        pool_dim=1536,
        proj_hidden=4096,
        proj_out=projection_output_dim,
        set_trainable=False,
        tokenizer=tokenizer,
        pooler_type=config.get("pooler_type", "attention"),
        head_type=config.get("head_type", "mlp"),
        num_queries=int(config.get("num_queries", 1)),
    )

    if "siam_pool" in ckpt and "siam_head" in ckpt:
        siam.pool.load_state_dict(ckpt["siam_pool"])
        siam.head.load_state_dict(ckpt["siam_head"])
        if "backbone_trainable" in ckpt and ckpt["backbone_trainable"]:
            siam.backbone.load_state_dict(ckpt["backbone_trainable"], strict=False)
    elif "model_state_dict" in ckpt:
        siam.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        raise ValueError("Checkpoint sem pesos reconhecidos (siam_pool/siam_head ou model_state_dict).")

    siam.to(device)
    siam.eval()
    return siam, config


def _resolve_checkpoint_root(user_value: Optional[str]) -> Path:
    if user_value:
        return Path(user_value).expanduser().resolve()
    if Path("/mnt/large/checkpoints").exists():
        return Path("/mnt/large/checkpoints")
    return (WORKSPACE_ROOT / "checkpoints").resolve()


def _find_latest_checkpoint_for_loss(checkpoint_root: Path, loss_name: str, name_filter: Optional[str]) -> Optional[Path]:
    candidates = []

    for candidate in checkpoint_root.rglob("best_siam.pt"):
        parent_name = candidate.parent.name.lower()
        if loss_name.lower() not in parent_name:
            continue
        if name_filter and name_filter.lower() not in parent_name:
            continue
        candidates.append(candidate)

    if not candidates:
        for candidate in checkpoint_root.rglob("last_checkpoint.pt"):
            parent_name = candidate.parent.name.lower()
            if loss_name.lower() not in parent_name:
                continue
            if name_filter and name_filter.lower() not in parent_name:
                continue
            candidates.append(candidate)

    if not candidates:
        return None

    return max(candidates, key=lambda path: path.stat().st_mtime)


def _build_eval_criterion(loss_type: str, cfg: dict, num_classes: int, device: str):
    margin = float(cfg.get("margin", 0.5))
    scale = float(cfg.get("scale", 64.0))
    num_sub_centers = int(cfg.get("num_sub_centers", cfg.get("k", 3)))
    std = float(cfg.get("std", 0.05))
    in_features = int(cfg.get("projection_output_dim", 1536))

    criterion = build_loss(
        loss_type,
        margin=margin,
        m=margin,
        s=scale,
        gamma=scale,
        num_classes=num_classes,
        k=num_sub_centers,
        std=std,
        in_features=in_features,
    ).to(device)
    return criterion


def _collate_fn(batch):
    img_a = [item["image_a"] for item in batch]
    img_b = [item["image_b"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)

    class_a = torch.tensor([item.get("class_a", 0) for item in batch], dtype=torch.long)
    class_b = torch.tensor([item.get("class_b", 0) for item in batch], dtype=torch.long)
    return img_a, img_b, labels, class_a, class_b


def _build_loader(dataset, batch_size: int, num_workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fn,
    )


def _measure_avg_batch_time(siam, loader, device: str, warmup_batches: int = 2, timed_batches: int = 8):
    timings = []

    for batch_index, batch in enumerate(loader):
        if batch_index >= warmup_batches + timed_batches:
            break

        images_a, images_b = batch[0], batch[1]

        torch.cuda.synchronize() if device == "cuda" else None
        start = time.perf_counter()
        with torch.no_grad():
            _ = siam(images_a)
            _ = siam(images_b)
        torch.cuda.synchronize() if device == "cuda" else None
        elapsed = time.perf_counter() - start

        if batch_index >= warmup_batches:
            timings.append(elapsed)

    if not timings:
        return None
    return float(sum(timings) / len(timings))


def _clear_cuda_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def main():
    parser = argparse.ArgumentParser(
        description="Teste de inferência de transfer learning no RVL-CDIP zero-shot split 1"
    )
    parser.add_argument("--checkpoint-path", default=None, help="Caminho para best_siam.pt ou last_checkpoint.pt (opcional)")
    parser.add_argument(
        "--checkpoint-root",
        default=None,
        help="Raiz para busca automática de checkpoints por loss (padrão: /mnt/large/checkpoints ou checkpoints)",
    )
    parser.add_argument(
        "--losses",
        default=",".join(DEFAULT_TOP5_LOSSES),
        help="Lista de losses separadas por vírgula para testar",
    )
    parser.add_argument(
        "--checkpoint-name-filter",
        default="Sprint1_",
        help="Filtro opcional no nome da pasta do checkpoint (ex.: Sprint1_). Use vazio para desabilitar",
    )
    parser.add_argument(
        "--checkpoint-map-json",
        default=None,
        help="JSON opcional com mapeamento loss->checkpoint_path. Ex.: '{\"triplet\":\"/path/best_siam.pt\"}'",
    )
    parser.add_argument(
        "--pairs-csv",
        default="data/generated_splits/RVL-CDIP_zsl_split_1/validation_pairs.csv",
        help="CSV de validação do split 1",
    )
    parser.add_argument(
        "--base-image-dir",
        default="/mnt/data/zs_rvl_cdip/data",
        help="Diretório base das imagens RVL-CDIP",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--retry-batch-sizes",
        default="8,4,2,1",
        help="Sequência de batch sizes para retry em caso de OOM",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-classes", type=int, default=16)
    parser.add_argument("--output-csv", default="results/RVL-CDIP_transfer_split1_eval.csv")
    parser.add_argument("--run-name", default="transfer_split1_eval")
    parser.add_argument(
        "--measure-speed",
        action="store_true",
        help="Se ativado, mede tempo médio por batch após a avaliação (usa inferência extra)",
    )
    args = parser.parse_args()

    losses = [item.strip() for item in args.losses.split(",") if item.strip()]
    if not losses:
        raise ValueError("Nenhuma loss informada em --losses")

    checkpoint_root = _resolve_checkpoint_root(args.checkpoint_root)
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"Raiz de checkpoints não encontrada: {checkpoint_root}")

    checkpoint_name_filter = args.checkpoint_name_filter.strip() if args.checkpoint_name_filter else None
    manual_checkpoint_map = {}
    if args.checkpoint_map_json:
        loaded = json.loads(args.checkpoint_map_json)
        if not isinstance(loaded, dict):
            raise ValueError("--checkpoint-map-json deve ser um JSON objeto (loss -> path)")
        manual_checkpoint_map = {str(k): str(v) for k, v in loaded.items()}

    pairs_csv = Path(args.pairs_csv)
    if not pairs_csv.is_absolute():
        pairs_csv = (WORKSPACE_ROOT / pairs_csv).resolve()
    if not pairs_csv.exists():
        raise FileNotFoundError(f"pairs-csv não encontrado: {pairs_csv}")

    retry_batch_sizes = [int(item.strip()) for item in args.retry_batch_sizes.split(",") if item.strip()]
    if not retry_batch_sizes:
        retry_batch_sizes = [args.batch_size]
    if args.batch_size not in retry_batch_sizes:
        retry_batch_sizes.insert(0, args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    print("[INFO] Carregando backbone InternVL3-2B...")

    backbone, processor, tokenizer, _, _ = load_model(
        model_name="InternVL3-2B",
        adapter_path=None,
        load_in_4bit=False,
        projection_output_dim=1536,
    )
    backbone.requires_grad_(False)

    image_context_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == image_context_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    warm_up_model(backbone, processor)

    dataset = DocumentPairDataset(
        csv_path=str(pairs_csv),
        base_dir=args.base_image_dir,
        input_size=448,
        max_num=12,
        device="cpu",
    )

    records = []
    for loss_name in losses:
        print("-" * 80)
        print(f"[INFO] Avaliando loss: {loss_name}")

        if loss_name in manual_checkpoint_map:
            ckpt_path = Path(manual_checkpoint_map[loss_name]).expanduser().resolve()
        elif args.checkpoint_path:
            ckpt_path = Path(args.checkpoint_path).expanduser().resolve()
        else:
            found = _find_latest_checkpoint_for_loss(
                checkpoint_root=checkpoint_root,
                loss_name=loss_name,
                name_filter=checkpoint_name_filter,
            )
            ckpt_path = found if found else None

        if ckpt_path is None or not ckpt_path.exists():
            print(f"[WARN] Checkpoint não encontrado para loss={loss_name}")
            records.append(
                {
                    "run_name": args.run_name,
                    "checkpoint_path": "",
                    "dataset": "RVL-CDIP",
                    "protocol": "zsl",
                    "split": 1,
                    "pairs_csv": str(pairs_csv),
                    "loss_type_eval": loss_name,
                    "num_pairs": len(dataset),
                    "batch_size": args.batch_size,
                    "num_workers": args.num_workers,
                    "val_loss": None,
                    "eer": None,
                    "eer_percent": None,
                    "threshold": None,
                    "recall_at_1": None,
                    "batch_recall_at_1": None,
                    "avg_batch_seconds": None,
                    "device": device,
                    "status": "missing_checkpoint",
                }
            )
            continue

        try:
            print(f"[INFO] Carregando checkpoint: {ckpt_path}")
            siam, ckpt_config = _load_siam_from_checkpoint(ckpt_path, backbone, tokenizer, device)
            criterion = _build_eval_criterion(loss_name, ckpt_config, args.num_classes, device)

            last_error = None
            success_record = None
            for batch_size in retry_batch_sizes:
                loader = _build_loader(dataset, batch_size=batch_size, num_workers=args.num_workers)
                try:
                    print(f"[INFO] Rodando validação zero-shot (split 1) | batch_size={batch_size}...")
                    mean_loss, eer, threshold, recall_at_1, batch_recall_at_1 = validate_siam_on_loader(
                        siam,
                        loader,
                        device,
                        criterion,
                    )

                    avg_batch_time = None
                    if args.measure_speed:
                        avg_batch_time = _measure_avg_batch_time(siam, loader, device)

                    success_record = {
                        "run_name": args.run_name,
                        "checkpoint_path": str(ckpt_path),
                        "dataset": "RVL-CDIP",
                        "protocol": "zsl",
                        "split": 1,
                        "pairs_csv": str(pairs_csv),
                        "loss_type_eval": loss_name,
                        "num_pairs": len(dataset),
                        "batch_size": batch_size,
                        "num_workers": args.num_workers,
                        "val_loss": float(mean_loss),
                        "eer": float(eer),
                        "eer_percent": float(eer * 100.0),
                        "threshold": float(threshold),
                        "recall_at_1": float(recall_at_1),
                        "batch_recall_at_1": float(batch_recall_at_1),
                        "avg_batch_seconds": avg_batch_time,
                        "device": device,
                        "status": "success",
                    }
                    print(f"[OK] {loss_name} | batch_size={batch_size} | EER={eer*100.0:.2f}% | R@1={recall_at_1:.4f}")
                    break
                except RuntimeError as error:
                    last_error = error
                    if "out of memory" not in str(error).lower():
                        raise
                    print(f"[WARN] OOM com batch_size={batch_size}. Tentando menor batch...")
                    _clear_cuda_memory()
                    continue

            if success_record is not None:
                records.append(success_record)
            else:
                raise last_error if last_error is not None else RuntimeError("Falha desconhecida na avaliação")
        except Exception as error:
            print(f"[ERROR] Falha na loss {loss_name}: {error}")
            records.append(
                {
                    "run_name": args.run_name,
                    "checkpoint_path": str(ckpt_path),
                    "dataset": "RVL-CDIP",
                    "protocol": "zsl",
                    "split": 1,
                    "pairs_csv": str(pairs_csv),
                    "loss_type_eval": loss_name,
                    "num_pairs": len(dataset),
                    "batch_size": args.batch_size,
                    "num_workers": args.num_workers,
                    "val_loss": None,
                    "eer": None,
                    "eer_percent": None,
                    "threshold": None,
                    "recall_at_1": None,
                    "batch_recall_at_1": None,
                    "avg_batch_seconds": None,
                    "device": device,
                    "status": f"error: {error}",
                }
            )
        finally:
            try:
                del criterion
            except Exception:
                pass
            try:
                del siam
            except Exception:
                pass
            _clear_cuda_memory()

    output_csv = Path(args.output_csv)
    if not output_csv.is_absolute():
        output_csv = (WORKSPACE_ROOT / output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame(records)
    if output_csv.exists():
        existing = pd.read_csv(output_csv)
        updated = pd.concat([existing, new_df], ignore_index=True)
    else:
        updated = new_df

    updated.to_csv(output_csv, index=False)

    print("=" * 80)
    print("Teste de transfer learning (RVL-CDIP ZSL split 1) concluído")
    print("=" * 80)
    if not new_df.empty:
        ok = new_df[new_df["status"] == "success"]
        if not ok.empty:
            best = ok.nsmallest(1, "eer_percent").iloc[0]
            print(
                f"Melhor loss: {best['loss_type_eval']} | EER={best['eer_percent']:.2f}% | "
                f"R@1={best['recall_at_1']:.4f}"
            )
        print(f"Runs avaliados nesta execução: {len(new_df)}")
    print(f"Resultado salvo em: {output_csv}")


if __name__ == "__main__":
    main()
