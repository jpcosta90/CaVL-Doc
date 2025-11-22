#!/usr/bin/env python3
"""
Evaluate a trained SiameseInternVL.
Compatible with both "Smart Checkpoints" (new trainer) and Legacy checkpoints.
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
import warnings

# Supressão de warnings
warnings.filterwarnings("ignore", message=".*use_reentrant parameter.*")
warnings.filterwarnings("ignore", message=".*None of the inputs have requires_grad.*")

import torch
import pandas as pd

# Project imports
from cavl_doc.models.backbone_loader import load_model, warm_up_model
from cavl_doc.data.dataset import DocumentPairDataset
from cavl_doc.utils.visualization import plot_density
from cavl_doc.evaluation.metrics import compute_eer
from cavl_doc.utils.checkpointing import load_trained_siamese

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\nLoading backbone model...")
    backbone, processor, tokenizer, _, _ = load_model(
        model_name=args.model_name,
        load_in_4bit=args.load_in_4bit,
        projection_output_dim=512 # dummy
    )
    backbone.requires_grad_(False)
    backbone.eval()
    warm_up_model(backbone, processor)

    siam = load_trained_siamese(args.checkpoint_path, backbone, tokenizer, device, default_proj_out=args.proj_out)

    dataset = DocumentPairDataset(
        csv_path=args.pairs_csv,
        base_dir=args.base_image_dir,
        input_size=args.input_size,
        max_num=args.max_num_image_tokens,
        device="cpu"
    )
    print(f"Dataset: {len(dataset)} pairs.")

    if args.dataset_name:
        ds_name = args.dataset_name
    else:
        ds_name = Path(args.pairs_csv).parent.name

    ckpt_name = Path(args.checkpoint_path).stem
    results = []

    print("Starting evaluation loop...")
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc=f"Eval {ds_name}"):
            item = dataset[idx]
            # Dataset retorna [Patches, 3, H, W].
            # Precisamos de [1, Patches, 3, H, W] para passar para o encode_fn 
            # que vai detectar o 5D e tratar.
            img_a = item["image_a"].unsqueeze(0).to(device)
            img_b = item["image_b"].unsqueeze(0).to(device)
            label = float(item["label"])

            try:
                # Agora chamamos o modelo de forma que ele use a encode_fn corrigida acima
                # Modo 1: Se o modelo aceita images=...
                z_a = siam(images=img_a)
                z_b = siam(images=img_b)
                
            except Exception as e:
                # Modo 2: Fallback manual chamando a encode_fn diretamente
                # Caso o forward padrão não esteja chamando a encode_fn corretamente
                try:
                    tokens_a, mask_a = siam._extract_tokens_via_encode_fn(img_a, device=device)
                    tokens_b, mask_b = siam._extract_tokens_via_encode_fn(img_b, device=device)
                    
                    z_a = siam.head(siam.pool(tokens_a, mask_a))
                    z_b = siam.head(siam.pool(tokens_b, mask_b))
                except Exception as inner_e:
                    print(f"\n[ERROR] Batch {idx} falhou.")
                    print(f"Input shape: {img_a.shape}")
                    print(f"Erro original: {e}")
                    print(f"Erro fallback: {inner_e}")
                    continue

            if args.metric == "cosine":
                score = torch.nn.functional.cosine_similarity(z_a, z_b).item()
            else:
                score = torch.norm(z_a - z_b, p=2, dim=1).item()

            results.append({"idx": idx, "is_equal": label, "metric_score": score})

    if not results:
        print("Nenhum resultado gerado. Verifique os erros acima.")
        return

    df = pd.DataFrame(results)
    scores = -df["metric_score"].values if args.metric == "euclidean" else df["metric_score"].values
    eer, thr = compute_eer(df["is_equal"].values, scores)

    print(f"\n--- {ds_name} Results ---")
    print(f"EER: {eer*100:.3f}% | Thr: {thr:.4f}")

    if not args.output_csv:
        args.output_csv = f"results/{ds_name}_{ckpt_name}_{args.metric}_eval.csv"
    
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved results to: {args.output_csv}")
    
    if args.plot:
        plot_density(df, eer, thr, f"{ckpt_name}_{args.metric}", ds_name, args.metric)
        print("Saved density plot.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-csv", required=True)
    parser.add_argument("--base-image-dir", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--model-name", default="InternVL3-2B")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--metric", default="euclidean", choices=["cosine", "euclidean"])
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--max-num-image-tokens", type=int, default=12)
    parser.add_argument("--proj-out", type=int, default=512)
    parser.add_argument("--output-csv", type=str)
    parser.add_argument("--plot", action="store_true")
    
    main(parser.parse_args())