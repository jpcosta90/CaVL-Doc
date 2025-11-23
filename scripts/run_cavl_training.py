#!/usr/bin/env python3
# scripts/run_cavl_training.py

import os
import warnings
import argparse
import json
import time
import random
import torch

# --- SILENCIADOR DE WARNINGS ---
def _custom_warn(message, category=None, stacklevel=1, source=None):
    msg_str = str(message)
    block_list = [
        "use_reentrant parameter should be passed explicitly",
        "None of the inputs have requires_grad=True",
        "torch.utils.checkpoint"
    ]
    if any(s in msg_str for s in block_list):
        return
    _original_warn(message, category, stacklevel, source)

if not hasattr(warnings, '_original_warn'):
    _original_warn = warnings.warn
    warnings.warn = _custom_warn
# ----------------------------------

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Imports do Pacote cavl_doc
from cavl_doc.data.dataset import DocumentPairDataset
from cavl_doc.models.backbone_loader import load_model, warm_up_model
from cavl_doc.utils.helpers import setup_experiment_dir
from cavl_doc.models.policy import ProfessorNetwork
from cavl_doc.trainers.rl_trainer import run_rl_siamese_loop

# Importa o builder de heads (para usar o default ou custom)
from cavl_doc.modules.heads import build_head

def prepare_experiment(args):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if not args.wandb_run_name:
        args.wandb_run_name = f"{args.dataset_name}_{args.model_name}_rl_{timestamp}"
    
    experiment_name = args.wandb_run_name
    base_path = "/mnt/large/checkpoints" if os.path.exists("/mnt/large") else "checkpoints"
    outdir = setup_experiment_dir(base_path, experiment_name)
    
    cfg = vars(args)
    cfg['timestamp'] = timestamp
    cfg['outdir'] = str(outdir)
    
    with open(os.path.join(outdir, "training_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    
    return outdir

def main(args):
    outdir = prepare_experiment(args)
    
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            group="cavl-rl-training"
        )
        print(f"🚀 WandB Inicializado: Run {args.wandb_run_name}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- 1) Carregar Backbone ---
    print(f"Loading backbone model '{args.model_name}' (4-bit={args.load_in_4bit}) ...")
    backbone, processor, tokenizer, _, _ = load_model(
        model_name=args.model_name,
        adapter_path=None,
        load_in_4bit=args.load_in_4bit,
        projection_output_dim=args.projection_output_dim
    )
    backbone.requires_grad_(False)
    warm_up_model(backbone, processor)

    # --- 2) Instanciar Head e Professor ---
    LLM_HIDDEN_DIM = 1536
    
    # Agora usamos o BUILDER para criar o head, permitindo troca fácil via args
    print(f"Construindo Head do tipo: '{args.head_type}'")
    student_head = build_head(
        head_type=args.head_type,
        input_dim=LLM_HIDDEN_DIM, 
        proj_hidden=4096,
        proj_out=args.projection_output_dim
    ).to(device)
    student_head.train()

    professor_model = ProfessorNetwork(input_dim=1).to(device)
    professor_model.train()

    # --- 3) Dataset de Treino ---
    print("Loading training dataset...")
    dataset = DocumentPairDataset(
        csv_path=args.pairs_csv, 
        base_dir=args.base_image_dir,
        input_size=args.input_size, 
        max_num=args.max_num_image_tokens, 
        device='cpu'
    )
    
    if args.training_sample_size > 0 and args.training_sample_size < len(dataset):
        indices = random.sample(range(len(dataset)), args.training_sample_size)
        dataset = torch.utils.data.Subset(dataset, indices)

    # --- 4) Validação ---
    val_csv = args.pairs_csv.replace("train_pairs.csv", "validation_pairs.csv")
    if not os.path.exists(val_csv):
        val_csv = None
        print("Aviso: validation_pairs.csv não encontrado. Usando split automático.")
    else:
        print(f"Usando validação externa: {val_csv}")

    # --- 5) Iniciar Treinamento ---
    print("Starting run_rl_siamese_loop trainer ...")
    run_rl_siamese_loop(
        base_model=backbone,
        student_head=student_head,
        professor_model=professor_model,
        tokenizer=tokenizer,
        dataset=dataset,
        epochs=args.epochs,
        student_lr=args.student_lr,
        professor_lr=args.professor_lr,
        device=device,
        output_dir=str(outdir),
        candidate_pool_size=args.candidate_pool_size,
        student_batch_size=args.student_batch_size,
        max_num_image_tokens=args.max_num_image_tokens,
        cut_layer=args.cut_layer,
        projection_output_dim=args.projection_output_dim,
        val_csv_path=val_csv,
        base_image_dir=args.base_image_dir,
        val_fraction=args.val_fraction,
        val_min_size=args.val_min_size,
        patience=args.patience,
        lr_reduce_factor=args.lr_reduce_factor,
        baseline_alpha=args.baseline_alpha,
        entropy_coeff=args.entropy_coeff,
        seed=args.seed,
        use_wandb=args.use_wandb,
        # Novos argumentos modulares
        loss_type=args.loss_type,
        pooler_type=args.pooler_type,
        head_type=args.head_type
    )
    
    if args.use_wandb:
        import wandb
        wandb.finish()

def parse_args():
    p = argparse.ArgumentParser(description="Script to run CaVL (Siamese) RL training.")
    
    # WandB Arguments
    p.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb-project", type=str, default="CaVL-Doc", help="WandB Project Name")
    p.add_argument("--wandb-run-name", type=str, default=None, help="WandB Run Name (Optional)")

    # Modular Arguments (Novos)
    p.add_argument("--loss-type", type=str, default="contrastive", choices=["contrastive"], help="Type of loss function")
    p.add_argument("--pooler-type", type=str, default="attention", choices=["attention", "mean"], help="Type of pooling layer")
    p.add_argument("--head-type", type=str, default="mlp", choices=["mlp", "simple_mlp"], help="Type of projection head")

    p.add_argument("--dataset-name", type=str, default="LA-CDIP")
    p.add_argument("--pairs-csv", type=str, required=True)
    p.add_argument("--base-image-dir", type=str, required=True)
    p.add_argument("--model-name", type=str, default="InternVL3-2B")
    
    p.add_argument("--projection-output-dim", type=int, default=512)
    p.add_argument("--max-num-image-tokens", dest="max_num_image_tokens", type=int, default=12)
    p.add_argument("--input-size", type=int, default=448)

    p.add_argument("--training-sample-size", dest="training_sample_size", type=int, default=0)
    p.add_argument("--epochs", type=int, default=5)
    
    p.add_argument("--load-in-4bit", action="store_true", default=False)

    p.add_argument("--student-lr", type=float, default=1e-4)
    p.add_argument("--professor-lr", type=float, default=1e-4)
    p.add_argument("--candidate-pool-size", type=int, default=8)
    p.add_argument("--student-batch-size", type=int, default=4)
    
    p.add_argument("--cut-layer", type=int, default=27)
    
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--val-min-size", type=int, default=200)
    p.add_argument("--patience", type=int, default=3)
    
    p.add_argument("--baseline-alpha", type=float, default=0.01)
    p.add_argument("--entropy-coeff", type=float, default=0.01)
    p.add_argument("--lr-reduce-factor", type=float, default=0.5)
    
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)