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

from cavl_doc.data.dataset import DocumentPairDataset
from cavl_doc.models.backbone_loader import load_model, warm_up_model
from cavl_doc.utils.helpers import setup_experiment_dir
from cavl_doc.modules.heads import build_head
from cavl_doc.models.policy import ProfessorNetwork
from cavl_doc.trainers.rl_trainer import run_rl_siamese_loop
from cavl_doc.trainers.curriculum_trainer import CurriculumTrainer
from cavl_doc.models.modeling_cavl import build_cavl_model
from cavl_doc.utils.embedding_utils import prepare_inputs_for_multimodal_embedding
from torch.utils.data import DataLoader, Subset

EMBEDDING_PROMPT = "<image> Analyze this document"

def rl_full_collate_fn(batch):
    img_a_list = [item['image_a'] for item in batch]
    img_b_list = [item['image_b'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float32)
    
    if 'class_a' in batch[0]:
        class_a = torch.tensor([item['class_a'] for item in batch], dtype=torch.long)
        class_b = torch.tensor([item['class_b'] for item in batch], dtype=torch.long)
    else:
        class_a = torch.zeros(len(batch), dtype=torch.long)
        class_b = torch.zeros(len(batch), dtype=torch.long)
        
    return img_a_list, img_b_list, labels, class_a, class_b

def prepare_experiment(args):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if not args.wandb_run_name:
        args.wandb_run_name = f"{args.dataset_name}_{args.model_name}_{args.loss_type}_{timestamp}"
    
    experiment_name = args.wandb_run_name
    base_path = "/mnt/large/checkpoints" if os.path.exists("/mnt/large") else "checkpoints"
    
    wandb_id = None
    
    # Se estiver retomando, usa o diretório do checkpoint
    if args.resume_from and os.path.exists(args.resume_from):
        # Assume que resume_from é .../run_name/last_checkpoint.pt
        outdir = os.path.dirname(args.resume_from)
        print(f"📂 Usando diretório existente para resume: {outdir}")
        
        # Tenta recuperar o ID do WandB do config
        cfg_path = os.path.join(outdir, "training_config.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, 'r') as f:
                    old_cfg = json.load(f)
                    wandb_id = old_cfg.get('wandb_id')
            except Exception as e:
                print(f"⚠️ Erro ao ler config anterior: {e}")
    else:
        outdir = setup_experiment_dir(base_path, experiment_name)
    
    cfg = vars(args)
    cfg['timestamp'] = timestamp
    cfg['outdir'] = str(outdir)
    
    with open(os.path.join(outdir, "training_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    
    return outdir, wandb_id

def main(args):
    outdir, resume_wandb_id = prepare_experiment(args)
    
    # Prioridade: Argumento CLI > Config do Checkpoint
    final_wandb_id = args.wandb_id if args.wandb_id else resume_wandb_id

    if args.use_wandb:
        import wandb
        
        # Check if group is provided via environment variable (from sweep script)
        default_group = os.environ.get("WANDB_RUN_GROUP", "cavl-rl-training")
        
        init_kwargs = {
            "project": args.wandb_project,
            "name": args.wandb_run_name,
            "config": vars(args),
            "group": default_group,
            "settings": wandb.Settings(init_timeout=300)
        }
        
        if final_wandb_id:
            print(f"🔄 Resuming WandB Run ID: {final_wandb_id}")
            init_kwargs['id'] = final_wandb_id
            init_kwargs['resume'] = 'allow'
            
        try:
            wandb.init(**init_kwargs)
        except Exception as e:
            print(f"❌ Erro crítico ao inicializar WandB (Resume ID: {final_wandb_id}): {e}")
            if final_wandb_id:
                print("⚠️ Falha ao retomar. Iniciando um NOVO run para garantir a continuidade do treino...")
                # Remove ID e Resume para forçar um run novo
                init_kwargs.pop('id', None)
                init_kwargs.pop('resume', None)
                # Tenta novamente como um run limpo
                wandb.init(**init_kwargs)
            else:
                # Se não era resume e falhou, então é erro de conexão/api mesmo
                raise e

        print(f"🚀 WandB Inicializado: Run {wandb.run.name} (ID: {wandb.run.id})")
        
        if wandb.run:
            # Salva o ID no args para persistência futura
            args.wandb_id = wandb.run.id
            
            for k, v in wandb.config.items():
                if hasattr(args, k):
                    setattr(args, k, v)
        
        # Atualiza o arquivo de configuração com o ID do WandB
        # Isso garante que futuros resumes possam encontrar o ID correto
        cfg = vars(args)
        cfg['outdir'] = str(outdir)
        with open(os.path.join(outdir, "training_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- 1) Carregar Backbone ---
    print(f"Loading backbone model '{args.model_name}'...")
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
    
    print(f"Construindo Head do tipo: '{args.head_type}'")
    student_head = build_head(
        head_type=args.head_type,
        input_dim=LLM_HIDDEN_DIM, 
        proj_hidden=4096,
        proj_out=args.projection_output_dim,
        dropout=0.1 # Opcional: expor no futuro
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
    
    # Detecção de Classes
    detected_classes = getattr(dataset, 'num_classes', 0)
    if detected_classes > 0:
        num_classes = detected_classes
        print(f"Classes detectadas no dataset: {num_classes}")
    else:
        num_classes = args.num_classes
        print(f"Usando num_classes do argumento: {num_classes}")

    if args.training_sample_size > 0 and args.training_sample_size < len(dataset):
        indices = random.sample(range(len(dataset)), args.training_sample_size)
        dataset = torch.utils.data.Subset(dataset, indices)

    # --- 4) Validação ---
    val_csv = args.pairs_csv.replace("train_pairs.csv", "validation_pairs.csv")
    if not os.path.exists(val_csv):
        val_csv = None
        print("Aviso: validation_pairs.csv não encontrado. Usando split automático.")

    # --- 5) Iniciar Treinamento ---
    if args.use_curriculum:
        # Lógica de Fallback para Curriculum
        if args.phase1_loss is None:
            if args.loss_type != "contrastive":
                print(f"Info: Usando --loss-type='{args.loss_type}' para Phase 1.")
                args.phase1_loss = args.loss_type
            else:
                args.phase1_loss = "expface"

        if args.phase2_loss is None:
            if args.loss_type != "contrastive":
                print(f"Info: Usando --loss-type='{args.loss_type}' para Phase 2.")
                args.phase2_loss = args.loss_type
            else:
                args.phase2_loss = "elastic_expface"

        phases_str = f"{args.phase1_loss} -> {args.phase2_loss}"
        if args.phase3_loss:
            phases_str += f" -> {args.phase3_loss}"
        print(f"🚀 Starting Curriculum Training ({phases_str})")
        
        # 5.1 Prepare DataLoaders
        if val_csv and os.path.exists(val_csv):
            print(f"Carregando validação: {val_csv}")
            val_dataset = DocumentPairDataset(val_csv, args.base_image_dir, args.input_size, args.max_num_image_tokens, 'cpu')
            train_dataset = dataset
        else:
            print("Split automático treino/val.")
            if isinstance(dataset, Subset): full_ds, ds_indices = dataset.dataset, list(dataset.indices)
            else: full_ds, ds_indices = dataset, list(range(len(dataset)))
            val_size = min(max(args.val_min_size, int(len(ds_indices) * args.val_fraction)), len(ds_indices)//10)
            if val_size <= 0: val_size = min(args.val_min_size, len(ds_indices)//10)
            val_indices = ds_indices[-val_size:]; train_indices = ds_indices[:-val_size]
            if not train_indices:
                random.shuffle(ds_indices); val_indices = ds_indices[:val_size]; train_indices = ds_indices[val_size:]
            train_dataset = Subset(full_ds, train_indices)
            val_dataset = Subset(full_ds, val_indices)

        train_loader = DataLoader(train_dataset, batch_size=args.candidate_pool_size, shuffle=True, num_workers=0, collate_fn=rl_full_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=96, shuffle=False, num_workers=0, collate_fn=rl_full_collate_fn)

        # 5.2 Build Full Model
        def _encode_fn(backbone, images, cut_layer=args.cut_layer, **kwargs):
            # images: Tensor [B, N, C, H, W] ou List[Tensor [Ni, C, H, W]]
            
            input_ids_list = []
            pixel_values_list = []
            image_flags_list = []
            
            # Normaliza entrada para lista
            if isinstance(images, torch.Tensor):
                if images.dim() == 5:
                    images_list = [images[i] for i in range(images.shape[0])]
                else:
                    images_list = [images]
            else:
                images_list = images

            # Prepara inputs individualmente (para lidar com N variável e gerar input_ids corretos)
            for img in images_list:
                # img: [N, C, H, W]
                out = prepare_inputs_for_multimodal_embedding(backbone, tokenizer, img, EMBEDDING_PROMPT)
                input_ids_list.append(out['input_ids'][0]) # [L]
                pixel_values_list.append(out['pixel_values']) # [N, C, H, W]
                image_flags_list.append(out['image_flags']) # [N]

            # Padding manual dos input_ids
            max_len = max(len(ids) for ids in input_ids_list)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            
            padded_input_ids = []
            padded_attention_mask = []
            
            for ids in input_ids_list:
                pad_len = max_len - len(ids)
                # Right padding
                p_ids = torch.cat([ids, torch.full((pad_len,), pad_id, device=ids.device, dtype=ids.dtype)])
                p_mask = torch.cat([torch.ones_like(ids), torch.zeros((pad_len,), device=ids.device, dtype=ids.dtype)])
                padded_input_ids.append(p_ids)
                padded_attention_mask.append(p_mask)

            # Batch final
            batch_input_ids = torch.stack(padded_input_ids).to(device)
            batch_attention_mask = torch.stack(padded_attention_mask).to(device)
            batch_pixel_values = torch.cat(pixel_values_list, dim=0).to(device, dtype=torch.bfloat16)
            batch_image_flags = torch.cat(image_flags_list, dim=0).to(device)

            out = backbone(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                pixel_values=batch_pixel_values,
                image_flags=batch_image_flags,
                output_hidden_states=True,
                return_dict=True
            )
            hidden_states = out.hidden_states
            lm = backbone.language_model.model
            idx = cut_layer + 1 if len(hidden_states) == (len(lm.layers) + 1) else cut_layer
            return hidden_states[idx], None

        siam = build_cavl_model(
            backbone=backbone, 
            cut_layer=args.cut_layer, 
            encode_fn=_encode_fn,
            pool_dim=1536, # InternVL hidden dim
            proj_hidden=4096,
            proj_out=args.projection_output_dim,
            set_trainable=True,
            tokenizer=tokenizer,
            pooler_type=args.pooler_type,
            head_type=args.head_type,
            num_queries=args.num_queries
        ).to(device)
        siam.set_default_trainable()

        # 5.3 Run Curriculum
        config = vars(args)
        config['lr'] = args.student_lr
        config['professor_lr'] = args.professor_lr
        config['weight_decay'] = 1e-4
        config['output_dir'] = str(outdir)
        config['num_classes'] = num_classes
        config['entropy_coeff'] = args.entropy_coeff
        config['baseline_alpha'] = args.baseline_alpha
        
        wandb_run = wandb if args.use_wandb else None
        
        trainer = CurriculumTrainer(
            model=siam,
            professor_model=professor_model,
            tokenizer=tokenizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=config,
            wandb_run=wandb_run
        )
        trainer.run(epochs=args.epochs)

    else:
        print(f"Starting run_rl_siamese_loop (Loss: {args.loss_type})...")
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
            # Args Modulares
            loss_type=args.loss_type,
            pooler_type=args.pooler_type,
            head_type=args.head_type,
            num_queries=args.num_queries,
            num_classes=num_classes,
            val_samples_per_class=args.val_samples_per_class,
            # Hiperparâmetros de Loss
            margin=args.margin,
            scale=args.scale,
            num_sub_centers=args.num_sub_centers,
            std=args.std,
            # Resume
            resume_checkpoint_path=args.resume_from,
            # Optimizer/Scheduler
            optimizer_type=args.optimizer_type,
            scheduler_type=args.scheduler_type,
            # Debug/Sweep Control
            max_steps_per_epoch=args.max_steps_per_epoch,
            professor_warmup_steps=args.professor_warmup_steps,
            easy_mining_steps=args.easy_mining_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            weight_decay=args.weight_decay
        )
    
    if args.use_wandb:
        import wandb
        wandb.finish()

def parse_args():
    p = argparse.ArgumentParser(description="Script to run CaVL (Siamese) RL training.")
    
    # WandB
    p.add_argument("--use-wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="CaVL-Doc")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--wandb-id", type=str, default=None, help="Explicit WandB Run ID for resuming")

    # Modular Arguments
    p.add_argument("--loss-type", type=str, default="contrastive", 
                   choices=["contrastive", "triplet", "arcface", "elastic_arcface", "cosface", "elastic_cosface", "expface", "elastic_expface", "subcenter_arcface", "subcenter_cosface", "circle", "elastic_circle", "subcenter_circle", "angular", "iso_arcface", "iso_cosface", "iso_circle"],
                   help="Type of loss function")
    p.add_argument("--pooler-type", type=str, default="attention", choices=["attention", "mean", "gem", "netvlad"])
    p.add_argument("--head-type", type=str, default="mlp", choices=["mlp", "simple_mlp", "residual"])
    p.add_argument("--num-queries", type=int, default=1, help="Number of attention queries for pooling")
    p.add_argument("--num-classes", type=int, default=16, help="Number of classes (fallback if not detected)")

    # Hiperparâmetros de Loss
    p.add_argument("--margin", type=float, default=0.5, help="Margin (m) for ArcFace/CosFace/ExpFace")
    p.add_argument("--scale", type=float, default=64.0, help="Scale (s/gamma) for ArcFace/Circle")
    p.add_argument("--num-sub-centers", type=int, default=3, help="Number of sub-centers (k) for SubCenter losses")
    p.add_argument("--std", type=float, default=0.05, help="Standard Deviation (std) for ElasticArcFace")

    # Curriculum Learning
    p.add_argument("--use-curriculum", action="store_true", help="Enable Two-Phase Curriculum Learning")
    p.add_argument("--phase1-loss", type=str, default=None, help="Loss for Phase 1 (GGA). Defaults to 'expface' or --loss-type.")
    p.add_argument("--phase2-loss", type=str, default=None, help="Loss for Phase 2 (EHR). Defaults to 'elastic_expface' or --loss-type.")
    p.add_argument("--phase3-loss", type=str, default=None, help="Optional Loss for Phase 3")

    # Dataset & Model
    p.add_argument("--dataset-name", type=str, default="LA-CDIP")
    p.add_argument("--pairs-csv", type=str, required=True)
    p.add_argument("--base-image-dir", type=str, required=True)
    p.add_argument("--model-name", type=str, default="InternVL3-2B")
    
    p.add_argument("--projection-output-dim", type=int, default=512)
    p.add_argument("--max-num-image-tokens", dest="max_num_image_tokens", type=int, default=12)
    p.add_argument("--input-size", type=int, default=448)

    # Training config
    p.add_argument("--training-sample-size", dest="training_sample_size", type=int, default=0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--max-steps-per-epoch", type=int, default=None, help="Limita o número de steps (batches) por época para debugging/sweeps")
    p.add_argument("--professor-warmup-steps", type=int, default=0, help="Steps iniciais com amostragem aleatória (sem Professor ativo)")
    p.add_argument("--easy-mining-steps", type=int, default=0, help="Steps iniciais com recompensa invertida (Easy Mining)")
    p.add_argument("--load-in-4bit", action="store_true", default=False)

    p.add_argument("--student-lr", type=float, default=1e-4)
    p.add_argument("--professor-lr", type=float, default=1e-4)
    p.add_argument("--candidate-pool-size", type=int, default=8)
    p.add_argument("--student-batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Steps de acumulação de gradiente")
    
    p.add_argument("--cut-layer", type=int, default=27)
    
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--val-min-size", type=int, default=200)
    p.add_argument("--val-samples-per-class", type=int, default=20, help="Number of samples per class for balanced validation subset")
    p.add_argument("--patience", type=int, default=3)
    
    p.add_argument("--baseline-alpha", type=float, default=0.01)
    p.add_argument("--entropy-coeff", type=float, default=0.01)
    p.add_argument("--lr-reduce-factor", type=float, default=0.5)
    
    p.add_argument("--seed", type=int, default=42)
    
    # Resume
    p.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to resume from")
    
    # Optimizer & Scheduler
    p.add_argument("--optimizer-type", type=str, default="adam", choices=["adam", "adamw", "sgd"], help="Optimizer type")
    p.add_argument("--scheduler-type", type=str, default=None, choices=["step", "cosine", "plateau", "constant"], help="Scheduler type")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for optimizer")

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)