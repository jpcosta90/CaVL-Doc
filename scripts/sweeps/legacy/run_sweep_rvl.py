#!/usr/bin/env python3
import subprocess
import sys
import os
import time
import argparse

# ==============================================================================
# ⚙️ GLOBAL CONFIG
# ==============================================================================

WORKSPACE_ROOT = "/home/joaopaulo/Projects/CaVL-Doc"
SCRIPT_PATH = os.path.join(WORKSPACE_ROOT, "scripts/training/run_cavl_training.py")
PREP_SCRIPT_PATH = os.path.join(WORKSPACE_ROOT, "scripts/utils/prepare_splits.py")
RAW_DATA_ROOT = "/mnt/data/zs_rvl_cdip" 
RAW_IMAGES_DIR = os.path.join(RAW_DATA_ROOT, "data") 
BASE_IMAGE_DIR = RAW_IMAGES_DIR

# Common Settings for Protocol V3
PROTOCOL_DEFAULTS = {
    "wandb_project": "CaVL-Calibration-Protocol",
    "epochs": "5",
    "max_steps": "1000",
    "batch_size": "4",
    "grad_accum": "4", # Effective 32
    "pool_size": "8",
    "prof_lr": "1e-4",
    "cut_layer": "27",
    "proj_dim": "1536",
    "pairs": 15
}

# ==============================================================================
# 🧪 EXPERIMENT REGISTRY (SWEEP DEFINITIONS)
# ==============================================================================

EXP_REGISTRY = {
    "arcface": {
        "lr": [
            {"scale": 64.0, "margin": 0.5, "lr": 0.1, "optimizer": "sgd", "warmup": 100},
            {"scale": 64.0, "margin": 0.5, "lr": 1e-2, "optimizer": "sgd", "warmup": 100},
            {"scale": 64.0, "margin": 0.5, "lr": 1e-3, "optimizer": "sgd", "warmup": 100},
            {"scale": 64.0, "margin": 0.5, "lr": 1e-3, "optimizer": "adamw", "warmup": 100},
            {"scale": 64.0, "margin": 0.5, "lr": 5e-4, "optimizer": "adamw", "warmup": 100},
            {"scale": 64.0, "margin": 0.5, "lr": 3e-4, "optimizer": "adamw", "warmup": 100},
            {"scale": 64.0, "margin": 0.5, "lr": 1e-4, "optimizer": "adamw", "warmup": 100},
            {"scale": 64.0, "margin": 0.5, "lr": 5e-5, "optimizer": "adamw", "warmup": 100},
        ],
        "scale": [
            {"scale": 64.0, "margin": 0.5, "lr": 0.1, "optimizer": "sgd", "warmup": 100},
            {"scale": 32.0, "margin": 0.5, "lr": 0.1, "optimizer": "sgd", "warmup": 100},
            {"scale": 24.0, "margin": 0.5, "lr": 0.1, "optimizer": "sgd", "warmup": 100},
            {"scale": 16.0, "margin": 0.5, "lr": 0.1, "optimizer": "sgd", "warmup": 100},
        ],
        "margin": [
            {"scale": 24.0, "margin": 0.3, "lr": 0.1, "optimizer": "sgd", "warmup": 100},
            {"scale": 24.0, "margin": 0.5, "lr": 0.1, "optimizer": "sgd", "warmup": 100},
            {"scale": 24.0, "margin": 0.7, "lr": 0.1, "optimizer": "sgd", "warmup": 100},
        ],
    },
    "cosface": {
        "lr": [
            {"scale": 64.0, "margin": 0.35, "lr": 0.2, "optimizer": "sgd", "warmup": 100},
            {"scale": 64.0, "margin": 0.35, "lr": 0.1, "optimizer": "sgd", "warmup": 100},
            {"scale": 64.0, "margin": 0.35, "lr": 0.05, "optimizer": "sgd", "warmup": 100},
            {"scale": 64.0, "margin": 0.35, "lr": 0.01, "optimizer": "sgd", "warmup": 100},
            {"scale": 64.0, "margin": 0.35, "lr": 1e-3, "optimizer": "adamw", "warmup": 100},
            {"scale": 64.0, "margin": 0.35, "lr": 5e-4, "optimizer": "adamw", "warmup": 100},
            {"scale": 64.0, "margin": 0.35, "lr": 1e-4, "optimizer": "adamw", "warmup": 100},
            
        ],
        "scale": [
            # Placeholder values for CosFace
            {"scale": 64.0, "margin": 0.35, "lr": 1e-4, "optimizer": "adamw", "warmup": 100},
            {"scale": 32.0, "margin": 0.35, "lr": 1e-4, "optimizer": "adamw", "warmup": 100},
            {"scale": 24.0, "margin": 0.35, "lr": 1e-4, "optimizer": "adamw", "warmup": 100},
            {"scale": 16.0, "margin": 0.35, "lr": 1e-4, "optimizer": "adamw", "warmup": 100},
        ],
        "margin": [
            {"scale": 24.0, "margin": 0.2, "lr": 1e-4, "optimizer": "adamw", "warmup": 100},
            {"scale": 24.0, "margin": 0.4, "lr": 1e-4, "optimizer": "adamw", "warmup": 100},
            {"scale": 24.0, "margin": 0.6, "lr": 1e-4, "optimizer": "adamw", "warmup": 100},
        ],
    },
"circle": {
        "lr": [
            {"scale": 256.0, "margin": 0.25, "lr": 1e-3, "optimizer": "adamw", "warmup": 100},
            {"scale": 256.0, "margin": 0.25, "lr": 5e-4, "optimizer": "adamw", "warmup": 100},
            {"scale": 256.0, "margin": 0.25, "lr": 1e-4, "optimizer": "adamw", "warmup": 100},
            {"scale": 256.0, "margin": 0.25, "lr": 5e-5, "optimizer": "adamw", "warmup": 100},
            
            {"scale": 256.0, "margin": 0.25, "lr": 0.1, "optimizer": "sgd", "warmup": 100},
            {"scale": 256.0, "margin": 0.25, "lr": 0.01, "optimizer": "sgd", "warmup": 100},
            {"scale": 256.0, "margin": 0.25, "lr": 0.001, "optimizer": "sgd", "warmup": 100}
        ],
        "scale": [

            {"scale": 512.0, "margin": 0.25, "lr": 5e-4, "optimizer": "adamw", "warmup": 100},
            {"scale": 256.0, "margin": 0.25, "lr": 5e-4, "optimizer": "adamw", "warmup": 100},
            {"scale": 128.0, "margin": 0.25, "lr": 5e-4, "optimizer": "adamw", "warmup": 100},
            {"scale": 64.0, "margin": 0.25, "lr": 5e-4, "optimizer": "adamw", "warmup": 100}
        ],
        "margin": [
            {"scale": 256.0, "margin": 0.15, "lr": 1e-4, "optimizer": "adamw", "warmup": 100},
            {"scale": 256.0, "margin": 0.25, "lr": 1e-4, "optimizer": "adamw", "warmup": 100},
            {"scale": 256.0, "margin": 0.35, "lr": 1e-4, "optimizer": "adamw", "warmup": 100}
        ]
    },
    "subcenter_arcface": {
        # Vamos chamar essa bateria de "k_lr_sweep" pois varia ambos
        "k": [
            # --- GRUPO K=1 (BASELINE - O Vencedor Atual) ---
            # Serve para confirmar que o código Subcenter com k=1 bate com o ArcFace normal
            {"scale": 24.0, "margin": 0.5, "lr": 0.1, "optimizer": "sgd", "warmup": 100, "k": 1},

            # --- GRUPO K=2 (Conservador: Frente/Verso) ---
            # Hipótese: Com 2 centros, a LR 0.1 pode ser agressiva demais.
            # Testamos do agressivo (0.1) até o seguro (0.005)
            {"scale": 24.0, "margin": 0.5, "lr": 0.1,   "optimizer": "sgd", "warmup": 100, "k": 2},
            {"scale": 24.0, "margin": 0.5, "lr": 0.05,  "optimizer": "sgd", "warmup": 100, "k": 2},
            {"scale": 24.0, "margin": 0.5, "lr": 0.01,  "optimizer": "sgd", "warmup": 100, "k": 2},
            # K=2 com LRs na zona de conforto do gráfico (5e-5 a 1e-4)
            {"scale": 24.0, "margin": 0.5, "lr": 1e-4,  "optimizer": "adamw", "warmup": 100, "k": 2},
            {"scale": 24.0, "margin": 0.5, "lr": 5e-5,  "optimizer": "adamw", "warmup": 100, "k": 2},
            
            # --- GRUPO K=3 (Flexível: Variações de iluminação/ruído) ---
            # Hipótese: 3 centros exigem mais "calma" para se organizarem no início.
            {"scale": 24.0, "margin": 0.5, "lr": 0.1,  "optimizer": "sgd", "warmup": 100, "k": 3},
            {"scale": 24.0, "margin": 0.5, "lr": 0.05,  "optimizer": "sgd", "warmup": 100, "k": 3},
            {"scale": 24.0, "margin": 0.5, "lr": 0.01, "optimizer": "sgd", "warmup": 100, "k": 3},
            # K=3 (Mais difícil, então testamos a LR vencedora e uma menor)
            {"scale": 24.0, "margin": 0.5, "lr": 5e-5,  "optimizer": "adamw", "warmup": 100, "k": 3},
            {"scale": 24.0, "margin": 0.5, "lr": 1e-5,  "optimizer": "adamw", "warmup": 100, "k": 3},
            ],
        "margin": [
            # ====================================================================
            # GRUPO K=2 (O Conservador Estável)
            # ====================================================================
            # Testamos se relaxar ou apertar a margem melhora o K=2
            {"scale": 24.0, "margin": 0.2, "lr": 5e-5, "optimizer": "adamw", "warmup": 100, "k": 2},
            {"scale": 24.0, "margin": 0.3, "lr": 5e-5, "optimizer": "adamw", "warmup": 100, "k": 2},
            {"scale": 24.0, "margin": 0.5, "lr": 5e-5, "optimizer": "adamw", "warmup": 100, "k": 2}, # Seu baseline atual
            {"scale": 24.0, "margin": 0.7, "lr": 5e-5, "optimizer": "adamw", "warmup": 100, "k": 2},
            {"scale": 24.0, "margin": 0.8, "lr": 5e-5, "optimizer": "adamw", "warmup": 100, "k": 2},

            # ====================================================================
            # GRUPO K=3 (O Flexível)
            # ====================================================================
            # K=3 tem mais centros, então talvez precise de mais margem (0.7) para
            # garantir que os grupos não se misturem, ou menos (0.3) para estabilizar.
            {"scale": 24.0, "margin": 0.2, "lr": 5e-5, "optimizer": "adamw", "warmup": 100, "k": 3},
            {"scale": 24.0, "margin": 0.3, "lr": 5e-5, "optimizer": "adamw", "warmup": 100, "k": 3},
            {"scale": 24.0, "margin": 0.5, "lr": 5e-5, "optimizer": "adamw", "warmup": 100, "k": 3}, # Seu baseline atual
            {"scale": 24.0, "margin": 0.7, "lr": 5e-5, "optimizer": "adamw", "warmup": 100, "k": 3},
            {"scale": 24.0, "margin": 0.8, "lr": 5e-5, "optimizer": "adamw", "warmup": 100, "k": 3},
        ]
    },
    "subcenter_cosface": {
        # Vamos chamar essa bateria de "k_lr_sweep" pois varia ambos
    "k": [
            # ====================================================================
            # HIPÓTESE 1: "A Força Bruta" (Scale 64 - Herdado do seu CosFace K=1)
            # ====================================================================
            # Mantemos a escala alta que funcionou no K=1.
            # Como S=64 é agressivo, testamos K=2 com a LR original e uma reduzida.
            {"scale": 24.0, "margin": 0.35, "lr": 1e-4, "optimizer": "adamw", "warmup": 100, "k": 2}, # Aposta agressiva
            {"scale": 24.0, "margin": 0.35, "lr": 5e-5, "optimizer": "adamw", "warmup": 100, "k": 2}, # Aposta segura
            
            # Para K=3, S=64 pode ser instável demais, então vamos só na LR segura
            {"scale": 24.0, "margin": 0.35, "lr": 5e-5, "optimizer": "adamw", "warmup": 100, "k": 3},

            # ====================================================================
            # HIPÓTESE 2: "Estabilidade para Sub-centros" (Scale 32)
            # ====================================================================
            # Sub-centros geralmente convergem melhor com escalas menores (30~32).
            # Se o grupo acima falhar (Loss reta), este grupo deve salvar o dia.
            {"scale": 24.0, "margin": 0.35, "lr": 1e-4, "optimizer": "adamw", "warmup": 100, "k": 2},
            {"scale": 24.0, "margin": 0.35, "lr": 1e-4, "optimizer": "adamw", "warmup": 100, "k": 3},
        ],
        "margin": [
            # ====================================================================
            # GRUPO K=2 (O Conservador Estável)
            # ====================================================================
            # Testamos se relaxar ou apertar a margem melhora o K=2
            {"scale": 24.0, "margin": 0.2, "lr": 5e-5, "optimizer": "adamw", "warmup": 100, "k": 2},
            {"scale": 24.0, "margin": 0.3, "lr": 5e-5, "optimizer": "adamw", "warmup": 100, "k": 2},
            {"scale": 24.0, "margin": 0.5, "lr": 5e-5, "optimizer": "adamw", "warmup": 100, "k": 2}, # Seu baseline atual
            {"scale": 24.0, "margin": 0.7, "lr": 5e-5, "optimizer": "adamw", "warmup": 100, "k": 2},

            # # ====================================================================
            # # GRUPO K=3 (O Flexível)
            # # ====================================================================
            # # K=3 tem mais centros, então talvez precise de mais margem (0.7) para
            # # garantir que os grupos não se misturem, ou menos (0.3) para estabilizar.
            # {"scale": 24.0, "margin": 0.2, "lr": 5e-5, "optimizer": "adamw", "warmup": 100, "k": 3},
            # {"scale": 24.0, "margin": 0.3, "lr": 5e-5, "optimizer": "adamw", "warmup": 100, "k": 3},
            # {"scale": 24.0, "margin": 0.5, "lr": 5e-5, "optimizer": "adamw", "warmup": 100, "k": 3}, # Seu baseline atual
            # {"scale": 24.0, "margin": 0.7, "lr": 5e-5, "optimizer": "adamw", "warmup": 100, "k": 3},
            # {"scale": 24.0, "margin": 0.8, "lr": 5e-5, "optimizer": "adamw", "warmup": 100, "k": 3},
        ]
    },
    # =========================================================
    # CONTRASTIVE LOSS (Siamese)
    # Inputs: Pares. Param: Margin.
    # =========================================================
    "contrastive": {
        "lr": [
            # Teste 1: LR Baixa (Segurança para pares) e Margem Padrão Euclidiana (1.0)
            {"margin": 1.0, "lr": 1e-5, "optimizer": "adamw", "scale": 1.0, "warmup": 100},

            # Teste 2: LR Média
            {"margin": 1.0, "lr": 3e-5, "optimizer": "adamw", "scale": 1.0, "warmup": 100},

            # Teste 3: LR Alta (5e-5) - Se funcionar, converge mais rápido
            {"margin": 1.0, "lr": 5e-5, "optimizer": "adamw", "scale": 1.0, "warmup": 100},
        ],
        
        # Só rode este se já tiver achado a LR ideal acima
        "margin": [
            # Margem Apertada (0.5): Relaxa a exigência de separação
            {"margin": 0.5, "lr": 3e-5, "optimizer": "adamw", "scale": 1.0, "warmup": 100},
            # Margem Agressiva (1.5): Força separação muito grande
            {"margin": 1.5, "lr": 3e-5, "optimizer": "adamw", "scale": 1.0, "warmup": 100},
        ]
    },

    # =========================================================
    # TRIPLET LOSS (Batch Hard)
    # Inputs: Batch. Param: Margin.
    # =========================================================
    "triplet": {
        "lr": [
            # Teste 1: LR Baixa (Crucial para Triplet não colapsar no início)
            # Margem 0.3 é o padrão ouro da literatura (FaceNet)
            {"margin": 0.3, "lr": 5e-6, "optimizer": "adamw", "scale": 1.0, "warmup": 100},

            {"margin": 0.3, "lr": 1e-5, "optimizer": "adamw", "scale": 1.0, "warmup": 100},

            # Teste 2: LR Média
            {"margin": 0.3, "lr": 3e-5, "optimizer": "adamw", "scale": 1.0, "warmup": 100},

            # Teste 3: LR Alta
            {"margin": 0.3, "lr": 5e-5, "optimizer": "adamw", "scale": 1.0, "warmup": 100},

            {"margin": 0.3, "lr": 1e-4, "optimizer": "adamw", "scale": 1.0, "warmup": 100},

            {"margin": 0.3, "lr": 5e-4, "optimizer": "adamw", "scale": 1.0, "warmup": 100},

            # {"margin": 0.3, "lr": 1e-3, "optimizer": "adamw", "scale": 1.0, "warmup": 100},
        ],

        # Só rode este se já tiver achado a LR ideal acima
        "margin": [
            # Margem Pequena (0.2): Foco em consistência local
            {"margin": 0.2, "lr": 5e-5, "optimizer": "adamw", "scale": 1.0, "warmup": 100},
            # Margem Grande (0.5): "Hard Margin", força clusters muito compactos
            {"margin": 0.5, "lr": 5e-5, "optimizer": "adamw", "scale": 1.0, "warmup": 100},
        ]
    }
}



# ==============================================================================
# 🛠️ UTILS
# ==============================================================================

def prepare_data(protocol="zsl", split_idx=1):
    num_pairs = PROTOCOL_DEFAULTS['pairs']
    # Adiciona o número de pares ao nome do diretório para evitar cache incorreto
    generated_data_dir = os.path.join(WORKSPACE_ROOT, f"data/generated_splits/RVL-CDIP_{protocol}_split_{split_idx}_{num_pairs}pairs")
    pairs_csv = os.path.join(generated_data_dir, "train_pairs.csv")
    
    if not os.path.exists(pairs_csv):
        print(f"🛠️  Generating training pairs for {protocol} split {split_idx} ({num_pairs} pairs/class)...")
        cmd_prep = [
            sys.executable, PREP_SCRIPT_PATH,
            "--data-root", RAW_DATA_ROOT,
            "--output-dir", generated_data_dir,
            "--split-idx", str(split_idx),
            "--protocol", protocol,
            "--pairs-per-class", str(num_pairs)
        ]
        subprocess.run(cmd_prep, check=True)
    return pairs_csv

def run_suite(experiment_name, sweep_param, experiments, dry_run=False):
    pairs_csv = prepare_data()
    
    # Dynamic Project Name for better organization
    # UNIFIED PROJECT NAME WITH GROUPS
    wandb_project = "CaVL-Protocol-Sweeps" 
    wandb_group = f"{experiment_name}_{sweep_param}" # ex: arcface_lr, arcface_scale
    
    print(f"\n🎯 Running Protocol V3 | Exp: {experiment_name} | Sweep: {sweep_param}")
    print(f"📂 WandB Project: {wandb_project} | Group: {wandb_group}")
    print("=" * 60)

    for i, exp in enumerate(experiments):
        # Auto-generate description keys
        # Format: Arcface_OptSGD_S64.0_M0.5_LR0.0003
        desc = f"Opt{exp.get('optimizer', 'adamw')}_S{exp['scale']}_M{exp['margin']}_LR{exp['lr']}"
        
        # Se tiver SubCenters, adiciona ao nome
        if 'k' in exp:
            desc += f"_K{exp['k']}"

        # REMOVIDO o sweep_param do nome para permitir compartilhamento de checkpoints entre sweeps
        # Adicionei o wandb_group como prefixo APENAS para log visual se quiser, mas o run_name deve ser único por config
        run_name = f"{experiment_name}_{desc}"
        
        print(f"\n▶️  Running {i+1}/{len(experiments)}: {run_name}")
        
        # Check for existing checkpoint to resume
        base_checkpoint_dir = "/mnt/large/checkpoints" if os.path.exists("/mnt/large") else "checkpoints"
        possible_ckpt = os.path.join(base_checkpoint_dir, run_name, "last_checkpoint.pt")
        
        resume_args = []
        if os.path.exists(possible_ckpt):
             print(f"🔄 Checkpoint found! Resuming from: {possible_ckpt}")
             resume_args = ["--resume-from", possible_ckpt]

        # Build Command
        cmd = [
            sys.executable, SCRIPT_PATH,
            "--model-name", "InternVL3-2B",
            "--dataset-name", "RVL-CDIP",
            "--use-wandb",
            "--wandb-project", wandb_project,
            "--wandb-run-name", run_name,
            # ADICIONANDO O ARGUMENTO DE GRUPO (precisa ser suportado no script de treino ou via env var)
            # Como o script principal não tem --wandb-group explícito, vamos passar via env var que o wandb pega
        ]
        
        # Injetando Environment Variable para o Grupo
        env = os.environ.copy()
        env["WANDB_RUN_GROUP"] = wandb_group

        cmd += [
            # Loss & Optimizer
            "--loss-type", experiment_name,
            "--optimizer-type", exp.get('optimizer', 'adamw'),
            "--student-lr", str(exp['lr']),
            "--scheduler-type", "constant",
            
            # Loss Params
            "--scale", str(exp['scale']),
            "--margin", str(exp['margin']),
            "--num-sub-centers", str(exp.get('k', 3)), # Default 3 se não especificado
            
            # Training Params
            "--epochs", PROTOCOL_DEFAULTS['epochs'],
            "--max-steps-per-epoch", PROTOCOL_DEFAULTS['max_steps'],
            "--student-batch-size", PROTOCOL_DEFAULTS['batch_size'],
            "--gradient-accumulation-steps", PROTOCOL_DEFAULTS['grad_accum'],
            "--candidate-pool-size", PROTOCOL_DEFAULTS['pool_size'],
            
            # Curriculum
            "--professor-warmup-steps", str(exp['warmup']),
            "--easy-mining-steps", str(exp['warmup']),
            "--professor-lr", PROTOCOL_DEFAULTS['prof_lr'],
            
            # Model details
            "--cut-layer", PROTOCOL_DEFAULTS['cut_layer'],
            "--projection-output-dim", PROTOCOL_DEFAULTS['proj_dim'],
            
            # Paths
            "--base-image-dir", BASE_IMAGE_DIR,
            "--pairs-csv", pairs_csv
        ] + resume_args

        if dry_run:
            print(f"Env: WANDB_RUN_GROUP={wandb_group}")
            print(" ".join(cmd))
            continue

        try:
            subprocess.run(cmd, check=True, env=env)  # Passando o env modificado
            print("✅ Success")
        except subprocess.CalledProcessError:
            print("❌ Failed")
        except KeyboardInterrupt:
            print("\n🛑 Interrupted")
            break
        
        time.sleep(2)

# ==============================================================================
# 🚀 MAIN
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CaVL Sweep Runner Protocol V3")
    parser.add_argument("--experiment", type=str, required=True, help="Loss type (e.g. arcface, cosface)")
    parser.add_argument("--parameter", type=str, required=True, help="Parameter to sweep (e.g. scale, margin)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    
    args = parser.parse_args()
    
    if args.experiment not in EXP_REGISTRY:
        print(f"Error: Experiment '{args.experiment}' not found in registry.")
        print(f"Available: {list(EXP_REGISTRY.keys())}")
        sys.exit(1)
        
    if args.parameter not in EXP_REGISTRY[args.experiment]:
        print(f"Error: Parameter '{args.parameter}' not defined for {args.experiment}.")
        print(f"Available: {list(EXP_REGISTRY[args.experiment].keys())}")
        sys.exit(1)
        
    experiments = EXP_REGISTRY[args.experiment][args.parameter]
    run_suite(args.experiment, args.parameter, experiments, args.dry_run)
