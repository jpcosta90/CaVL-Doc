#!/usr/bin/env python3
import wandb
import subprocess
import sys
import argparse
import hashlib
import json

def get_run_command(config, target_project):
    """
    Reconstrói o comando de treinamento usando a config do WandB,
    mas injetando os caminhos do RVL-CDIP.
    """
    # 1. Mapeamento de Argumentos Fixos para o RVL-CDIP
    cmd = [
        sys.executable, "scripts/training/run_cavl_training.py",
        "--dataset-name", "RVL-CDIP",
        "--pairs-csv", "data/RVL-CDIP/train_pairs.csv",
        "--base-image-dir", "/mnt/data/rvl-cdip-small-200",
        "--use-wandb"
    ]

    # 2. Lista de Hiperparâmetros a serem copiados do experimento original
    # (Nomes exatos das chaves no wandb.config)
    hyperparams = [
        "model_name", "epochs", "training_sample_size", 
        "student_lr", "professor_lr", "candidate_pool_size", "student_batch_size",
        "cut_layer", "projection_output_dim", "patience",
        "baseline_alpha", "entropy_coeff", "lr_reduce_factor", "max_num_image_tokens",
        # Modulares
        "loss_type", "pooler_type", "head_type", "num_queries", "num_classes",
        # Args específicos de Loss
        "margin", "scale", "num_sub_centers"
    ]

    # 3. Constrói o comando dinamicamente
    for param in hyperparams:
        if param in config:
            # Converte o nome do python (student_lr) para CLI (--student-lr)
            cli_arg = "--" + param.replace("_", "-")
            value = str(config[param])
            cmd.extend([cli_arg, value])

    # 4. Tratamento especial para Booleans (Flags)
    if config.get("load_in_4bit", False):
        cmd.append("--load-in-4bit")

    # 5. Nome do Run (Para rastrear a origem)
    # Ex: "From-LA-CDIP_contrastive_mlp"
    run_name = f"{config.get('loss_type')}_{config.get('head_type')}_{config.get('pooler_type')}"
    cmd.extend(["--wandb-run-name", run_name])

    # 6. Adiciona o target project como último argumento
    cmd.extend(["--wandb-project", target_project])

    return cmd

def main(args):
    api = wandb.Api()
    
    # Caminho do projeto original (Usuário/Projeto)
    # Se der erro de entidade, adicione entity="seu_usuario" no api.runs
    source_path = f"{args.entity}/{args.source_project}"
    print(f"🔍 Buscando experimentos em: {source_path}...")
    
    try:
        runs = api.runs(source_path)
    except Exception as e:
        print(f"❌ Erro ao acessar WandB: {e}")
        return

    print(f" -> Encontrados {len(runs)} runs no total.")
    
    # Filtrar e Deduplicar
    unique_configs = set()
    commands_to_run = []

    for run in runs:
        # Só queremos experimentos que terminaram (ignoramos crashed/running)
        if run.state != "finished" and not args.force_all:
            continue

        config = run.config
        
        # Cria um hash da configuração para evitar rodar a mesma coisa 2 vezes
        # (Filtramos chaves irrelevantes como 'wandb_version' ou caminhos antigos)
        relevant_keys = sorted([k for k in config.keys() if k not in ['dataset_name', 'pairs_csv', 'base_image_dir', 'wandb_run_name', 'wandb_project']])
        config_str = json.dumps({k: config[k] for k in relevant_keys}, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()

        if config_hash not in unique_configs:
            unique_configs.add(config_hash)
            cmd = get_run_command(config, args.target_project)
            commands_to_run.append(cmd)
    
    print(f"✅ {len(commands_to_run)} configurações únicas extraídas para reprodução.")
    print("-" * 60)

    # Execução
    for i, cmd in enumerate(commands_to_run):
        print(f"\n🚀 [Experimento {i+1}/{len(commands_to_run)}] Iniciando...")
        print(f"Comando: {' '.join(cmd)}")
        
        if not args.dry_run:
            try:
                # Executa e espera terminar antes de ir para o próximo
                subprocess.run(cmd, check=True)
                print(f"✅ Experimento {i+1} concluído com sucesso.")
            except subprocess.CalledProcessError as e:
                print(f"❌ Experimento {i+1} falhou. Continuando para o próximo...")
                continue
            except KeyboardInterrupt:
                print("\n🛑 Interrompido pelo usuário.")
                sys.exit(0)
        else:
            print("   (Dry-Run: Apenas simulação, comando não executado)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migra experimentos do WandB de um dataset para outro.")
    parser.add_argument("--entity", type=str, default="jpcosta1990-university-of-brasilia", help="Seu usuário/org no WandB")
    parser.add_argument("--source-project", type=str, default="CaVL-Doc-Experiments-LA-CDIP", help="Projeto Origem (WandB)")
    parser.add_argument("--target-project", type=str, default="CaVL-Doc-Experiments-RVL-CDIP", help="Projeto Destino (WandB)")
    parser.add_argument("--dry-run", action="store_true", help="Apenas lista os comandos sem executar")
    parser.add_argument("--force-all", action="store_true", help="Inclui runs falhados/crashed também")
    
    args = parser.parse_args()
    main(args)