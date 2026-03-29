import streamlit as st
import os
import json
import glob
import pandas as pd
import subprocess
import sys
from cavl_doc.utils.tracking import fetch_wandb_runs # <--- Importe a nova função

# ... (Funções auxiliares load_experiment_history e run_training mantidas) ...

st.title("🧠 CaVL-Doc Manager")
st.markdown("Gerenciador de Treinamento e Histórico para Modelos Vision-Language.")

# ==========================================
# ATUALIZAÇÃO: 3 ABAS AGORA
# ==========================================
tab_new, tab_local, tab_wandb = st.tabs(["🚀 Novo Treinamento", "📂 Histórico Local", "☁️ Histórico WandB"])

# Configuração da Página
st.set_page_config(page_title="CaVL-Doc Manager", page_icon="🧠", layout="wide")

# --- FUNÇÕES AUXILIARES ---

def load_experiment_history(base_dir="checkpoints"):
    """Escaneia a pasta de checkpoints em busca de arquivos de configuração."""
    data = []
    # Procura recursivamente por arquivos JSON
    config_files = glob.glob(os.path.join(base_dir, "**", "training_config.json"), recursive=True)
    
    for cfg_file in config_files:
        try:
            with open(cfg_file, 'r') as f:
                config = json.load(f)
                # Adiciona o caminho relativo para referência
                config['checkpoint_path'] = os.path.dirname(cfg_file)
                
                # Tenta encontrar o best_siam.pt para saber se terminou
                if os.path.exists(os.path.join(config['checkpoint_path'], "best_siam.pt")):
                    config['status'] = "✅ Concluído"
                else:
                    config['status'] = "⚠️ Incompleto/Andamento"
                
                data.append(config)
        except Exception as e:
            continue
            
    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    # Move colunas importantes para o começo
    cols_order = ['timestamp', 'status', 'dataset_name', 'model_name', 'head_type', 'loss_type', 'training_sample_size']
    existing_cols = [c for c in cols_order if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in existing_cols]
    return df[existing_cols + remaining_cols].sort_values(by='timestamp', ascending=False)

def run_training(command_list):
    """Executa o comando no terminal e faz stream do output."""
    process = subprocess.Popen(
        command_list, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True
    )
    return process

# --- INTERFACE ---

st.title("🧠 CaVL-Doc Experiment Manager")
st.markdown("Gerenciador de Treinamento e Histórico para Modelos Vision-Language.")

tab_new, tab_history = st.tabs(["🚀 Novo Treinamento", "📜 Histórico de Experimentos"])

# ==========================================
# ABA 1: NOVO TREINAMENTO
# ==========================================
with tab_new:
    col_l, col_r = st.columns([1, 2])
    
    with col_l:
        st.subheader("Configuração Básica")
        
        dataset_name = st.selectbox("Dataset", ["RVL-CDIP", "LA-CDIP", "Custom"])
        model_name = st.text_input("Backbone Model", value="InternVL3-2B")
        
        # Caminhos (Padrões inteligentes)
        default_csv = f"data/{dataset_name}/train_pairs.csv"
        default_img_dir = f"/mnt/data/{dataset_name.lower()}-small-200"
        
        pairs_csv = st.text_input("Caminho CSV Treino", value=default_csv)
        base_img_dir = st.text_input("Diretório de Imagens", value=default_img_dir)
        
        st.subheader("Arquitetura Modular")
        # Aqui definimos as opções que criamos nos Registries
        pooler_type = st.selectbox("Pooler Type", ["attention", "mean"], help="Como agregar os tokens visuais.")
        head_type = st.selectbox("Head Type", ["mlp", "simple_mlp", "residual"], help="Arquitetura da cabeça de projeção.")
        loss_type = st.selectbox("Loss Function", ["contrastive", "arcface", "cosface"], help="Função de perda a ser otimizada.")
        
        load_4bit = st.checkbox("Load in 4-bit (QLoRA)", value=False)
        use_wandb = st.checkbox("Usar Weights & Biases", value=True)
        if use_wandb:
            wandb_proj = st.text_input("WandB Project", value="CaVL-Doc-Experiments")

    with col_r:
        st.subheader("Hiperparâmetros")
        c1, c2, c3 = st.columns(3)
        
        with c1:
            epochs = st.number_input("Épocas", min_value=1, value=5)
            sample_size = st.number_input("Amostras de Treino", min_value=0, value=2000, help="0 = Dataset inteiro")
            patience = st.number_input("Paciência (Early Stop)", value=3)
        
        with c2:
            student_lr = st.number_input("Learning Rate (Aluno)", value=1e-4, format="%.1e")
            prof_lr = st.number_input("Learning Rate (Prof)", value=1e-4, format="%.1e")
            cut_layer = st.number_input("Cut Layer (InternVL)", value=27)

        with c3:
            cand_pool = st.number_input("Candidate Pool Size", value=16, help="Quantos pares o Professor vê.")
            stud_batch = st.number_input("Student Batch Size", value=4, help="Quantos pares o Aluno treina.")
            proj_dim = st.number_input("Projection Dim", value=512)

        st.divider()
        st.subheader("Parâmetros RL & Avançados")
        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            entropy_c = st.number_input("Entropy Coeff", value=0.01, format="%.3f")
        with ac2:
            baseline_a = st.number_input("Baseline Alpha", value=0.01, format="%.3f")
        with ac3:
            max_tokens = st.number_input("Max Image Tokens", value=12)
            num_queries = st.number_input("Num Queries (Pooler)", value=1, min_value=1, help="Qtd de vetores de atenção (Multi-Query).")

    # --- MONTAGEM DO COMANDO ---
    cmd = [
        sys.executable, "scripts/training/run_cavl_training.py",
        "--dataset-name", dataset_name,
        "--model-name", model_name,
        "--pairs-csv", pairs_csv,
        "--base-image-dir", base_img_dir,
        "--epochs", str(epochs),
        "--training-sample-size", str(sample_size),
        "--student-lr", str(student_lr),
        "--professor-lr", str(prof_lr),
        "--candidate-pool-size", str(cand_pool),
        "--student-batch-size", str(stud_batch),
        "--cut-layer", str(cut_layer),
        "--projection-output-dim", str(proj_dim),
        "--patience", str(patience),
        "--baseline-alpha", str(baseline_a),
        "--entropy-coeff", str(entropy_c),
        "--max-num-image-tokens", str(max_tokens),
        "--num-queries", str(num_queries),
        "--pooler-type", pooler_type,
        "--head-type", head_type,
        "--loss-type", loss_type
    ]
    
    if load_4bit:
        cmd.append("--load-in-4bit")
    
    if use_wandb:
        cmd.append("--use-wandb")
        cmd.append("--wandb-project")
        cmd.append(wandb_proj)

    cmd_str = " ".join(cmd)

    st.markdown("### 🖥️ Comando Gerado")
    st.code(cmd_str, language="bash")

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        run_btn = st.button("▶️ Executar Treinamento", type="primary")
    
    # ... (código anterior do app.py) ...

    if run_btn:
        st.info("Iniciando processo... Acompanhe os logs abaixo.")
        output_area = st.empty()
        logs = []
        
        process = None # Inicializa variável
        
        try:
            # Inicia o processo
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True,
                bufsize=1, # Buffer de linha para aparecer rápido
                universal_newlines=True
            )
            
            # Loop de leitura seguro
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                
                if line:
                    logs.append(line)
                    # Mantém apenas as últimas 30 linhas para não travar o navegador
                    output_area.code("".join(logs[-30:]), language="text")
            
            if process.returncode == 0:
                st.success("Treinamento finalizado com sucesso! Verifique a aba 'Histórico'.")
            else:
                st.error(f"Ocorreu um erro. Código de saída: {process.returncode}")

        except Exception as e:
            st.warning("Processo interrompido pelo usuário ou erro de sistema.")
            st.error(f"Detalhe: {e}")
            
        finally:
            # O BLOCO DE SEGURANÇA
            # Se você clicar em "Stop" no Streamlit, este bloco é executado.
            if process and process.poll() is None:
                print("Matando processo zumbi...")
                process.terminate() # Tenta fechar educadamente
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill() # Mata forçado se não fechar
                st.warning("🛑 O processo de treinamento foi encerrado forçadamente.")

# ==========================================
# ABA 3: HISTÓRICO WANDB (NOVA)
# ==========================================
with tab_wandb:
    st.header("Histórico na Nuvem (Weights & Biases)")
    st.caption("Visualize métricas finais e comparações de todos os runs sincronizados.")

    # Inputs de Conexão
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        wb_entity = st.text_input("Entity/Usuário", value="jpcosta1990-university-of-brasilia")
    with c2:
        wb_proj = st.text_input("Projeto", value="CaVL-Doc-Experiments")
    with c3:
        st.write("") # Espaçamento
        btn_load_wb = st.button("🔄 Baixar do WandB")

    if btn_load_wb:
        with st.spinner(f"Baixando dados de {wb_entity}/{wb_proj}..."):
            try:
                df_wandb = fetch_wandb_runs(entity=wb_entity, project=wb_proj)
                if not df_wandb.empty:
                    # Salva na sessão para não perder ao recarregar
                    st.session_state['wandb_data'] = df_wandb
                    st.success(f"Carregados {len(df_wandb)} experimentos!")
                else:
                    st.warning("Nenhum run encontrado ou erro de conexão.")
            except Exception as e:
                st.error(f"Erro: {e}")

    # Exibição da Tabela WandB
    if 'wandb_data' in st.session_state:
        df = st.session_state['wandb_data']
        
        # 1. Filtro de Colunas (WandB traz muita coisa)
        all_cols = df.columns.tolist()
        # Colunas prioritárias que queremos ver
        priority_cols = [
            'name', 'status', 
            'val/best_eer', 'val/recall_at_1',  # Métricas Chave
            'loss_type', 'head_type', 'pooler_type', # Arquitetura
            'training_sample_size', 'epochs'
        ]
        # Interseção para garantir que existem
        cols_to_show = [c for c in priority_cols if c in all_cols]
        
        # 2. Ordenação Inteligente (Melhor modelo primeiro)
        if 'val/best_eer' in df.columns:
            df = df.sort_values(by='val/best_eer', ascending=True)

        st.dataframe(
            df,
            column_order=cols_to_show,
            column_config={
                "name": st.column_config.TextColumn("Run Name", width="medium"),
                "val/best_eer": st.column_config.NumberColumn("Melhor EER", format="%.4f"),
                "val/recall_at_1": st.column_config.NumberColumn("R@1 (k-NN)", format="%.4f"),
                "status": st.column_config.TextColumn("Status", width="small"),
            },
            use_container_width=True,
            hide_index=True
        )
        
        # 3. Comparador Rápido
        st.divider()
        st.subheader("Comparação Rápida")
        if 'val/best_eer' in df.columns and 'val/recall_at_1' in df.columns:
             # Gráfico de dispersão EER vs Recall (Trade-off)
             st.scatter_chart(
                 df, 
                 x='val/recall_at_1', 
                 y='val/best_eer',
                 color='loss_type', # Colore por tipo de loss para ver qual é melhor
                 size='epochs'
             )
             st.caption("Eixo X: Recall@1 (Maior é melhor) | Eixo Y: EER (Menor é melhor)")