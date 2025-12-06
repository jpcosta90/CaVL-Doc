import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def generate_pairs(df, num_pairs_per_class=100, seed=42):
    """
    Gera pares de treino balanceados (50% pos, 50% neg) a partir de um DataFrame de documentos.
    df columns: ['class_name', 'doc_path', ...]
    """
    np.random.seed(seed)
    classes = df['class_name'].unique()
    pairs = []
    
    # Agrupa docs por classe
    class_docs = {c: df[df['class_name'] == c]['doc_path'].tolist() for c in classes}
    
    print(f"Gerando pares para {len(classes)} classes...")
    
    for c in classes:
        docs = class_docs[c]
        n_docs = len(docs)
        if n_docs < 2: continue
        
        # --- Positive Pairs ---
        # Gera num_pairs_per_class positivos para esta classe
        for _ in range(num_pairs_per_class):
            a, b = np.random.choice(docs, 2, replace=True) # Pode ser o mesmo? Idealmente não, mas para Augmentation sim. Vamos evitar.
            if a == b and n_docs > 1:
                # Tenta pegar outro
                b = np.random.choice(docs)
            
            pairs.append({
                'file_a_path': a,
                'file_b_path': b,
                'is_equal': 1,
                'class_a_name': c,
                'class_b_name': c
            })
            
        # --- Negative Pairs ---
        # Gera num_pairs_per_class negativos envolvendo esta classe
        for _ in range(num_pairs_per_class):
            doc_a = np.random.choice(docs)
            
            # Escolhe outra classe aleatória
            other_class = np.random.choice([x for x in classes if x != c])
            doc_b = np.random.choice(class_docs[other_class])
            
            pairs.append({
                'file_a_path': doc_a,
                'file_b_path': doc_b,
                'is_equal': 0,
                'class_a_name': c,
                'class_b_name': other_class
            })
            
    return pd.DataFrame(pairs)

def prepare_split(data_root, output_dir, split_idx=0, protocol='zsl', num_train_pairs_per_class=200):
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splits_csv_path = data_root / "splits.csv"
    protocol_csv_path = data_root / "protocol.csv"
    
    if not splits_csv_path.exists() or not protocol_csv_path.exists():
        raise FileNotFoundError(f"Arquivos CSV não encontrados em {data_root}")
        
    print(f"Lendo metadados de {data_root}...")
    df_splits = pd.read_csv(splits_csv_path)
    df_protocol = pd.read_csv(protocol_csv_path)
    
    # Coluna de split correta
    split_col = 'zsl_split' if protocol == 'zsl' else 'gzsl_split'
    
    # --- 1. Validation Pairs (from protocol.csv) ---
    # O protocol.csv define os pares de teste para o split específico
    print(f"Extraindo pares de validação para Split {split_idx} ({protocol})...")
    
    # Filtra protocol.csv
    # split_mode deve bater com o protocolo (ex: 'zsl_split')
    val_mask = (df_protocol['split_mode'] == split_col) & (df_protocol['split_number'] == split_idx)
    df_val_pairs = df_protocol[val_mask].copy()
    
    if len(df_val_pairs) == 0:
        print(f"⚠️ AVISO: Nenhum par encontrado no protocol.csv para {protocol} split {split_idx}.")
    
    # O protocol.csv tem 'file_a_name' e 'file_b_name'. Precisamos dos caminhos completos relativos.
    # O splits.csv tem o mapeamento 'doc_id' -> 'doc_path'.
    # Vamos criar um dicionário id -> path
    id_to_path = pd.Series(df_splits.doc_path.values, index=df_splits.doc_id).to_dict()
    id_to_class = pd.Series(df_splits.class_name.values, index=df_splits.doc_id).to_dict()
    
    # Mapeia os nomes dos arquivos para os caminhos
    # Assumindo que file_a_name no protocol é o doc_id (nome do arquivo)
    df_val_pairs['file_a_path'] = df_val_pairs['file_a_name'].map(id_to_path)
    df_val_pairs['file_b_path'] = df_val_pairs['file_b_name'].map(id_to_path)
    
    # Adiciona nomes das classes (para métricas por classe se necessário)
    df_val_pairs['class_a_name'] = df_val_pairs['file_a_name'].map(id_to_class)
    df_val_pairs['class_b_name'] = df_val_pairs['file_b_name'].map(id_to_class)
    
    # Remove pares onde não achamos o caminho (erro de mapeamento?)
    df_val_pairs = df_val_pairs.dropna(subset=['file_a_path', 'file_b_path'])
    
    val_out = output_dir / "validation_pairs.csv"
    df_val_pairs.to_csv(val_out, index=False)
    print(f"✅ Validation pairs salvos em: {val_out} ({len(df_val_pairs)} pares)")
    
    # --- 2. Training Pairs (Generated from Training Splits) ---
    # Treino = Todos os splits EXCETO o split_idx atual
    print(f"Gerando pares de treino (Splits != {split_idx})...")
    train_mask = df_splits[split_col] != split_idx
    df_train_docs = df_splits[train_mask]
    
    # Gera pares
    df_train_pairs = generate_pairs(df_train_docs, num_pairs_per_class=num_train_pairs_per_class)
    
    train_out = output_dir / "train_pairs.csv"
    df_train_pairs.to_csv(train_out, index=False)
    print(f"✅ Train pairs salvos em: {train_out} ({len(df_train_pairs)} pares)")
    
    return train_out, val_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="/mnt/data/la-cdip")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split-idx", type=int, default=0)
    parser.add_argument("--protocol", type=str, default="zsl", choices=["zsl", "gzsl"])
    parser.add_argument("--pairs-per-class", type=int, default=200)
    
    args = parser.parse_args()
    
    prepare_split(args.data_root, args.output_dir, args.split_idx, args.protocol, args.pairs_per_class)
