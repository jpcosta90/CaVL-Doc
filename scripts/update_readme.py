# scripts/update_readme.py

import pandas as pd
import glob
import argparse
import os
import re

# Importa a função de plotagem
from cavl_doc.utils.visualization import generate_performance_plot

def generate_all_tables_string(results_pattern: str) -> str:
    """
    Encontra os CSVs de resultado, gera as tabelas Markdown e as retorna como string.
    """
    result_files = glob.glob(results_pattern)
    if not result_files:
        return "### No Results Found\n\n*Execute avaliações para gerar arquivos de resultado.*"

    df_list = [pd.read_csv(f) for f in result_files]
    if not df_list:
        return "### No Data\n\n*Arquivos encontrados mas vazios.*"

    results_df = pd.concat(df_list, ignore_index=True).drop_duplicates()

    if 'dataset' not in results_df.columns:
        return "### Erro\n\n*Os arquivos de resultado não contêm a coluna 'dataset'.*"

    all_tables_string = ""
    unique_datasets = sorted(results_df['dataset'].unique())
    
    for dataset_name in unique_datasets:
        dataset_df = results_df[results_df['dataset'] == dataset_name].copy()

        # Cria link para a imagem (presumindo o padrão de nomeação)
        dataset_df['Link Figura'] = dataset_df.apply(
            lambda row: f"[Link](results/plots/{row['dataset']}_{row['method_name']}.png)", axis=1
        )
        
        if 'eer' in dataset_df.columns:
            dataset_df = dataset_df.rename(columns={'eer': 'EER (%)'})
            # Se estiver em decimal (0.05), passa para % (5.0), senão mantém
            if dataset_df['EER (%)'].max() <= 1.0:
                dataset_df['EER (%)'] = (dataset_df['EER (%)'] * 100).round(2)
        
        display_columns = {
            'method_name': 'Method', 'EER (%)': 'EER (%)', 'model': 'Model/Adapter',
            'metric': 'Metric', 'Link Figura': 'Link Figura'
        }
        existing_cols = {k: v for k, v in display_columns.items() if k in dataset_df.columns}
        table_df = dataset_df[list(existing_cols.keys())].rename(columns=existing_cols)
        
        table_df.fillna('N/A', inplace=True)
        
        if 'EER (%)' in table_df.columns:
            table_df = table_df.sort_values(by='EER (%)', ascending=True)

        markdown_table = table_df.to_markdown(index=False)
        all_tables_string += f"### {dataset_name} Results\n\n" + markdown_table + "\n\n"
        
    return all_tables_string.strip()

def update_readme(readme_path: str, all_tables_content: str):
    """
    Substitui o conteúdo entre os marcadores no README.
    """
    start_marker = "## Results"
    end_marker = "### Performance vs. Parameters (LA-CDIP Dataset)" # Ajuste se seu README tiver outro marcador

    if not os.path.exists(readme_path):
        print(f"❌ ERRO: Arquivo README não encontrado em '{readme_path}'")
        return

    with open(readme_path, 'r', encoding='utf-8') as f:
        readme_content = f.read()

    replacement_block = f"{start_marker}\n\n{all_tables_content}\n\n{end_marker}"
    pattern = re.compile(f"{re.escape(start_marker)}.*?{re.escape(end_marker)}", re.DOTALL)
    
    new_readme_content, replacements_made = pattern.subn(replacement_block, readme_content)

    if replacements_made == 0:
        print(f"❌ AVISO: Marcadores não encontrados no README. Verifique se '{start_marker}' e '{end_marker}' existem.")
        # Fallback: Tenta adicionar ao final se não achar
        # new_readme_content = readme_content + "\n\n" + replacement_block
        return

    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(new_readme_content)
        
    print(f"✅ README.md atualizado com sucesso!")

# --- BLOCO PRINCIPAL ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-pattern', type=str, default='results/*_master_results.csv')
    parser.add_argument('--readme-path', type=str, default='README.md')
    args = parser.parse_args()

    # --- 1. DIAGNÓSTICO PRÉVIO DO CSV (Para debug do Baseline) ---
    csv_path = 'results/LA-CDIP_master_results.csv'
    
    print(f"\n🔍 Verificando arquivo de resultados: {csv_path}")
    if os.path.exists(csv_path):
        try:
            df_check = pd.read_csv(csv_path)
            methods = df_check['method_name'].astype(str).str.strip().unique()
            print(f"   -> Arquivo encontrado com {len(df_check)} linhas.")
            print(f"   -> Métodos encontrados: {methods}")
            
            has_baseline = any('baseline' in m.lower() for m in methods)
            if has_baseline:
                print("   ✅ Baseline encontrado no CSV!")
            else:
                print("   ⚠️ AVISO: Nenhum método contendo 'baseline' foi encontrado neste CSV.")
        except Exception as e:
            print(f"   ❌ Erro ao ler CSV: {e}")
    else:
        print(f"   ❌ ERRO CRÍTICO: O arquivo '{csv_path}' NÃO EXISTE. O gráfico será gerado vazio.")

    # --- 2. GERAÇÃO DO GRÁFICO ---
    print("\n📊 Chamando generate_performance_plot...")
    generate_performance_plot(la_cdip_results_path=csv_path)

    # --- 3. ATUALIZAÇÃO DO README ---
    print("\n📝 Gerando tabelas para o README...")
    all_tables_md = generate_all_tables_string(args.results_pattern)
    update_readme(args.readme_path, all_tables_md)