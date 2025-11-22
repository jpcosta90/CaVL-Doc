#src/cavl_doc/utils/tracking.py

import pandas as pd
from datetime import datetime
import os

class ExperimentTracker:
    """
    Gerencia uma tabela de resultados de EER, salvando todos os parâmetros
    de cada experimento de forma flexível.
    """
    def __init__(self, file_path='results/default_results.csv'):
        """
        Inicializa o tracker, criando o diretório de resultados se necessário
        e carregando dados existentes.
        """
        self.file_path = file_path
        # Garante que o diretório de resultados exista antes de tentar carregar
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        self.results = self.load()
        print(f"ExperimentTracker inicializado. Rastreando resultados em '{self.file_path}'.")

    def load(self) -> pd.DataFrame:
        """Carrega a tabela de resultados de um arquivo CSV."""
        try:
            return pd.read_csv(self.file_path)
        except FileNotFoundError:
            return pd.DataFrame()

    def save(self):
        """Salva a tabela de resultados atual no arquivo CSV."""
        self.results.to_csv(self.file_path, index=False)
        print(f"Tabela de resultados salva em '{self.file_path}'.")

    def log(self, **kwargs):
        """
        Registra uma nova linha no log de experimentos com todos os dados fornecidos.
        Usa **kwargs para aceitar qualquer número de parâmetros de experimento.
        """
        # Adiciona um carimbo de data/hora universal a cada registro
        new_data = kwargs
        new_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        new_result_df = pd.DataFrame([new_data])
        
        # Concatena o novo resultado, garantindo que todas as colunas sejam mantidas
        if self.results.empty:
            self.results = new_result_df
        else:
            self.results = pd.concat([self.results, new_result_df], ignore_index=True)
        
        method_name = new_data.get('method_name', 'N/A')
        eer_score = new_data.get('eer', 'N/A')
        print(f"Resultado registrado: Método='{method_name}', EER={eer_score:.4f}")
        self.save()

    def display(self):
        """Exibe a tabela de resultados formatada, ordenada pelo EER."""
        print(f"\n--- Tabela de Resultados de '{os.path.basename(self.file_path)}' ---")
        if self.results.empty:
            print("Nenhum resultado registrado ainda.")
        else:
            # Define uma ordem de colunas preferencial para exibição
            display_cols = [
                'method_name', 'eer', 'model', 'metric', 
                'dataset', 'partition', 'timestamp', 'prompt_text'
            ]
            # Filtra apenas as colunas que realmente existem no DataFrame
            existing_cols = [col for col in display_cols if col in self.results.columns]
            
            # Ordena pelo EER para mostrar os melhores resultados primeiro
            sorted_df = self.results.sort_values(by='eer', ascending=True)
            print(sorted_df[existing_cols].to_string())
        print("---------------------------------------")