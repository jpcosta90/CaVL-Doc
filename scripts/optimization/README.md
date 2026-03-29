# 🚀 Optimization & Hyperparameter Sweeps

Este diretório contém as ferramentas para a busca e refinamento de hiperparâmetros do CaVL-Doc. O processo é dividido em dois estágios principais para maximizar a eficiência computacional e a qualidade dos embeddings.

## 🛠️ Pipeline de Dois Estágios

### Estágio 1: Coarse Search (Busca Global)
O objetivo é explorar o espaço de busca de forma ampla para identificar regiões promissoras de **Learning Rate**, **Margin** e **Scale**.

*   **Local**: `scripts/optimization/coarse_search/`
*   **Método**: Bayesian Search (WandB).
*   **Como Executar**:
    ```bash
    # Inicializar todos os sweeps (RVL e LA-CDIP)
    bash scripts/optimization/coarse_search/init_sweeps.sh
    ```

### Estágio 2: Fine Search (Refinamento Data-Driven)
Após a conclusão do Estágio 1, utilizamos análise estatística para podar o espaço de busca e focar apenas nos parâmetros que realmente impactam a performance.

*   **Local**: `scripts/optimization/fine_search/`
*   **Ferramenta**: `analyze_and_build_finesearch.py`
*   **Processo**:
    1.  O script baixa os resultados do WandB.
    2.  Identifica parâmetros "inertes" (baixa correlação com EER).
    3.  Gera novos arquivos YAML de alta fidelidade para as regiões vencedoras.
*   **Como Executar**:
    ```bash
    python scripts/optimization/fine_search/analyze_and_build_finesearch.py \
      --project "Seu-Projeto-WandB" \
      --dataset lacdip
    ```

---

## 📂 Estrutura do Diretório

*   `coarse_search/`: Arquivos YAML originais e scripts de setup iniciais.
*   `fine_search/`: Scripts de análise, geração de novos sweeps e armazenamento de experimentos de refinamento.
*   `legacy/`: Arquivos históricos e scripts baseados em loops Python manuais (substituídos pelo WandB nativo).

---

## 📈 Rastreabilidade e Melhores Práticas

1.  **Nomenclatura**: Todos os sweeps gerados incluem o timestamp e o tipo de loss no nome do projeto.
2.  **Splits**: Por padrão, o Stage 1 utiliza o **Split 1** (ZSL). Certifique-se de que os dados foram preparados corretamente via `prepare_splits.py`.
3.  **Relatórios**: O Stage 2 gera um `sweep_report.html` que deve ser revisado antes de lançar os agentes de refinamento.
