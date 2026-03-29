# 📊 Data Manifest

Este diretório contém os arquivos de pares (treino/validação) utilizados para o treinamento métrico dos modelos. Os dados brutos (imagens) são assumidos como localizados em `/mnt/data/`.

## 📂 Estrutura de Dados

### 1. LA-CDIP (`data/LA-CDIP/`)
Conjunto de dados focado em documentos administrativos.
*   **Split Oficial**: **Split 0** (ZSL).
*   **Protocolo**: Zero-Shot Learning (`zsl_split`).
*   **Volume**: ~9.000 pares de treino e ~1.000 de validação.
*   **Observação**: Este arquivo é a versão consolidada de "produção".

### 2. RVL-CDIP (`data/RVL-CDIP/`)
Conjunto de dados de referência para classificação de documentos.
*   **Split Oficial**: **Split 1** (ZSL).
*   **Protocolo**: Zero-Shot Learning (`zsl_split`).
*   **Volume**: ~9.000 pares de treino e ~1.000 de validação.

### 3. Generated Splits (`data/generated_splits/`)
Diretório dinâmico utilizado pelo pipeline de **Optimization/Sweeps**.
*   Contém subpastas nomeadas conforme o protocolo, split e quantidade de pares por classe (ex: `LA-CDIP_zsl_split_1_100pairs/`).
*   Estes arquivos são gerados via `scripts/utils/prepare_splits.py`.

---

## 🛠️ Como Gerar Novos Splits

Caso queira utilizar um split ou protocolo diferente (ex: GZSL), utilize o script central:

```bash
python scripts/utils/prepare_splits.py \
  --data-root /mnt/data/la-cdip \
  --protocol gzsl \
  --split-idx 2 \
  --pairs-per-class 50
```

> [!NOTE]
> O roteiro de treinamento (`run_cavl_training.py`) utiliza o argumento `--pairs-csv` para apontar para qualquer um destes arquivos. Sempre verifique se o split usado no treino condiz com o split esperado na avaliação.
