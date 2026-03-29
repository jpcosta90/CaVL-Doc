# 🤝 Guia de Contribuição e Boas Práticas (CaVL-Doc)

Este documento define os padrões de desenvolvimento e o fluxo de trabalho experimental para garantir que o projeto CaVL-Doc permaneça rastreável, organizado e reproduzível.

## 🧪 Workflow Experimental

Todo novo experimento de perda (Loss) ou arquitetura deve seguir estes quatro passos:

1.  **Preparação de Dados**: Gerar os pares necessários via `scripts/utils/prepare_splits.py`. Nunca edite manualmente os arquivos `train_pairs.csv` na pasta `data/`.
2.  **Otimização de Hiperparâmetros (Sweeps)**: 
    *   Lançar um **Coarse Sweep** (Stage 1) em `scripts/optimization/coarse_search/`.
    *   Analisar os resultados com o `analyze_and_build_finesearch.py` (Stage 2).
3.  **Treinamento Final**: Utilizar o `scripts/training/run_cavl_training.py` com o YAML gerado no Stage 2.
4.  **Avaliação**: Validar o checkpoint resultante usando o `scripts/evaluation/run_siamese_eval.py`.

---

## 📂 Padrão de Nomenclatura

*   **Projetos WandB**: `CaVL-Doc_[Dataset]_[Model]_[Loss]`
*   **Run Names**: `[Timestamp]_[Model]_[Config]` (ex: `20260329_InternVL_ArcFace_LR1e-4`)
*   **Pastas de Checkpoints**: Devem ser criadas automaticamente em `checkpoints/` seguindo o nome do Run do WandB.

---

## 💻 Padrões de Código

*   **Novas Perdas**: Devem ser implementadas em `src/cavl_doc/modules/losses.py` e registradas no `LOSS_REGISTRY`.
*   **Arquiteturas**: Devem herdar de `BaseCaVLModel` em `src/cavl_doc/models/`.
*   **Imports**: Sempre use imports absolutos: `from cavl_doc.models.backbone import ...`

---

## 🧹 Manutenção do Repositório

*   **Arquivos na Raiz**: A raiz do projeto deve conter apenas arquivos de configuração global (`README.md`, `LICENSE`, `setup.py`, `requirements.txt`). Scripts experimentais devem ir para as subpastas de `scripts/`.
*   **Documentação**: Ao adicionar um script novo, descreva-o no `README.md` do diretório correspondente.
*   **Atualização do README**: Após rodar novos experimentos e gerar melhores EERs, execute `python scripts/utils/update_readme.py` para atualizar as tabelas de resultados.
