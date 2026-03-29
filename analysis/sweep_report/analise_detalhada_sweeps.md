# Análise detalhada dos sweeps (coarse + fine)

## 1) Reestabelecimento do objetivo e critérios usados

Esta análise resume os resultados dos sweeps de **duas fases** para os datasets **RVL-CDIP** e **LA-CDIP** com o enquadramento correto:

- **Fase coarse**: exploração ampla do espaço de hiperparâmetros com orçamento reduzido.
- **Fase fine search**: validação dos candidatos em regime mais fiel/exigente (maior orçamento efetivo), com espaço de busca reduzido.

Logo, coarse e fine **não devem ser tratados como competição direta** por melhor EER absoluto, mas como pipeline de descoberta + confirmação de robustez.

## 📂 Tratamento de Splits e Protocolos

A dúvida sobre "qual split usar para treino" é fundamental. Aqui está a comparação:

| Item | Legacy (Legado) | Atual (Novo Pipeline) |
| :--- | :--- | :--- |
| **Protocolo** | `zsl` (padrão) | `zsl` (padrão em `analyze_and_build_finesearch.py`) |
| **Índice do Split** | `1` (padrão) | `1` (padrão em `analyze_and_build_finesearch.py`) |
| **Definição de Treino** | Todos os splits EXCETO o `split_idx`. | Todos os splits EXCETO o `split_idx` (mantido em `prepare_splits.py`). |
| **Pares por Classe** | 15 (muito pequeno) | Configurável (padrão 100 ou 200). |
| **Caminho do CSV** | Dinâmico: `..._split_1_15pairs/...` | Dinâmico: `..._split_1_{N}pairs/...` |

### ⚠️ Observação Importante sobre os YAMLs Base
Notei que os arquivos YAML em `scripts/sweeps/configs/rvlcdip/` estão apontando para um caminho estático: `data/RVL-CDIP/train_pairs.csv`. 
- Este arquivo possui **9.000 pares**, o que é uma base muito mais larga do que os 15 pares/classe do legacy.
- Se você deseja reproduzir exatamente o comportamento do legacy com os scripts novos, você deve usar o `analyze_and_build_finesearch.py` passando `--pairs-per-class 15`, o que forçará a geração do caminho dinâmico igual ao antigo.

---

## 🎯 Conclusão
O **tratamento dos splits** no novo pipeline **segue a lógica do legacy** (Protocolo `zsl`, Split `1` para validação e o restante para treino). A única diferença é a escala (quantidade de pares), que agora é maior por padrão para garantir melhor convergência, mas a lógica de exclusão mútua entre treino e validação permanece idêntica.

Crítica de processo (antes da crítica de resultado):

- **Suficiência do coarse**: suficiente para exploração e priorização inicial, não suficiente para concluir robustez sozinho.
- **Suficiência das regras do fine**: boas para primeira versão (space reduction + ranges data-driven), porém insuficientes para validação robusta sem ablações e sensibilidade por grupo.
- **Impacto de orçamento**: plausível e relevante; com `run_cap` curto, parte das diferenças coarse→fine pode refletir subexploração e não ausência real de sinal.

Os critérios implementados no pipeline foram:

1. Métrica-alvo principal: **val/eer** (menor é melhor).
2. Métricas derivadas por run:
   - `eer_final = min(val/eer)`
   - `eer_drop = eer_epoch1 - eer_final`
   - `eer_stability = std(val/eer)`
3. Filtro de runs tóxicos:
   - `diverged = eer_last > eer_first`
   - remoção de `eer_final >= eer_ceiling`
4. Inércia de parâmetros:
   - parâmetro considerado inerte se `|corr(param, eer_final)| < 0.3`
5. Score composto para ranqueamento:
   - `score = eer_final - 0.5*eer_drop + eer_stability`
6. Seleção por grupo (`loss + k`) com top-k por menor score.

### 1.1 Respostas diretas às perguntas centrais

**(A) Achamos de forma defensável qual perda funciona melhor?**
- **Sim, parcialmente**. Em LA-CDIP, há evidência mais consistente para `subcenter_arcface_k3` e `subcenter_cosface_k3`.
- **Não, ainda não de forma definitiva global** para todos os grupos/famílias, pois a transferência coarse→fine foi heterogênea.

**(B) Achamos uma arquitetura que aprende de fato?**
- **Sim, no contexto LA-CDIP e setup atual** (mesma arquitetura base com cabeças/loss diferentes) há aprendizado real e estável em subgrupos.
- Isso **não confirma** superioridade arquitetural global em todos os datasets/grupos.

**(C) O filtro coarse→fine foi bom e defensável?**
- **Defensável como triagem inicial**, pois reduz espaço de busca e custo.
- **Insuficiente como critério final isolado**, pois `eer_ceiling + top-k + score` não garante robustez sob mudança de regime (fine).

### 1.2 Quadro de decisão (status atual)

| Pergunta | Status | Leitura prática |
|---|---|---|
| Já sabemos qual perda funciona melhor? | **Parcial** | Em LA, `subcenter_arcface_k3` e `subcenter_cosface_k3` são as candidatas mais defensáveis hoje. |
| Já sabemos se “aprende de fato”? | **Sim (LA), parcial global** | Há aprendizagem consistente no LA em subgrupos; não há confirmação global para todos os grupos/datasets. |
| O filtro coarse→fine está validado? | **Triagem validada; decisão final não** | Manter filtro de entrada, mas exigir gate multi-métrica no fechamento. |
| O fine confirmou as hipóteses? | **Parcial** | Confirmou em parte dos grupos, refutou ganho amplo automático. |
| Próximo passo correto? | **Treino final enxuto no LA** | Confirmar 2 candidatas por seeds, escolher 1 vencedora por média+variância e então treinar final. |

---

## 2) RVL-CDIP

### 2.1 Resumo quantitativo

- `eer_ceiling`: **0.35**
- Runs coarse totais: **135**
- Runs coarse sobreviventes: **26** (taxa de sobrevivência: **19.26%**)
- Runs fine: **50**
- Grupos comparáveis coarse vs fine: **5**

### 2.2 Avaliação por loss (coarse)

Melhor loss no coarse:
- **triplet**
  - melhor EER: **0.280112**
  - EER médio: **0.341151**

Demais losses com melhores EERs no coarse:
- subcenter_arcface: **0.293226**
- subcenter_cosface: **0.312707**
- contrastive: **0.329004**

### 2.3 Transferência de sinal coarse → fine por grupo (loss+k)

Resultado global no RVL:
- grupos com `fine_best < coarse_best`: **1/5 (20%)**
- delta médio `fine_best - coarse_best`: **+0.019954**
- delta mediano: **+0.023066**

#### Melhora observada
- `subcenter_cosface_k3`: **0.329004 → 0.327610** (delta **-0.001394**)

#### Grupos com perda de patamar sob regime fine
- `triplet_k3`: **0.280112 → 0.318675** (delta **+0.038563**)
- `subcenter_arcface_k1`: **0.293226 → 0.323529** (delta **+0.030303**)
- `subcenter_cosface_k1`: **0.312707 → 0.335772** (delta **+0.023066**)

### 2.4 Interpretação técnica (RVL)

O coarse já encontrou bons mínimos (especialmente em `triplet_k3`) e o fine search não conseguiu, em geral, superar esse piso. Possíveis explicações:

- `run_cap` relativamente curto para Bayes em espaço já estreito;
- ranges refinados podem ter ficado **muito conservadores** em torno de ótimos locais;
- maior sensibilidade do RVL às combinações específicas de `lr`/`margin` com escala fixa por grupo.

---

## 3) LA-CDIP

### 3.1 Resumo quantitativo

- `eer_ceiling`: **0.05**
- Runs coarse totais: **173**
- Runs coarse sobreviventes: **119** (taxa de sobrevivência: **68.79%**)
- Runs fine: **96**
- Grupos coarse: **11**
- Grupos fine: **10**
- Grupos comparáveis coarse vs fine: **10**

### 3.2 Avaliação por loss (coarse)

No coarse, vários losses atingem EER muito baixo (mínimos próximos entre si). Exemplo:

- `triplet`: melhor EER **0.000000** (run específico), média **0.029208**
- grupos com melhor patamar recorrente: `contrastive_k3`, `circle_k3`, arcface/cosface em k variados com mínimos próximos de **0.007463** em diversos casos

### 3.3 Transferência de sinal coarse → fine por grupo (loss+k)

Resultado global no LA:
- grupos com `fine_best < coarse_best`: **3/10 (30%)**
- delta médio `fine_best - coarse_best`: **+0.012637**
- delta mediano: **+0.011301**

#### Melhores ganhos no fine
- `subcenter_arcface_k3`: **0.007463 → 0.005128** (delta **-0.002334**)
- `subcenter_cosface_k3`: **0.007463 → 0.005128** (delta **-0.002334**)
- `subcenter_arcface_k2`: **0.017879 → 0.016363** (delta **-0.001517**)

#### Grupos com perda de patamar sob regime fine
- `circle_k3`: **0.007463 → 0.046198** (delta **+0.038736**)
- `subcenter_arcface_k4`: **0.014925 → 0.043634** (delta **+0.028709**)
- `subcenter_cosface_k1`: **0.010417 → 0.032400** (delta **+0.021983**)

### 3.4 Interpretação técnica (LA)

No LA, houve sinais positivos em famílias `subcenter_*` com `k=3`, mas o comportamento é heterogêneo e alguns grupos pioraram bastante no fine. Isso sugere:

- o ajuste fino está funcionando melhor para grupos com geometria de embedding mais estável (arcface/cosface k3);
- alguns grupos podem exigir ranges mais amplos ou mais orçamento de runs;
- o coarse já estava próximo de ótimo para parte dos grupos.

---

## 4) Síntese comparativa

1. **RVL**: o coarse encontrou candidatos fortes, mas a transferência para o regime fine foi frágil na configuração atual.
2. **LA**: houve transferência positiva em subgrupos específicos (notadamente `subcenter_*_k3`), porém ainda heterogênea no conjunto.
3. O resultado central é de **validação parcial sob maior fidelidade**, e não de superioridade universal do fine.

---

## 5) Conclusões objetivas

- A estratégia coarse cumpriu bem o papel de exploração inicial.
- O fine deve ser interpretado como teste de robustez sob regime mais exigente.
- A avaliação científica deve priorizar:
   - persistência de ranking entre fases,
   - estabilidade (`eer_stability`) e comportamento de aprendizado (`eer_drop`),
   - ganho por família/grupo, não apenas melhor run global.

### 5.1 O que ficou confirmado vs refutado no fine search

**Confirmado**
- O coarse identifica candidatos úteis para validação posterior.
- Em LA-CDIP, famílias `subcenter_*` com `k=3` mostraram transferência positiva coarse→fine.
- O pipeline coarse→fine é metodologicamente válido para descoberta + validação.

**Refutado (na configuração atual)**
- Que o fine melhoraria de forma ampla a maioria dos grupos.
- Que o critério atual de filtro coarse→fine seria suficiente para decisão final sem gates adicionais.
- Que `best run` isolado basta para concluir desempenho robusto.

### 5.2 Próximo passo enxuto (novo treino)

1. **Selecionar 2 perdas candidatas no LA-CDIP**: `subcenter_arcface_k3` e `subcenter_cosface_k3`.
2. **Rodar confirmação curta** (3 seeds por candidata, config fixa) e comparar `best/mean/std`.
3. **Escolher 1 vencedora** por melhor média + menor variância (não apenas melhor run).
4. **Rodar treino final** com orçamento maior na vencedora e validar se o ganho persiste.
5. **Transferência mínima para RVL** com a receita vencedora do LA (sem novo sweep amplo).

### 5.3 Tabela final — top 5 perdas por dataset (coarse vs fine)

#### RVL-CDIP — Coarse (top perdas por best EER)

| Rank | Loss | Best EER |
|---:|---|---:|
| 1 | `triplet` | 0.280112 |
| 2 | `subcenter_arcface` | 0.293226 |
| 3 | `subcenter_cosface` | 0.312707 |
| 4 | `contrastive` | 0.329004 |

#### RVL-CDIP — Fine (top perdas por best EER)

| Rank | Loss | Best EER |
|---:|---|---:|
| 1 | `triplet` | 0.318675 |
| 2 | `subcenter_arcface` | 0.323529 |
| 3 | `subcenter_cosface` | 0.327610 |
| 4 | `contrastive` | 0.338236 |

#### LA-CDIP — Coarse (top perdas por best EER)

| Rank | Loss | Best EER |
|---:|---|---:|
| 1 | `circle` | 0.007463 |
| 2 | `contrastive` | 0.007463 |
| 3 | `subcenter_arcface` | 0.007463 |
| 4 | `subcenter_cosface` | 0.007463 |
| 5 | `triplet` | 0.007463 |

#### LA-CDIP — Fine (top perdas por best EER)

| Rank | Loss | Best EER |
|---:|---|---:|
| 1 | `subcenter_cosface` | 0.005128 |
| 2 | `subcenter_arcface` | 0.005128 |
| 3 | `triplet` | 0.016037 |
| 4 | `contrastive` | 0.021491 |
| 5 | `circle` | 0.046198 |

Leitura rápida:
- Em **RVL-CDIP**, coarse e fine mantêm o mesmo conjunto de famílias no topo, com `triplet` em 1º lugar em ambos.
- Em **LA-CDIP**, o fine reorganiza o topo para `subcenter_*` (arcface/cosface), enquanto `circle` cai para a 5ª posição.

---

## 6) Arquivos de suporte usados

- `analysis/sweep_report/sweep_report.html`
- `analysis/sweep_report/dataframes/rvlcdip_coarse_survivors.csv`
- `analysis/sweep_report/dataframes/rvlcdip_fine_all.csv`
- `analysis/sweep_report/dataframes/lacdip_coarse_survivors.csv`
- `analysis/sweep_report/dataframes/lacdip_fine_all.csv`
