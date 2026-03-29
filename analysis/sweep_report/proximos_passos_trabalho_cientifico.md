# Próximos passos para trabalho científico (com base nos resultados dos sweeps)

## 1) Pergunta científica central

**Em quais condições o fine search melhora de forma consistente o desempenho sobre o coarse em RVL-CDIP e LA-CDIP?**

Os resultados atuais indicam melhora localizada (mais evidente em alguns grupos do LA), mas não melhoria global robusta.

---

## 2) Hipóteses testáveis

### H1 — Orçamento insuficiente no fine
O `run_cap` (10) pode ser baixo para a otimização Bayes convergir em cada grupo.

### H2 — Ranges de refino estreitos em excesso
Mesmo data-driven, os ranges podem ter ficado próximos demais de ótimos locais do coarse.

### H3 — Dependência de grupo (loss+k)
Famílias `subcenter_*` com `k=3` parecem responder melhor ao refino no LA; outras (ex.: circle) degradam.

### H4 — Escala fixa ajuda regularização, mas pode limitar alguns grupos
Fixar `scale` reduz variância e espaço de busca, porém pode impedir ajuste fino em grupos onde `scale` ainda é sensível.

---

## 3) Plano experimental recomendado (curto prazo)

## Bloco A — Robustez do fine (prioridade alta)
Para cada grupo finalista:
1. Repetir fine com `run_cap = {10, 20, 30}`.
2. Manter mesmo seed-set para comparação justa.
3. Reportar `best`, `mean`, `std` de EER por configuração.

**Critério de sucesso**: melhorar `best EER` e reduzir variância sem overfitting no val.

## Bloco B — Abertura controlada de ranges (prioridade alta)
Comparar três políticas de range:
- **Narrow**: atual.
- **Medium**: +50% de largura.
- **Wide**: +100% de largura (com clamps físicos).

**Objetivo**: verificar se regressões no fine foram causadas por range excessivamente restrito.

## Bloco C — Ablação de escala fixa vs semi-fixa (prioridade média)
Para grupos com regressão forte (ex.: circle_k3 no LA):
- Escala fixa (baseline atual).
- Escala semi-fixa (janela estreita em torno da mediana, ex. ±10%).

**Objetivo**: testar se fixar escala está bloqueando melhora em grupos específicos.

---

## 4) Plano de análise estatística para artigo

Para cada dataset e grupo (loss+k):

1. **Comparação pareada** coarse vs fine:
   - diferença de melhor EER (`Δbest`),
   - diferença de média (`Δmean`),
   - diferença de estabilidade (`Δstd`).
2. **Intervalo de confiança** por bootstrap (95%) para `Δbest` e `Δmean`.
3. **Teste de significância** não paramétrico quando aplicável (ex.: Mann–Whitney) entre distribuições de runs.
4. **Múltiplas comparações**: controlar FDR quando analisar muitos grupos.

---

## 5) Estrutura sugerida de resultados no paper

### 5.1 Tabela principal
- Melhor EER por dataset em coarse e fine.
- Ganho absoluto e relativo.

### 5.2 Tabela por loss family
- contrastive, triplet, circle, subcenter_arcface, subcenter_cosface.
- média/variância e melhor resultado em cada fase.

### 5.3 Tabela por grupo (loss+k)
- foco nos grupos com maior ganho e maior regressão.

### 5.4 Figura de trade-off
- eixo x: estabilidade (`eer_stability`), eixo y: `eer_final`.
- destacar runs dominantes (fronteira de Pareto).

### 5.5 Discussão
- quando fine search ajuda;
- quando coarse já é suficiente;
- relação entre tamanho do espaço de busca e robustez.

---

## 6) Riscos metodológicos e mitigação

1. **Viés de seleção por top-k**
   - Mitigar com validação em janelas top-k distintas (5, 10, 15).
2. **Overfitting ao val/eer**
   - Mitigar com avaliação holdout final fixa e cega.
3. **Dependência de seed**
   - Mitigar com múltiplas seeds e reporte de dispersão.
4. **Comparação desigual de orçamento**
   - Mitigar normalizando custo computacional entre estratégias.

---

## 7) Decisões práticas imediatas

1. Priorizar no próximo ciclo:
   - **RVL**: `triplet_k3`, `subcenter_arcface_k1`, `subcenter_cosface_k1`.
   - **LA**: `subcenter_arcface_k3`, `subcenter_cosface_k3`, `subcenter_arcface_k2`.
2. Rodar grid de orçamento (`run_cap` 10/20/30) nesses grupos.
3. Só depois escalar para todos os grupos.

---

## 8) Entregáveis recomendados para a próxima iteração

- `report_v2.html` com:
  - métricas por orçamento,
  - gráficos de sensibilidade de range,
  - estatística inferencial.
- `results_summary_v2.csv` consolidando coarse/fine por grupo.
- texto de 1–2 páginas de discussão para incorporar diretamente na seção de Experimentos do paper.
