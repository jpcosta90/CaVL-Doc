# Checklist executável — Fase 2 (LA-CDIP)

Use este arquivo como controle operacional de cada rodada do sweep.  
Marque `[x]` ao concluir e preencha os campos de evidência.

## Modo enxuto (recomendado agora)

Se o objetivo for custo-benefício máximo, execute apenas este bloco:

- [ ] Selecionar só 2 candidatas: `subcenter_arcface_k3` e `subcenter_cosface_k3`.
- [ ] Rodar 3 seeds por candidata com config fixa (sem novo sweep amplo).
- [ ] Comparar `best/mean/std` e escolher 1 vencedora por **média + variância**.
- [ ] Rodar 1 treino final com orçamento maior e validar persistência do ganho.
- [ ] Rodar transferência mínima no RVL com a vencedora (1 mini-ajuste, sem sweep novo).

> Se o modo enxuto falhar em robustez, usar o checklist completo abaixo.

---

## 0) Preparação

- [ ] Confirmar versão do código e configs usadas na rodada.
  - Commit/hash:
  - Script principal:
  - Data/hora:
- [ ] Confirmar datasets e projeto WandB corretos.
  - Entity:
  - Projeto coarse base:
  - Projeto fine alvo:
- [ ] Definir orçamento da rodada.
  - `run_cap` planejado:
  - `max_steps_per_epoch`:
  - Número de grupos:

---

## 1) Gate de entrada do coarse (confiança do candidato)

### 1.1 Construção dos candidatos
- [ ] Exportar/atualizar dataframe coarse survivors.
  - Arquivo:
- [ ] Calcular score de confiança `C_conf` por grupo (`loss+k`).
  - Fórmula usada:
  - Script:
- [ ] Definir limiar de entrada (ex.: percentil 60 do grupo).
  - Limiar adotado:

### 1.2 Aprovação para fine
- [ ] Apenas candidatos com `C_conf` acima do limiar seguiram para fine.
- [ ] Cobertura mínima por grupo foi atingida.
  - `runs_por_grupo` mínimo planejado:
  - `runs_por_grupo` mínimo observado:

**Evidência (links/arquivos):**
- 

---

## 2) Política por trilha de grupo

### Trilhas
- Trilha A (ganho):
- Trilha B (neutro):
- Trilha C (regressivo):

### Verificações
- [ ] Trilha A executada com foco em consolidação (orçamento maior).
- [ ] Trilha B executada com reteste controlado.
- [ ] Trilha C executada com redesign de range e ablação de regra.

**Evidência (grupos por trilha):**
- 

---

## 3) Sprint 1 — Validação Top-5 quase final + ablação professor

### Planejamento
- [ ] Selecionar Top-5 perdas: `subcenter_cosface`, `subcenter_arcface`, `triplet`, `contrastive`, `circle`.
- [ ] Definir melhor config base por perda (melhor `eer_final`, desempate por `eer_stability`).
- [ ] Rodar cada perda por `10` épocas com `max_steps_per_epoch` maior que o fine atual (referência: `140`).
- [ ] Rodar variante **com professor nas últimas 5 épocas** (`warmup = 5 * max_steps_per_epoch`).
- [ ] Rodar variante **sem professor nas últimas 5 épocas** (professor desligado nas 10 épocas).
- [ ] Manter demais hiperparâmetros fixos por perda para comparação justa.

### Gate Sprint 1
- [ ] Tabela comparativa pronta por perda: `best/mean/std` (com vs sem professor).
- [ ] Efeito do professor classificado por perda: ajuda / neutro / atrapalha.
- [ ] Top-2 perdas escolhidas para Sprint 2 por `mean EER` + `std`.

**Evidência:**
- Tabela `best/mean/std` por perda e variante:
- Link do relatório:

---

## 4) Sprint 2 — Sensibilidade de ranges

### Planejamento
- [ ] Testar políticas de range: Narrow / Medium / Wide.
- [ ] Orçamento igual por política (comparação justa).

### Gate Sprint 2
- [ ] Política dominante por grupo/família identificada.
- [ ] Regressões fortes reduziram com ajuste de range.

**Evidência:**
- Matriz grupo × política:
- Melhores políticas por grupo:

---

## 5) Sprint 3 — Ablação fixa vs semi-fixa

### Planejamento
- [ ] Rodar variante `fixa`.
- [ ] Rodar variante `semi-fixa` (ex.: ±10% da mediana).

### Gate Sprint 3
- [ ] Comparação pareada concluída por grupo regressivo.
- [ ] Política vencedora definida por desempenho + estabilidade.

**Evidência:**
- Tabela `Δbest_fix_vs_semifix`, `Δmean_fix_vs_semifix`:

---

## 6) Sprint 4 — Inferência estatística

### Planejamento
- [ ] Bootstrap 95% para `Δbest` e `Δmean`.
- [ ] Teste Mann–Whitney por grupo.
- [ ] Correção FDR para múltiplas comparações.

### Gate Sprint 4
- [ ] Tabela com IC + p-valor ajustado por grupo/família pronta.
- [ ] Conclusão estatística redigida (não apenas descritiva).

**Evidência:**
- Arquivo de resultados estatísticos:
- Resumo de significância:

---

## 7) Gate final da Fase 2 (aprovação científica)

Marque apenas quando TODOS os critérios forem atendidos:

- [ ] ≥ 50% dos grupos comparáveis com `Δbest < 0`.
- [ ] ≥ 40% dos grupos comparáveis com `Δmean < 0`.
- [ ] Grupos líderes (ex.: `subcenter_*_k3`) mantiveram ganho com variância controlada.
- [ ] Regressões críticas (ex.: `circle_k3`) foram mitigadas ou explicadas com evidência estatística.

**Resultado final da rodada:**
- [ ] APROVADO para consolidar LA-CDIP como eixo principal
- [ ] PARCIAL (repetir sprints específicos)
- [ ] REPROVADO (redefinir regras do fine)

**Observações finais:**
- 

---

## 8) Log operacional (preencher a cada rodada)

| Rodada | Data | Config principal | Grupos | run_cap | Status | Observação curta |
|---|---|---|---|---:|---|---|
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
