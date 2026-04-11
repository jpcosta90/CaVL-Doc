# Cronograma de Fases da Pesquisa (CaVL-Doc)

## Objetivo
Consolidar o estado real do projeto em fases, destacando claramente o que já foi concluído e o que falta até a submissão.

## Legenda
- ✅ Concluído
- 🟡 Em andamento
- ⬜ Pendente

---

## Quadro de fases

| Fase | Escopo | Entregável principal | Início | Fim | Status script | Status processamento |
|---|---|---|---|---|---|---|
| 1 | Protocolo experimental (datasets, splits, métricas, seeds) | Protocolo base definido para LA-CDIP e RVL-CDIP | N/D (histórico) | N/D (histórico) | ✅ Concluído | ✅ Concluído |
| 2 | Pipeline base de treino/eval | Treino com checkpoints e rastreamento W&B operacional | N/D (histórico) | N/D (histórico) | ✅ Concluído | ✅ Concluído |
| 3 | Coarse Sweep (Stage 1) | Exploração ampla de hiperparâmetros por loss | 08/02/2026* | 20/02/2026* | ✅ Concluído | ✅ Concluído |
| 3B | Ciclo RVL-CDIP (coarse + fine, paralelo) | Registro consolidado do esforço RVL com janela de execução e horas consumidas (**301.90 h**) | 21/02/2026* | 12/03/2026* | ✅ Concluído | ✅ Concluído (baixa conversão em ganho final) |
| 4 | Fine Search (Stage 2) | Refinamento das melhores regiões do coarse + seleção top por loss | 14/03/2026* | 28/03/2026* | ✅ Concluído | ✅ Concluído |
| 5 | Sprint 1 (LA-CDIP): comparação das melhores configs por loss | Comparativo final entre losses com configs vindas do Fine Search | 29/03/2026 | 09/04/2026 | ✅ Concluído | ✅ Concluído |
| 6 | Sprint 2: Teacher Sweep (rede pequena) | Sweep Bayes por loss partindo da best_siam, com baseline OFF e 3 épocas | 13/04/2026 | 14/04/2026 | ✅ Concluído | 🟡 Em andamento |
| 7 | Sprint 3: Ablação + resultado final LA-CDIP | Matriz com/sem teacher × 5 splits × 2 losses + consolidação final no LA-CDIP | 15/04/2026 | 29/04/2026 | ⬜ Pendente | ⬜ Pendente |
| 8 | Sprint 4: Transfer learning + sweep de batch (RVL zero-shot) | Melhor loss/config, 5 épocas, com/sem transfer × 5 batch sizes (máx. 2h/época) | 30/04/2026 | 08/05/2026 | ⬜ Pendente | ⬜ Pendente |
| 9 | Sprint 5: Resultado final RVL zero-shot | 1 configuração final × 5 splits RVL | 11/05/2026 | 14/05/2026 | ⬜ Pendente | ⬜ Pendente |
| 10 | Sprint 6: Fechamento e submissão | Tabelas/figuras finais e escrita de resultados/discussão/conclusão | 15/05/2026 | 29/05/2026 | ⬜ Pendente | ⬜ Pendente |

\* Datas marcadas com asterisco são **janelas observadas nos logs de runs disponíveis nesta máquina** (CSV + W&B local/runtime). Elas não implicam ausência de trabalho fora desses logs (ex.: preparação, análise, ajustes offline ou runs em outro storage). No contexto deste projeto, o intervalo entre Coarse e Fine é compatível com processamento/rodadas de RVL-CDIP.

**Nota de interpretação (RVL-CDIP):** a fase foi relevante em tempo computacional e serviu como evidência negativa/diagnóstico, mas não apresentou conversão tão clara em melhoria de resultado final quanto o ciclo LA-CDIP.

---

## Fases já concluídas (detalhamento)

### Fase 3 — Coarse Sweep (Stage 1)
- Objetivo: explorar amplamente LR, margin, scale e k para as losses candidatas.
- Artefato consolidado: `analysis/sweep_report/dataframes/lacdip_coarse_all.csv`.
- Volume LA-CDIP: 173 runs (174 linhas no CSV incluindo cabeçalho).
- Resultado: triagem inicial das regiões promissoras por loss.

### Fase 4 — Fine Search (Stage 2)
- Objetivo: refinar os candidatos do coarse em regime mais fiel.
- Artefatos:
  - `scripts/optimization/coarse_search/configs/lacdip/fine_search/runs_raw.csv`
  - `analysis/sweep_report/analise_detalhada_sweeps.md`
- Volume LA-CDIP: 173 runs (174 linhas no CSV incluindo cabeçalho).
- Resultado: seleção das melhores configurações por loss para consumo da Sprint 1.

### Fase 5 — Sprint 1 (Concluída)
- Escopo correto: comparação das melhores configurações de cada loss no LA-CDIP.
- Dependência: usa diretamente os melhores configs vindos do Fine Search (`runs_raw.csv`).
- Evidência no código: `run_sprint1_top5_validation_lacdip.py` lê `--runs-csv` e chama `load_best_configs(...)`.
- Importante: nesta modelagem, Sprint 1 não inclui ablação teacher como objetivo principal.

---

## Fluxo real até o estado atual

1. Coarse Sweep (Fase 3)
2. Fine Search (Fase 4)
3. Sprint 1: comparação das melhores losses/configs (Fase 5, concluída)
4. Próximo bloco: Teacher (sweep + ablação) como sprints seguintes

---

## Tempo utilizado por fase (baseado no histórico de runs)

Fonte dos tempos: `_runtime` dos `wandb-summary.json` cruzado com `run_id` dos CSVs de fase.

| Fase | Fonte de runs | Runs (CSV) | Cobertura de runtime | Tempo total observado | Tempo médio por run | Janela observada |
|---|---|---:|---:|---:|---:|---|
| RVL — Coarse Sweep (paralelo) | `analysis/sweep_report/dataframes/rvlcdip_coarse_all.csv` | 135 | 135/135 (100%) | 135.56 h | 60.25 min | 2026-02-21 → 2026-03-02 |
| RVL — Fine Search (paralelo) | `analysis/sweep_report/dataframes/rvlcdip_fine_all.csv` | 50 | 50/50 (100%) | 166.34 h | 199.61 min | 2026-03-05 → 2026-03-12 |
| 3 — Coarse (LA-CDIP) | `analysis/sweep_report/dataframes/lacdip_coarse_all.csv` | 173 | 33/173 (19.1%) | 32.83 h | 59.69 min | 2026-02-18 → 2026-02-20 |
| 4 — Fine (LA-CDIP) | `analysis/sweep_report/dataframes/lacdip_fine_all.csv` | 96 | 96/96 (100%) | 311.62 h | 194.76 min | 2026-03-14 → 2026-03-28 |
| 5 — Sprint 1 (LA-CDIP) | W&B project `CaVL-Doc_LA-CDIP_Sprint1_Top5Validation` | 23 | 23/23 (100%) | 171.78 h | 448.13 min | 2026-03-29 → 2026-04-09 |

Notas de leitura:
- RVL-CDIP consumiu **301.90 h** no ciclo de sweep (coarse+fine), mesmo sem ganhos finais fortes; esse esforço agora está explicitamente contabilizado no cronograma.
- Coarse tem cobertura parcial de runtime local; o tempo total real da fase 3 é maior que 32.83 h.
- Fine tem cobertura completa para os runs listados no CSV.
- Para Sprint 1, os números acima foram calculados do projeto W&B completo (não apenas do `sprint1_runs_raw.csv`, que é um snapshot parcial).

---

## Estimativa de prazo das próximas etapas (ajustada)

Planejamento atualizado conforme definição experimental:

| Etapa | Definição operacional | Prazo definido |
|---|---|---:|
| Sprint 2 — Teacher Sweep | Rede professor pequena nas losses finalistas + contrastive, partindo de best_siam com ~3 épocas adicionais | **2 dias** |
| Sprint 3 — Ablação + LA-CDIP final | Com/sem teacher × 5 splits × 2 losses | **2 semanas** |
| Sprint 4 — Transfer + batch sweep RVL | Melhor loss/config, 5 épocas, com/sem transfer × 5 batch sizes, limite de 2h por época | **1 semana** |
| Sprint 5 — Resultado final RVL zero-shot | 1 configuração × 5 splits | **4 dias** |
| Sprint 6 — Fechamento/submissão | Consolidação final de resultados e escrita | **2 semanas** |

Total planejado (Sprints 2 a 6): **~6 semanas e 4 dias úteis**.

---

## Cronograma com prazos (calendário)

Data de referência: **11/04/2026**.

| Sprint/Fase | Duração | Janela de execução | Marco de entrega |
|---|---:|---|---|
| Sprint 2 (Fase 6) | 2 dias | **13/04/2026 → 14/04/2026** | Sweep teacher (rede pequena) concluído |
| Sprint 3 (Fase 7) | 2 semanas | **15/04/2026 → 29/04/2026** | Ablação + resultado final LA-CDIP |
| Sprint 4 (Fase 8) | 1 semana | **30/04/2026 → 08/05/2026** | Transfer + sweep de batch no RVL zero-shot |
| Sprint 5 (Fase 9) | 4 dias | **11/05/2026 → 14/05/2026** | Resultado final RVL (1 config × 5 splits) |
| Sprint 6 (Fase 10) | 2 semanas | **15/05/2026 → 29/05/2026** | Fechamento de submissão |

**Prazo alvo final:** **29/05/2026**

### Marcos de controle

- **M1 (14/04/2026):** Sprint 2 concluída.
- **M2 (29/04/2026):** Sprint 3 concluída (LA-CDIP fechado).
- **M3 (08/05/2026):** Sprint 4 concluída (transfer + sweep batch).
- **M4 (14/05/2026):** Sprint 5 concluída (resultado final RVL).
- **M5 (29/05/2026):** Submissão fechada.

---

## Próximas sprints (pendentes)

### Sprint 2 — Teacher Sweep (Fase 6)
- Rodar rede professor **pequena** nas losses com melhor desempenho + `contrastive`.
- Inicializar a partir de `best_siam`, com ~3 épocas adicionais.
- Reservar **2 dias** de processamento; número de experimentos proporcional ao tempo disponível.
- Entrega: melhor configuração teacher para seguir para ablação.
- Status atual: scripts prontos; processamento ainda em andamento.

### Sprint 3 — Ablação Teacher (Fase 7)
- Rodar ablação com e sem teacher + resultados finais no LA-CDIP.
- Matriz experimental: com/sem teacher × 5 splits × 2 losses.
- Duração alvo: **2 semanas**.
- Entrega: tabela final LA-CDIP + análise de ganho com teacher.

### Sprint 4 — Transfer + sweep de batch no RVL zero-shot (Fase 8)
- Usar apenas a loss/configuração de melhor desempenho.
- Treinar por **5 épocas**.
- Sweep: com e sem transfer learning × 5 tamanhos de lote.
- Restrição operacional: **máx. 2 horas por época**.
- Duração alvo: **1 semana**.

### Sprint 5 — Resultado final RVL zero-shot (Fase 9)
- Fixar 1 configuração final.
- Rodar nos 5 splits do protocolo zero-shot RVL-CDIP.
- Duração alvo: **4 dias**.

### Sprint 6 — Fechamento e submissão (Fase 10)
- Consolidar tabelas e figuras finais.
- Fechar texto de resultados, discussão e conclusão.
- Duração alvo: **2 semanas**.

---

## Checklist final
- [x] Coarse Sweep documentado como fase concluída
- [x] Fine Search documentado como fase concluída
- [x] Sprint 1 posicionada como comparação de losses e marcada como concluída
- [x] Teacher e ablação movidos para sprints seguintes
- [ ] Executar Sprint 2 (Teacher Sweep)
- [ ] Executar Sprint 3 (Ablação Teacher)
- [ ] Executar Sprint 4 (Transfer + sweep batch RVL)
- [ ] Executar Sprint 5 (Resultado final RVL)
- [ ] Executar Sprint 6 (Fechamento/submissão)

---

## Vinculação com GitHub Project

Este cronograma é a **visão macro** da pesquisa. A execução operacional é rastreada em:

- **Arquivo fonte**: [`docs/tasks.yaml`](../docs/tasks.yaml)
- **Ferramenta**: GitHub Project (Roadmap view)
- **Sincronização**: Script Python [`scripts/project/sync_tasks_to_github_project.py`](../scripts/project/sync_tasks_to_github_project.py)

### Como usar

**Primiera vez:**
1. Criar o GitHub Project como Roadmap
2. Anotar o `project_number` (URL: `github.com/users/<username>/projects/<project_number>`)
3. Rodar: `python scripts/project/sync_tasks_to_github_project.py --project-number <NUMBER>`

**Dia a dia:**
- Editar [`docs/tasks.yaml`](../docs/tasks.yaml) quando a estrutura mudar
- Sincronizar: `python scripts/project/sync_tasks_to_github_project.py --project-number <NUMBER>`
- Movimentar no Project (To do → In Progress → Done) conforme executa

**Ao terminar uma tarefa:**
- Fechar a issue via GitHub (`close` ou `fixes #<N>` em commit)
- Atualizar o status em `docs/tasks.yaml` (opcional)

Ver [`scripts/project/README.md`](../scripts/project/README.md) para detalhes de setup.