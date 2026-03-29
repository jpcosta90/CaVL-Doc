# LA-CDIP — Pontos em aberto e plano de ação (Fase 2)

## Contexto

Este documento consolida os pontos em aberto do ciclo atual de sweeps no **LA-CDIP** e propõe um plano de Fase 2 com base nos dados já processados.

Enquadramento metodológico adotado:

- **coarse** = exploração ampla de baixo custo;
- **fine** = validação dos candidatos em regime mais fiel/exigente.

Assim, a pergunta central não é apenas “o fine foi melhor?”, mas “o sinal encontrado no coarse se sustenta quando aumentamos fidelidade?”.

Dados-base:

- `analysis/sweep_report/dataframes/lacdip_coarse_survivors.csv`
- `analysis/sweep_report/dataframes/lacdip_coarse_finalists.csv`
- `analysis/sweep_report/dataframes/lacdip_fine_all.csv`
- `analysis/sweep_report/dataframes/lacdip_inert_params.csv`
- `analysis/sweep_report/sweep_report.html`

---

## 0) Questões em aberto e resposta objetiva

| Questão | Resposta atual | Evidência disponível | Próxima validação |
|---|---|---|---|
| O coarse foi suficiente? | **Sim, para exploração; não para conclusão final** | 173 runs totais, 119 sobreviventes; boa separação de candidatos | Gate de confiança (`C_conf`) antes de entrar no fine |
| As regras do fine foram boas? | **Boas como V1; insuficientes como decisão final** | Ganho em parte dos grupos, perda em outros | Trilha A/B/C + ablação fixa/semi-fixa + range sensitivity |
| O filtro coarse→fine foi defensável? | **Defensável para triagem; insuficiente isoladamente** | Preservou grupos fortes (`subcenter_*_k3`), mas não garantiu robustez global | Complementar com gate multi-métrica (`Δbest`,`Δmean`,`std`,`IC95%`) |
| O fine confirmou a hipótese? | **Parcialmente** | 3/10 grupos com `fine_best < coarse_best`; 2/10 em média | Consolidar com seeds e inferência estatística |
| Já dá para escolher configuração final? | **Sim, de forma operacional (não definitiva estatística)** | Candidatas dominantes no LA: `subcenter_arcface_k3`, `subcenter_cosface_k3` | Rodar 3 seeds por candidata e escolher por média+variância |

**Decisão executiva atual:** seguir com plano enxuto de confirmação no LA, depois treino final, depois transferência mínima no RVL.

---

## 1) Crítica da forma como chegamos aos resultados

Antes da crítica de resultados, a crítica correta é de **processo**.

## 1.1 Os coarse foram suficientes?

**Parcialmente suficientes.**

Pontos fortes:
- O coarse cumpriu o objetivo de exploração ampla em baixo custo.
- Houve massa crítica de candidatos para priorização (173 runs totais, 119 sobreviventes).
- O coarse identificou grupos claramente promissores (ex.: `subcenter_*_k3`).

Limitações:
- A profundidade por grupo (`loss+k`) foi desigual.
- O filtro por `eer_ceiling` foi útil para triagem, mas não garante robustez sob mudança de regime.
- Faltou um critério explícito de "confiança do candidato" ao final do coarse (ex.: rank + variância + consistência por seed).

Conclusão: o coarse foi suficiente para **gerar hipóteses**, mas não para concluir robustez sem validação adicional.

## 1.2 As regras do fine search foram boas? Foram suficientes?

**Boas para início, mas ainda insuficientes para conclusão robusta.**

Acertos:
- Espaço reduzido (lr + margin, scale fixa), coerente com objetivo de validação.
- Ranges data-driven derivados dos finalistas do coarse.
- Estratégia bayesiana com `run_cap` para controle de custo.

Lacunas:
- Faltou política diferenciada por trilha de grupo (ganho/neutro/regressivo).
- Faltou ablação de regras (fixa vs semi-fixa) para parâmetros inertes.
- Faltou análise sistemática de sensibilidade de ranges (Narrow/Medium/Wide).
- Faltou critério de aceite além de "best run" (ex.: média + dispersão).

Conclusão: as regras foram adequadas para operacionalizar a Fase 2, mas não suficientes para afirmar validação consistente coarse→fine em todo o LA-CDIP.

## 1.3 O orçamento pode ter impactado o resultado?

**Sim, de forma plausível e relevante.**

- `run_cap=10` é adequado para triagem, mas pode ser curto para estabilizar Bayes em grupos mais complexos.
- Há grupos com variância alta no fine (`subcenter_arcface_k2`, `subcenter_arcface_k1`, `subcenter_cosface_k3`).
- Sem curva explícita de convergência (`run_cap=10/20/30`), não dá para separar efeito de regra vs efeito de orçamento.

Conclusão: orçamento deve ser tratado como eixo experimental explícito da Fase 2.

## 1.4 Veredito do filtro coarse→fine (defensabilidade)

### O filtro foi bom?
**Sim, como filtro de triagem.**

- `eer_ceiling` + score composto + top-k reduziu custo e tornou o fine executável.
- O filtro preservou grupos com sinal real (especialmente `subcenter_*_k3`).

### O filtro foi suficiente para decisão final?
**Não, isoladamente.**

- Não garantiu robustez em todos os grupos sob regime fine.
- Precisa ser complementado por gate de confiança, análise de variância e confirmação por seed.

### Decisão prática
- **Manter** o filtro atual como etapa de entrada.
- **Revisar/complementar** a etapa com gate multi-métrica antes da decisão final.

---

## 1.5 Matriz de fechamento de gaps (gap → mecanismo → métrica → gate)

| Gap identificado | Mecanismo no plano | Métrica de acompanhamento | Gate (aprova/reprova) |
|---|---|---|---|
| Profundidade desigual por grupo (`loss+k`) | Orçamento mínimo por grupo + alocação por trilha (A/B/C) | `runs_por_grupo`, cobertura de grupos comparáveis | Aprova se todos os grupos-alvo atingirem orçamento mínimo planejado |
| `eer_ceiling` útil para triagem, mas fraco para robustez | Usar `eer_ceiling` só como filtro inicial; decisão final no fine | `Δbest`, `Δmean`, `std_fine` por grupo | Aprova se decisão final não depender só de ceiling e considerar métricas de transferência |
| Falta de confiança do candidato ao fim do coarse | Score de confiança do candidato (rank + estabilidade + consistência + sinal de aprendizado) | `C_conf`, rank por grupo | Aprova se candidatos enviados ao fine estiverem acima de limiar de confiança |
| Falta de política por trilha (ganho/neutro/regressivo) | Execução por trilhas A/B/C com regras distintas | resultado por trilha (`Δbest`, `Δmean`) | Aprova se cada trilha seguir política própria e gerar leitura separada |
| Falta de ablação fixa vs semi-fixa | Sprint dedicado de ablação em grupos regressivos | `Δbest_fix_vs_semifix`, `Δmean_fix_vs_semifix` | Aprova se a melhor política ficar estatisticamente melhor definida |
| Falta de sensibilidade de ranges | Teste Narrow/Medium/Wide com orçamento controlado | curva de desempenho por range | Aprova se houver política dominante estável por grupo/família |
| Falta de critério além de best run | Gate multi-métrica por grupo | `Δbest`, `Δmean`, `std`, IC95% | Aprova se passar no gate composto (não apenas no best run) |

### Definição operacional do score de confiança no coarse

Para cada candidato de coarse, calcular um score de confiança `C_conf` normalizado:

`C_conf = 0.40 * rank_score + 0.20 * estabilidade + 0.20 * consistencia_seed + 0.20 * learning_signal`

Onde:

- `rank_score`: derivado do score composto (`eer_final - 0.5*eer_drop + eer_stability`), invertido/normalizado para "maior é melhor".
- `estabilidade`: componente inversamente proporcional à variância/instabilidade da curva.
- `consistencia_seed`: repetibilidade do candidato em seeds diferentes (quando disponível).
- `learning_signal`: intensidade de aprendizado (`eer_drop` positivo e consistente).

**Uso prático:** apenas candidatos acima de limiar (ex.: percentil 60 do `C_conf` do próprio grupo) seguem para o fine.

---

## 2) O que ficou em aberto (diagnóstico crítico dos resultados)

## 2.1 Transferência coarse→fine ainda é parcial no LA-CDIP

Apesar de ganhos pontuais, a validação no regime fine ainda não é consistente em todos os grupos:

- Grupos coarse sobreviventes: **11**
- Grupos no fine: **10**
- Grupos comparáveis coarse vs fine: **10**
- Melhoraram no `best EER`: **3/10 (30%)**
- Melhoraram no `mean EER`: **2/10 (20%)**
- Delta médio (`fine_best - coarse_best`): **+0.01264**
- Delta mediano: **+0.01130**

**Conclusão:** há sinal positivo local, mas a transferência coarse→fine ainda é parcial.

---

## 2.2 Grupos com perda de patamar no regime fine

As maiores regressões por grupo foram:

- `circle_k3`: **0.007463 → 0.046198** (Δ +0.038736)
- `subcenter_arcface_k4`: **0.014925 → 0.043634** (Δ +0.028709)
- `subcenter_cosface_k1`: **0.010417 → 0.032400** (Δ +0.021983)
- `subcenter_cosface_k2`: **0.017879 → 0.034964** (Δ +0.017084)

**Leitura crítica:** o plano atual de validação fine não preservou patamar de coarse em todos os grupos; é necessário calibrar orçamento e ranges por trilha de grupo.

---

## 2.3 Só parte dos grupos confirmou o sinal do coarse

Ganhos reais no fine foram concentrados em poucos grupos:

- `subcenter_arcface_k3`: **0.007463 → 0.005128** (Δ -0.002334)
- `subcenter_cosface_k3`: **0.007463 → 0.005128** (Δ -0.002334)
- `subcenter_arcface_k2`: **0.017879 → 0.016363** (Δ -0.001517)

**Leitura crítica:** a transferência parece dependente de família (`subcenter_*`, especialmente `k=3`), não generalizável de forma uniforme.

---

## 2.4 Inércia parcialmente modelada, mas sem validação de impacto

Foi detectada inércia de `margin` em 3 grupos:

- `circle_k3`
- `subcenter_arcface_k3`
- `triplet_k3`

Porém faltou validar formalmente se fixar esses parâmetros melhorou robustez comparado a uma versão semi-fixa.

**Leitura crítica:** regra de inércia foi aplicada, mas não foi abladada (faltou experimento de confirmação causal).

---

## 2.5 Orçamento insuficiente para conclusão estatística

Com `run_cap=10` por sweep de fine, a variância dentro de alguns grupos ainda é alta (ex.: `subcenter_arcface_k2`, `subcenter_arcface_k1`, `subcenter_cosface_k3`).

**Leitura crítica:** com esse orçamento, o Bayes pode não explorar suficientemente o espaço refinado para estabilizar conclusão.

---

## 2.6 Falta de inferência estatística para confirmar transferência

A análise atual está baseada em melhor run e médias descritivas, mas sem:

- intervalos de confiança de ΔEER,
- teste de hipótese por grupo/família,
- correção para múltiplas comparações.

**Leitura crítica:** sem inferência estatística, a conclusão ainda é indiciária, não confirmatória.

---

## 2.7 O que foi confirmado e o que foi refutado no fine search

**Confirmado**
- O coarse encontra candidatos relevantes para validação.
- Em LA-CDIP, `subcenter_arcface_k3` e `subcenter_cosface_k3` sustentaram melhora no fine.
- O pipeline coarse→fine é válido como estratégia de descoberta + validação.

**Refutado (na configuração atual)**
- Que o fine melhora a maioria dos grupos automaticamente.
- Que best-run isolado é critério suficiente de decisão.
- Que o filtro coarse→fine atual, sozinho, resolve robustez de seleção.

---

## 3) O que faltou explorar para validar coarse→fine

1. **Orçamento progressivo no fine** (`run_cap=10,20,30`) para curva de convergência.
2. **Sensibilidade de range** (Narrow/Medium/Wide) para testar se os ranges do fine ficaram estreitos demais.
3. **Ablation de estratégia de fixação** (fixa vs semi-fixa) para grupos com regressão forte.
4. **Estratificação por família/loss+k** com políticas diferentes de busca (não tratar todos os grupos igual).
5. **Inferência estatística** (bootstrap + teste não paramétrico) para sustentar resultado científico.

---

## 4) Como usar os dados já processados para a Fase 2

## 4.1 Priorização de grupos (sem novo custo inicial)

Usar os CSVs já salvos para separar grupos em três trilhas:

- **Trilha A (ganho):** manter e aprofundar
  - `subcenter_arcface_k3`, `subcenter_cosface_k3`, `subcenter_arcface_k2`
- **Trilha B (neutra/leve piora):** retestar com mais orçamento
  - `subcenter_arcface_k1`, `triplet_k3`, `contrastive_k3`
- **Trilha C (regressão forte):** redesign de range/fixação
  - `circle_k3`, `subcenter_arcface_k4`, `subcenter_cosface_k1`, `subcenter_cosface_k2`

## 4.2 Reuso das estatísticas atuais

- `lacdip_coarse_survivors.csv`: baseline de referência por grupo.
- `lacdip_fine_all.csv`: variação intra-grupo e estabilidade no fine.
- `lacdip_inert_params.csv`: parâmetros candidatos à fixação/semi-fixação.

## 4.3 Métrica de decisão para continuar/descartar grupo

Para cada grupo:

- `Δbest = fine_best - coarse_best`
- `Δmean = fine_mean - coarse_mean`
- `std_fine`

Regras:

- **Promissor:** `Δbest < 0` e `Δmean <= 0`
- **Incerto:** `Δbest < 0` mas `Δmean > 0`
- **Não promissor:** `Δbest >= 0` e `Δmean > 0`

---

## 5) Plano de ação (Fase 2 LA-CDIP)

## Sprint 1 — Validação Top-5 em regime quase final + ablação do professor (2–3 dias)

### Objetivo
Responder objetivamente:

1. **Qual é a melhor configuração de cada perda Top-5 sob regime quase final?**
2. **A rede professor no curriculum learning ajuda nas 5 épocas finais?**

### Execução
- Selecionar Top-5 perdas do LA-CDIP para validação:
  - `subcenter_cosface`, `subcenter_arcface`, `triplet`, `contrastive`, `circle`.
- Para cada perda, escolher **a melhor configuração atual** (melhor `eer_final`; desempate por menor `eer_stability`).
- Rodar **10 épocas** com `max_steps_per_epoch` **ligeiramente maior** que o fine atual
  (baseline operacional: `100 → 140`).
- Executar duas variantes por perda (ablação pareada):
  - **Com professor nas últimas 5 épocas**: professor desligado nas épocas 1–5 e ativo nas épocas 6–10.
  - **Sem professor nas últimas 5 épocas**: professor desligado durante todas as 10 épocas.
- Manter os demais hiperparâmetros fixos por perda para isolar efeito do professor.

### Entregável
- Tabela principal:
  - `loss | config_base | EER_best_com_prof | EER_mean_com_prof | EER_best_sem_prof | EER_mean_sem_prof | delta_mean`.
- Ranking final das 5 perdas no regime quase final.
- Veredito por perda: **professor ajuda / neutro / atrapalha**.

### Gate de decisão
- Selecionar até **2 perdas finalistas** para Sprint 2 com base em `mean EER` + estabilidade.
- Se a variante “com professor” não gerar ganho consistente, manter versão sem professor como padrão para o treino final.

---

## Sprint 2 — Confirmado vs refutado no fine (2–3 dias)

### Objetivo
Responder objetivamente: **o que foi confirmado e o que foi refutado na Fase Fine?**

### Execução
- Rodar confirmação curta com 3 seeds para candidatos da Trilha A (`subcenter_arcface_k3`, `subcenter_cosface_k3`) e pelo menos 1 grupo regressivo da Trilha C.
- Calcular por grupo: `Δbest`, `Δmean`, `std_fine`, IC95% de `Δmean`.
- Classificar grupo em:
  - **Confirmado** (`Δbest<0` e `Δmean<=0`),
  - **Parcial** (`Δbest<0` e `Δmean>0`),
  - **Refutado** (`Δbest>=0`).

### Entregável
- Tabela oficial de status: **Confirmado vs Parcial vs Refutado** por grupo/loss.

### Gate de decisão
- Só grupos “Confirmados” seguem para candidato de treino final.

---

## Sprint 3 — Resolver grupos regressivos com menor custo (2–3 dias)

### Objetivo
Testar se regressões relevantes podem ser mitigadas sem reabrir sweep amplo.

### Execução
- Para grupos regressivos prioritários (Trilha C), testar desenho mínimo:
  - **Regra**: fixa vs semi-fixa,
  - **Range**: Narrow vs Medium.
- Manter orçamento moderado (`run_cap=20`) para comparação justa.

### Entregável
- Matriz curta: `grupo | regra | range | best | mean | std` + melhor combinação por grupo.

### Gate de decisão
- Se nenhum ajuste recuperar grupo regressivo, remover o grupo do pipeline final (economia de custo).

---

## Sprint 4 — Escolha final de perda e treino principal (1–2 dias)

### Objetivo
Fechar a pergunta prática: **qual configuração de perda segue para o novo treino final?**

### Execução
- Selecionar até 2 candidatas finais (prioridade: grupos Confirmados).
- Rodar comparação final 3-seeds entre candidatas.
- Escolher vencedora por critério composto:
  - prioridade 1: menor `mean EER`,
  - prioridade 2: menor `std`,
  - prioridade 3: melhor `best EER`.
- Rodar 1 treino final de maior orçamento na vencedora.

### Entregável
- Decisão final documentada: `loss vencedora`, `config`, `evidência de persistência`.
- Plano de transferência mínima para RVL com a receita vencedora.

---

## 6) Critérios de sucesso da Fase 2

A Fase 2 será considerada bem-sucedida se:

1. Pelo menos **50%** dos grupos comparáveis tiverem `Δbest < 0`.
2. Pelo menos **40%** dos grupos tiverem melhora também em média (`Δmean < 0`).
3. Grupos líderes (`subcenter_*_k3`) mantiverem ganho com variância controlada.
4. Regressões críticas (ex.: `circle_k3`) forem mitigadas ou excluídas com justificativa técnica.
5. Uma configuração final de perda seja escolhida com critério explícito de média+variância.

---

## 7) Decisão para o trabalho científico após a Fase 2

Se critérios forem atendidos:

- LA-CDIP vira benchmark principal do paper para validação metodológica.
- RVL-CDIP entra como experimento de transferência/generalização (fase posterior).

Se critérios não forem atendidos:

- reposicionar contribuição como “ganho condicional por família de loss”,
- focar em análise de por que o fine falha em determinados regimes.

---

## 8) Próxima ação imediata (executável)

1. Confirmar apenas duas candidatas no LA: `subcenter_arcface_k3` e `subcenter_cosface_k3`.
2. Rodar 3 seeds por candidata com config fixa e comparar `best/mean/std`.
3. Escolher 1 vencedora por média + variância.
4. Fazer 1 treino final com orçamento maior na vencedora e validar persistência do ganho.
5. Testar transferência mínima no RVL com a receita vencedora (sem novo sweep amplo).

Esse encadeamento evita custo alto prematuro e aumenta a chance de obter evidência sólida para publicação.
