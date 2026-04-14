# Integração com GitHub Projects

## Visão geral do sistema

```
docs/tasks.yaml (fonte)
        ↓
        [Script Python]
        sync_tasks_to_github_project.py
        ↓
GitHub Project (view operacional)
        ↓
[DIA A DIA: Movimentar cards no projeto]
        ↓
[AO TERMINAR: Fechar issue]
        ↓
[OPCIONAL: Atualizar status em tasks.yaml]
```

## Estrutura de arquivos

```
CaVL-Doc/
├── docs/
│   ├── cronograma_fases_pesquisa.md     (cronograma macro)
│   └── tasks.yaml                        (fonte de tarefas)
├── scripts/project/
│   ├── sync_tasks_to_github_project.py  (script de sincronização)
│   ├── README.md                         (instruções)
│   └── INTEGRATION.md                    (este arquivo)
```

## 🔄 Scripts de Sincronização

### Sincronizar tasks.yaml → GitHub Project

```bash
cd /home/joaopaulo/Projects/CaVL-Doc
python scripts/project/sync_tasks_to_github_project.py \
    --project-number 1 \
    --repo jpcosta90/CaVL-Doc
```

**O que faz:**
- Lê `docs/tasks.yaml`
- Cria/atualiza issues no repositório
- Adiciona items ao GitHub Project 1
- Popula campos: title, description, status, labels, dates, milestone

### Sincronizar GitHub Project → tasks.yaml

```bash
cd /home/joaopaulo/Projects/CaVL-Doc
python scripts/project/sync_project_to_tasks.py \
    --project-number 1 \
    --repo jpcosta90/CaVL-Doc
```

**O que faz:**
- Lê issues atuais do repositório
- Atualiza `docs/tasks.yaml` com status real do GitHub
- Sincroniza labels, datas e outras métricas

### ⚡ Script completo de sincronização (bidirecional)

```bash
#!/bin/bash
# scripts/project/sync_all.sh

set -e

PROJECT_NUMBER=${1:-1}
REPO=${2:-"jpcosta90/CaVL-Doc"}

echo "🔄 Sincronizando GitHub Project com tasks.yaml..."
echo ""

# 1️⃣ Sincronizar tasks.yaml → GitHub Project
echo "📤 Pushando tasks.yaml para GitHub Project..."
python scripts/project/sync_tasks_to_github_project.py \
    --project-number "$PROJECT_NUMBER" \
    --repo "$REPO"

echo "✅ tasks.yaml → GitHub Project concluído"
echo ""

# 2️⃣ Sincronizar GitHub Project → tasks.yaml
echo "📥 Puxando status do GitHub Project..."
python scripts/project/sync_project_to_tasks.py \
    --project-number "$PROJECT_NUMBER" \
    --repo "$REPO"

echo "✅ GitHub Project → tasks.yaml concluído"
echo ""
echo "🎉 Sincronização bidirecional completa!"
```

**Usar:**
```bash
bash scripts/project/sync_all.sh 1 jpcosta90/CaVL-Doc
```

---

## Fluxo de trabalho recomendado

### 1️⃣ Setup inicial

```bash
# Descobrir seu project number
gh project list --owner joaopaulo

# Editar docs/tasks.yaml se necessário
# vim docs/tasks.yaml

# Sincronizar (push inicial)
python scripts/project/sync_tasks_to_github_project.py \
    --project-number 1 \
    --repo jpcosta90/CaVL-Doc
```

### 2️⃣ Durante a execução

- **Abrir GitHub Project** → Roadmap view
- **Colunas esperadas**: To do, In Progress, Done
- **Movimentar cards** conforme trabalha
- **Comentar em issues** quando tem updates

### 3️⃣ Ao terminar tarefa

Opção A (recomendado):
```bash
git commit -m "Concluído T-06-02: consolidar melhor config teacher"
git commit -m "fixes #123"  # fecha a issue automaticamente
```

Opção B (manual):
```bash
gh issue close 123
```

### 4️⃣ Atualizar cronograma

Quando termina uma sprint inteira, atualizar o status em:
- `docs/cronograma_fases_pesquisa.md`
- `docs/tasks.yaml` (status local)

## Exemplo: Sprint 2 em execução

**tasks.yaml:**
```yaml
sprints:
  sprint_2:
    start_date: "2026-04-13"
    end_date: "2026-04-14"
    tasks:
      T-06-01:
        title: "Executar sweep Bayes..."
        status: "in-progress"  # você atualiza conforme trabalha
        labels: ["phase:6", "topic:teacher", "status:doing"]
```

**GitHub Project (Roadmap):**
- Coluna "Sprint 2"
  - Card: [T-06-01] Executar sweep Bayes...
    - Status: In Progress
    - Issue #123
    - Labels: phase:6, topic:teacher, status:doing

**Seu terminal:**
```bash
# Terminei a tarefa
git commit -m "Concluído sweep Bayes com 5 trials"
git commit -m "fixes #123"

# Isso fecha a issue automaticamente
```

## Campos importantes no tasks.yaml

| Campo | Uso | Exemplo |
|-------|-----|---------|
| `task_id` | ID único da tarefa | `T-06-01` |
| `title` | Título curto | "Executar sweep Bayes..." |
| `description` | Detalhes, aceita Markdown | Ver exemplo acima |
| `priority` | Nível | `high`, `medium`, `low` |
| `labels` | Tags para o Project | `["phase:6", "topic:teacher"]` |
| `status` | Estado local | `todo`, `in-progress`, `done` |
| `research_question` | Vinculado a questão qual | `q1`, `q1,q2` |
| `evidence` | Que artefato prova conclusão | "W&B + CSV de runs" |

## Sincronização bidirecional (futuro)

Depois, podemos implementar um script que lê de volta do GitHub:

```bash
python scripts/project/sync_github_project_to_tasks.py \
    --project-number <NUMBER>
```

Isso atualizaria `docs/tasks.yaml` com o status real do GitHub.

## Troubleshooting

### "Erro: gh CLI not installed"
```bash
# No Ubuntu/Debian
sudo apt install gh

# No macOS
brew install gh
```

### "Erro: not authenticated"
```bash
gh auth login --scopes project,repo,issues
```

### "Projeto não encontrado"
```bash
# Verificar o número correto
gh project list --owner joaopaulo

# Ou descobrir pela URL
# https://github.com/users/joaopaulo/projects/123 → 123 é o number
```

### "Issues duplicadas"
O script checa se já existe antes de criar. Se não funcionar:
```bash
gh issue list --repo joaopaulo/CaVL-Doc --search "T-06-01"
```

## Links úteis

- [GitHub CLI documentation](https://cli.github.com/)
- [GitHub Projects documentation](https://docs.github.com/en/issues/planning-and-tracking-with-projects)
- [gh project command reference](https://cli.github.com/manual/gh_project)
