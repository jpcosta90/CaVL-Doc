# GitHub Project Sync

Sincronização entre `docs/tasks.yaml` e GitHub Projects.

## Setup

### 1. Obter o número do projeto

No seu GitHub Project (view em cualquier formato), a URL é:
```
https://github.com/users/<username>/projects/<project-number>
```

Ou via CLI:
```bash
gh project list --owner joaopaulo
```

### 2. Instalar dependências

```bash
pip install pyyaml
```

### 3. Autenticar com GitHub (se needed)

```bash
gh auth login
```

## Uso

### Dry-run (visualizar mudanças sem fazer)

```bash
python scripts/project/sync_tasks_to_github_project.py \
    --project-number 123 \
    --dry-run
```

### Sincronizar (criar issues e adicionar ao project)

```bash
python scripts/project/sync_tasks_to_github_project.py \
    --project-number 123
```

## Fluxo de trabalho

1. **Editar `docs/tasks.yaml`** — adicionar/remover/atualizar tarefas
2. **Rodar script** — sincroniza com GitHub
3. **Movimentar no Project** — arrasta cards entre colunas
4. **Fechar issue** — ao terminar tarefa (opcionalmente: com `git commit -m "fixes #123"`)

## Estrutura do YAML

```yaml
sprints:
  sprint_2:
    phase: 6
    title: "Sprint 2 — Teacher Sweep"
    start_date: "2026-04-13"
    end_date: "2026-04-14"
    milestone: "Sprint 2"
    tasks:
      T-06-01:
        title: "Executar sweep Bayes..."
        description: "..."
        status: "in-progress"
        priority: "high"
        labels: ["phase:6", "topic:teacher"]
        research_question: "q1"
```

## Notas

- Issues são criadas uma vez; rodar de novo não duplica
- Labels são adicionadas automaticamente
- Status local em `docs/tasks.yaml` é independente (você atualiza manualmente ou via script)
- Para sincronização **bidirecional** (ler status do GitHub back), é um passo futuro

## Troubleshooting

Se tiver erro de autenticação:
```bash
gh auth refresh --scopes project
```

Se não conseguir criar issues no projeto:
- Verificar se tem permissão de escrita no repo
- Verificar se o `project_number` está correto
- Conferir se o projeto está vinculado ao repo (não é organização)
