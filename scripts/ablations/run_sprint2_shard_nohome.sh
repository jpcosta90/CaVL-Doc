#!/usr/bin/env bash
set -euo pipefail

# Uso:
# scripts/ablations/run_sprint2_shard_nohome.sh <shard_index> <num_shards> <run_suffix> <runtime_root> [seeds]
# Exemplo (server main):
# scripts/ablations/run_sprint2_shard_nohome.sh 0 2 main /mnt/data/cavl_runtime/main 42
# Exemplo (server unb):
# scripts/ablations/run_sprint2_shard_nohome.sh 1 2 unb /mnt/data/cavl_runtime/unb 42
#
# Este launcher já aplica por padrão a política confirmada para base da Sprint 2:
# - fonte: W&B (não CSV local)
# - últimas 7 runs
# - somente prof_last5_off
# - losses únicas (deduplicadas)

SHARD_INDEX="${1:-0}"
NUM_SHARDS="${2:-2}"
RUN_SUFFIX="${3:-main}"
RUNTIME_ROOT="${4:-/mnt/data/cavl_runtime/main}"
SEEDS="${5:-42}"
EXTRA_ARGS=("${@:6}")

mkdir -p "$RUNTIME_ROOT"/{tmp,xdg/cache,xdg/config,xdg/data,wandb/runs,wandb/cache,wandb/config,torch,mpl,pip,pycache}

export TMPDIR="$RUNTIME_ROOT/tmp"
export XDG_CACHE_HOME="$RUNTIME_ROOT/xdg/cache"
export XDG_CONFIG_HOME="$RUNTIME_ROOT/xdg/config"
export XDG_DATA_HOME="$RUNTIME_ROOT/xdg/data"
export WANDB_DIR="$RUNTIME_ROOT/wandb/runs"
export WANDB_CACHE_DIR="$RUNTIME_ROOT/wandb/cache"
export WANDB_CONFIG_DIR="$RUNTIME_ROOT/wandb/config"
export HF_HOME="$HOME/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HOME/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface/hub"
export TORCH_HOME="$RUNTIME_ROOT/torch"
export MPLCONFIGDIR="$RUNTIME_ROOT/mpl"
export PIP_CACHE_DIR="$RUNTIME_ROOT/pip"
export PYTHONPYCACHEPREFIX="$RUNTIME_ROOT/pycache"
export HISTFILE=/dev/null

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN}"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  PYTHON_BIN=""
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERRO] Python não encontrado em: $PYTHON_BIN"
  echo "Defina PYTHON_BIN explicitamente ou crie o ambiente em: $REPO_ROOT/.venv"
  exit 1
fi

echo "[INFO] Usando Python: $PYTHON_BIN"

cd "$REPO_ROOT"

# Garante que o pacote local (src/cavl_doc) seja resolvido mesmo sem install -e .
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$REPO_ROOT/src:$PYTHONPATH"
else
  export PYTHONPATH="$REPO_ROOT/src"
fi

exec "$PYTHON_BIN" scripts/ablations/run_sprint2_teacher_sweep_lacdip.py \
  --sprint1-source wandb \
  --sprint1-only-last-n 7 \
  --sprint1-require-professor-mode off \
  --sprint1-selection-mode all-unique \
  --seeds "$SEEDS" \
  --num-shards "$NUM_SHARDS" \
  --shard-index "$SHARD_INDEX" \
  --run-suffix "$RUN_SUFFIX" \
  --runtime-root "$RUNTIME_ROOT" \
  "${EXTRA_ARGS[@]}"
