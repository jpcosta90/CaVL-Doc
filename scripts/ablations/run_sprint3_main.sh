#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/data/cavl_runtime/main}"
RUN_SUFFIX="${RUN_SUFFIX:-main}"
SEEDS="${SEEDS:-42}"

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

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$REPO_ROOT/src:$PYTHONPATH"
else
  export PYTHONPATH="$REPO_ROOT/src"
fi

exec "$PYTHON_BIN" scripts/ablations/run_sprint3_split5_staged_lacdip.py \
  --run-suffix "$RUN_SUFFIX" \
  --seeds "$SEEDS" \
  "$@"
