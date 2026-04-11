#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

source .venv/bin/activate

exec ./scripts/ablations/run_sprint2_shard_nohome.sh \
  0 \
  1 \
  main \
  /mnt/data/cavl_runtime/main \
  42 \
  --sprint1-selection-mode top2-plus-contrastive \
  --bayes-runs-per-loss 5 \
  --run-baseline-off \
  --max-steps-per-epoch 50 \
  --candidate-pool-sizes 8 \
  --sprint1-allowed-run-ids fj38t4vd,493rr25s,stlbr5vu,rzvqr2g5,69u6if67 \
  "$@"
