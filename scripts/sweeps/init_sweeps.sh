#!/bin/bash
# scripts/sweeps/init_sweeps.sh
# Helper script to initialize WandB sweeps

# Usage function
usage() {
    echo "Usage: $0 [metric|geometric|circle|all] [dataset: lacdip | rvlcdip] [project_name]"
    echo "Example: $0 metric lacdip CaVL-Calibration-LACDIP"
    exit 1
}

TARGET=$1
DATASET=$2
PROJECT=${3:-"CaVL-Sweeps"}

if [ -z "$TARGET" ]; then
    usage
fi

if [ "$DATASET" != "lacdip" ] && [ "$DATASET" != "rvlcdip" ]; then
    echo "❌ Missing or invalid dataset. Must be 'lacdip' or 'rvlcdip'"
    usage
fi

echo "🚀 Setting WandB Project to: $PROJECT"
echo "📂 Dataset: $DATASET"
export WANDB_PROJECT=$PROJECT

# Wrapper to run wandb sweep
run_sweep() {
    CONFIG_FILE=$1
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "⚠️ Config file not found: $CONFIG_FILE"
        return
    fi
    
    echo "--------------------------------------------------------"
    echo "Initializing sweep for config: $CONFIG_FILE"
    wandb sweep $CONFIG_FILE
    echo "--------------------------------------------------------"
}

if [ "$TARGET" == "metric" ] || [ "$TARGET" == "all" ]; then
    run_sweep "scripts/sweeps/configs/$DATASET/sweep_metric.yaml"
fi

if [ "$TARGET" == "geometric" ] || [ "$TARGET" == "all" ]; then
    run_sweep "scripts/sweeps/configs/$DATASET/sweep_geometric.yaml"
fi

if [ "$TARGET" == "circle" ] || [ "$TARGET" == "all" ]; then
    run_sweep "scripts/sweeps/configs/$DATASET/sweep_circle.yaml"
fi

echo "✅ Done. Copy the 'wandb agent ...' commands from above to start your workers."
