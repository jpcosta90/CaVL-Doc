#!/bin/bash
# scripts/sweeps/setup_rvlcdip_sweeps.sh

PROJECT="CaVL-Doc_RVL-CDIP_InternVL3-2B_Sweeps"
export WANDB_PROJECT=$PROJECT

echo "🚀 Initializing RVL-CDIP Sweeps on Project: $PROJECT"
echo "--------------------------------------------------"

# 1. ArcFace (Split K=1,2,3)
echo "Creating ArcFace (K=1) Sweep..."
OUT=$(wandb sweep scripts/optimization/coarse_search/configs/rvlcdip/sweep_arcface_k1.yaml 2>&1)
ARC1_AGENT=$(echo "$OUT" | grep "wandb agent" | sed 's/.*wandb agent //')

echo "Creating ArcFace (K=2) Sweep..."
OUT=$(wandb sweep scripts/optimization/coarse_search/configs/rvlcdip/sweep_arcface_k2.yaml 2>&1)
ARC2_AGENT=$(echo "$OUT" | grep "wandb agent" | sed 's/.*wandb agent //')

echo "Creating ArcFace (K=3) Sweep..."
OUT=$(wandb sweep scripts/optimization/coarse_search/configs/rvlcdip/sweep_arcface_k3.yaml 2>&1)
ARC3_AGENT=$(echo "$OUT" | grep "wandb agent" | sed 's/.*wandb agent //')

# 2. CosFace (Split K=1,2,3)
echo "Creating CosFace (K=1) Sweep..."
OUT=$(wandb sweep scripts/optimization/coarse_search/configs/rvlcdip/sweep_cosface_k1.yaml 2>&1)
COS1_AGENT=$(echo "$OUT" | grep "wandb agent" | sed 's/.*wandb agent //')

echo "Creating CosFace (K=2) Sweep..."
OUT=$(wandb sweep scripts/optimization/coarse_search/configs/rvlcdip/sweep_cosface_k2.yaml 2>&1)
COS2_AGENT=$(echo "$OUT" | grep "wandb agent" | sed 's/.*wandb agent //')

echo "Creating CosFace (K=3) Sweep..."
OUT=$(wandb sweep scripts/optimization/coarse_search/configs/rvlcdip/sweep_cosface_k3.yaml 2>&1)
COS3_AGENT=$(echo "$OUT" | grep "wandb agent" | sed 's/.*wandb agent //')

# 3. Circle (Moved up)
echo "Creating CircleFace Sweep..."
OUT=$(wandb sweep scripts/optimization/coarse_search/configs/rvlcdip/sweep_circle.yaml 2>&1)
CIRCLE_AGENT=$(echo "$OUT" | grep "wandb agent" | sed 's/.*wandb agent //')

# 4. Contrastive (Moved down - Already run)
echo "Creating Contrastive Sweep..."
OUT=$(wandb sweep scripts/optimization/coarse_search/configs/rvlcdip/sweep_contrastive.yaml 2>&1)
CONT_AGENT=$(echo "$OUT" | grep "wandb agent" | sed 's/.*wandb agent //')

# 5. Triplet (Moved down - Already run)
echo "Creating Triplet Sweep..."
OUT=$(wandb sweep scripts/optimization/coarse_search/configs/rvlcdip/sweep_triplet.yaml 2>&1)
TRIP_AGENT=$(echo "$OUT" | grep "wandb agent" | sed 's/.*wandb agent //')

echo ""
echo "=================================================================="
echo "✅ SWEEPS CRIADOS! Copie e rode os comandos abaixo:"
echo "=================================================================="
echo ""
echo "# ArcFace Variants"
echo "wandb agent $ARC1_AGENT --count 15"
echo "wandb agent $ARC2_AGENT --count 15"
echo "wandb agent $ARC3_AGENT --count 15"
echo ""
echo "# CosFace Variants"
echo "wandb agent $COS1_AGENT --count 15"
echo "wandb agent $COS2_AGENT --count 15"
echo "wandb agent $COS3_AGENT --count 15"
echo ""
echo "# Others (Circle First)"
echo "wandb agent $CIRCLE_AGENT --count 15"
echo ""
echo "# Already Run (Low Priority)"
echo "wandb agent $CONT_AGENT --count 15"
echo "wandb agent $TRIP_AGENT --count 15"
echo ""
