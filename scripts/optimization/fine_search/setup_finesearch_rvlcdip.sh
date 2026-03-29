#!/bin/bash
# setup_finesearch_rvlcdip.sh
# Gerado automaticamente por analyze_and_build_finesearch.py

PROJECT="CaVL-Doc_RVL-CDIP_FineSearch"
export WANDB_PROJECT=$PROJECT

echo "🔬 Registrando Fine Search Sweeps em: $PROJECT"
echo "--------------------------------------------------"

echo "Registrando contrastive_k3..."
OUT=$(wandb sweep scripts/optimization/coarse_search/configs/rvlcdip/fine_search/fine_search_contrastive_k3.yaml --project $PROJECT 2>&1)
SWEEP_CONTRASTIVE_K3=$(echo "$OUT" | grep "wandb agent" | sed 's/.*wandb agent //')

echo "Registrando subcenter_arcface_k1..."
OUT=$(wandb sweep scripts/optimization/coarse_search/configs/rvlcdip/fine_search/fine_search_subcenter_arcface_k1.yaml --project $PROJECT 2>&1)
SWEEP_SUBCENTER_ARCFACE_K1=$(echo "$OUT" | grep "wandb agent" | sed 's/.*wandb agent //')

echo "Registrando subcenter_cosface_k1..."
OUT=$(wandb sweep scripts/optimization/coarse_search/configs/rvlcdip/fine_search/fine_search_subcenter_cosface_k1.yaml --project $PROJECT 2>&1)
SWEEP_SUBCENTER_COSFACE_K1=$(echo "$OUT" | grep "wandb agent" | sed 's/.*wandb agent //')

echo "Registrando subcenter_cosface_k3..."
OUT=$(wandb sweep scripts/optimization/coarse_search/configs/rvlcdip/fine_search/fine_search_subcenter_cosface_k3.yaml --project $PROJECT 2>&1)
SWEEP_SUBCENTER_COSFACE_K3=$(echo "$OUT" | grep "wandb agent" | sed 's/.*wandb agent //')

echo "Registrando triplet_k3..."
OUT=$(wandb sweep scripts/optimization/coarse_search/configs/rvlcdip/fine_search/fine_search_triplet_k3.yaml --project $PROJECT 2>&1)
SWEEP_TRIPLET_K3=$(echo "$OUT" | grep "wandb agent" | sed 's/.*wandb agent //')


echo ""
echo "=================================================================="
echo "✅ USE OS COMANDOS ABAIXO PARA LANÇAR OS AGENTES:"
echo "=================================================================="
echo ""
echo "# contrastive_k3"
echo "wandb agent $SWEEP_CONTRASTIVE_K3 --count 10"
echo ""
echo "# subcenter_arcface_k1"
echo "wandb agent $SWEEP_SUBCENTER_ARCFACE_K1 --count 10"
echo ""
echo "# subcenter_cosface_k1"
echo "wandb agent $SWEEP_SUBCENTER_COSFACE_K1 --count 10"
echo ""
echo "# subcenter_cosface_k3"
echo "wandb agent $SWEEP_SUBCENTER_COSFACE_K3 --count 10"
echo ""
echo "# triplet_k3"
echo "wandb agent $SWEEP_TRIPLET_K3 --count 10"
echo ""
