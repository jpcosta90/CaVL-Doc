#!/bin/bash
# setup_finesearch_lacdip.sh
# Gerado automaticamente por analyze_and_build_finesearch.py

PROJECT="CaVL-Doc_LA-CDIP_FineSearch"
export WANDB_PROJECT=$PROJECT

echo "🔬 Registrando Fine Search Sweeps em: $PROJECT"
echo "--------------------------------------------------"

echo "Registrando circle_k3..."
OUT=$(wandb sweep scripts/sweeps/configs/lacdip/fine_search/fine_search_circle_k3.yaml --project $PROJECT 2>&1)
SWEEP_CIRCLE_K3=$(echo "$OUT" | grep "wandb agent" | sed 's/.*wandb agent //')

echo "Registrando contrastive_k3..."
OUT=$(wandb sweep scripts/sweeps/configs/lacdip/fine_search/fine_search_contrastive_k3.yaml --project $PROJECT 2>&1)
SWEEP_CONTRASTIVE_K3=$(echo "$OUT" | grep "wandb agent" | sed 's/.*wandb agent //')

echo "Registrando subcenter_arcface_k1..."
OUT=$(wandb sweep scripts/sweeps/configs/lacdip/fine_search/fine_search_subcenter_arcface_k1.yaml --project $PROJECT 2>&1)
SWEEP_SUBCENTER_ARCFACE_K1=$(echo "$OUT" | grep "wandb agent" | sed 's/.*wandb agent //')

echo "Registrando subcenter_arcface_k2..."
OUT=$(wandb sweep scripts/sweeps/configs/lacdip/fine_search/fine_search_subcenter_arcface_k2.yaml --project $PROJECT 2>&1)
SWEEP_SUBCENTER_ARCFACE_K2=$(echo "$OUT" | grep "wandb agent" | sed 's/.*wandb agent //')

echo "Registrando subcenter_arcface_k3..."
OUT=$(wandb sweep scripts/sweeps/configs/lacdip/fine_search/fine_search_subcenter_arcface_k3.yaml --project $PROJECT 2>&1)
SWEEP_SUBCENTER_ARCFACE_K3=$(echo "$OUT" | grep "wandb agent" | sed 's/.*wandb agent //')

echo "Registrando subcenter_arcface_k4..."
OUT=$(wandb sweep scripts/sweeps/configs/lacdip/fine_search/fine_search_subcenter_arcface_k4.yaml --project $PROJECT 2>&1)
SWEEP_SUBCENTER_ARCFACE_K4=$(echo "$OUT" | grep "wandb agent" | sed 's/.*wandb agent //')

echo "Registrando subcenter_cosface_k1..."
OUT=$(wandb sweep scripts/sweeps/configs/lacdip/fine_search/fine_search_subcenter_cosface_k1.yaml --project $PROJECT 2>&1)
SWEEP_SUBCENTER_COSFACE_K1=$(echo "$OUT" | grep "wandb agent" | sed 's/.*wandb agent //')

echo "Registrando subcenter_cosface_k2..."
OUT=$(wandb sweep scripts/sweeps/configs/lacdip/fine_search/fine_search_subcenter_cosface_k2.yaml --project $PROJECT 2>&1)
SWEEP_SUBCENTER_COSFACE_K2=$(echo "$OUT" | grep "wandb agent" | sed 's/.*wandb agent //')

echo "Registrando subcenter_cosface_k3..."
OUT=$(wandb sweep scripts/sweeps/configs/lacdip/fine_search/fine_search_subcenter_cosface_k3.yaml --project $PROJECT 2>&1)
SWEEP_SUBCENTER_COSFACE_K3=$(echo "$OUT" | grep "wandb agent" | sed 's/.*wandb agent //')

echo "Registrando triplet_k3..."
OUT=$(wandb sweep scripts/sweeps/configs/lacdip/fine_search/fine_search_triplet_k3.yaml --project $PROJECT 2>&1)
SWEEP_TRIPLET_K3=$(echo "$OUT" | grep "wandb agent" | sed 's/.*wandb agent //')


echo ""
echo "=================================================================="
echo "✅ USE OS COMANDOS ABAIXO PARA LANÇAR OS AGENTES:"
echo "=================================================================="
echo ""
echo "# circle_k3"
echo "wandb agent $SWEEP_CIRCLE_K3 --count 10"
echo ""
echo "# contrastive_k3"
echo "wandb agent $SWEEP_CONTRASTIVE_K3 --count 10"
echo ""
echo "# subcenter_arcface_k1"
echo "wandb agent $SWEEP_SUBCENTER_ARCFACE_K1 --count 10"
echo ""
echo "# subcenter_arcface_k2"
echo "wandb agent $SWEEP_SUBCENTER_ARCFACE_K2 --count 10"
echo ""
echo "# subcenter_arcface_k3"
echo "wandb agent $SWEEP_SUBCENTER_ARCFACE_K3 --count 10"
echo ""
echo "# subcenter_arcface_k4"
echo "wandb agent $SWEEP_SUBCENTER_ARCFACE_K4 --count 10"
echo ""
echo "# subcenter_cosface_k1"
echo "wandb agent $SWEEP_SUBCENTER_COSFACE_K1 --count 10"
echo ""
echo "# subcenter_cosface_k2"
echo "wandb agent $SWEEP_SUBCENTER_COSFACE_K2 --count 10"
echo ""
echo "# subcenter_cosface_k3"
echo "wandb agent $SWEEP_SUBCENTER_COSFACE_K3 --count 10"
echo ""
echo "# triplet_k3"
echo "wandb agent $SWEEP_TRIPLET_K3 --count 10"
echo ""
