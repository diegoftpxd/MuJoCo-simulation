#!/bin/bash
#SBATCH --job-name=export_stats        # exporta stats de des-normalizacion de pi0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=diego.toledo@uc.cl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40gb
#SBATCH --partition=ialab-low-unlimit
#SBATCH --gres=gpu:2080_ti:1           # 1 GPU: carga rapida y robusta del checkpoint
#SBATCH --qos=debug
#SBATCH --output=slurm/logs/%x.log
#SBATCH --time=0:30:00
#SBATCH --nodelist=scylla
pwd; hostname; date
set -euo pipefail

# --- Que hace ------------------------------------------------------------- #
# Corre scripts/export_action_stats.py, que carga pi0-LIBERO y guarda las stats
# de des-normalizacion de acciones (unnorm_mean/std) en un JSON. El notebook
# flow_matching_analisis las usa para escribir el CSV en cm/grados.
#
# NO necesita GPU para el calculo (es des-normalizacion afin), pero pedimos 1
# para que la carga del checkpoint de ~4B sea rapida y sin sustos de RAM en CPU.
# Si quieres correrlo en CPU: cambia PI0_DEVICE=cpu abajo y quita la linea --gres.

ENV=pizero                              # entorno conda con lerobot/torch/pi0
export HF_HOME="${HF_HOME:-$PWD/hf_cache}"   # cache de pesos (misma que el servidor)
export PI0_DEVICE="${PI0_DEVICE:-cuda}"      # "cuda" (default) o "cpu"

mkdir -p slurm/logs output/pi0_denoise

source ~/miniforge3/etc/profile.d/conda.sh
echo "Entorno: ${ENV} | device: ${PI0_DEVICE} | HF_HOME: ${HF_HOME}"

conda run --no-capture-output -n "${ENV}" \
    python scripts/export_action_stats.py

echo ""
echo "Listo. Revisa: output/pi0_denoise/action_norm_stats.json"
date
