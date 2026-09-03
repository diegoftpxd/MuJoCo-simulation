#!/bin/bash
#SBATCH --job-name=experiment          # servidor de UN modelo + notebook del benchmark
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=diego.toledo@uc.cl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40gb
#SBATCH --partition=ialab
#SBATCH --gres=gpu:2
#SBATCH --output=slurm/logs/%x.log
#SBATCH --time=2:00:00

pwd; hostname; date
set -euo pipefail

# --- ELIGE el modelo a servir: pi0 | openvla | random ---------------------- #
MODEL=pi0

# --- Mapa modelo -> entorno conda / puerto / chequeo de dependencias ------- #
case "$MODEL" in
    pi0)     MODEL_ENV=pizero;  PORT_MODEL=9000; CHECK="import lerobot, torch, numpy" ;;
    openvla) MODEL_ENV=openvla; PORT_MODEL=9001; CHECK="import torch, transformers, numpy" ;;
    random)  MODEL_ENV=openvla; PORT_MODEL=9002; CHECK="import numpy" ;;
    *) echo "MODEL desconocido: $MODEL (usa pi0|openvla|random)"; exit 1 ;;
esac

# Entorno del BENCHMARK (LIBERO + jupyter). NO necesita torch/lerobot/transformers;
# el notebook solo importa RemoteModel (numpy + stdlib). Hoy reutiliza el env que
# ya corre LIBERO; si creas uno dedicado sin deps de modelos, cambialo aqui.
BENCH_ENV=openvla
PORT_JUPYTER=2849
export HF_HOME="${HF_HOME:-$PWD/hf_cache}"    # cache de pesos compartido

source ~/miniforge3/etc/profile.d/conda.sh
mkdir -p slurm/logs

# --- 1) Verificar que el entorno del modelo esta SANO ---------------------- #
#     (pip necesita internet: crea/reconstruye los entornos en el LOGIN node.)
echo "Verificando entorno '${MODEL_ENV}' para el modelo '${MODEL}'..."
if ! conda run -n "${MODEL_ENV}" python -c "${CHECK}" 2>/dev/null; then
    echo ""
    echo "ERROR: el entorno '${MODEL_ENV}' no tiene las dependencias del modelo"
    echo "       '${MODEL}' ( ${CHECK} )."
    echo "Reconstruyelo en el LOGIN node de kraken (tiene internet):"
    if [ "$MODEL_ENV" = "pizero" ]; then
        echo "    conda env remove -n pizero -y"
        echo "    conda env create -f requirements/environment-pizero.yml"
    else
        echo "    # (env de OpenVLA/LIBERO; ver scripts/script.sh)"
    fi
    echo "Luego vuelve a lanzar: sbatch scripts/scriptExperiment.sh"
    exit 1
fi

# --- 2) Arrancar el SERVIDOR del modelo en background, en su entorno -------- #
SERVER_LOG="slurm/logs/server_${MODEL}_${SLURM_JOB_ID:-local}.log"
echo "Levantando servidor '${MODEL}' en el puerto ${PORT_MODEL} (log: ${SERVER_LOG})..."
HF_HOME="$HF_HOME" conda run --no-capture-output -n "${MODEL_ENV}" \
    python -m models.serving.server --model "${MODEL}" \
        --host 127.0.0.1 --port "${PORT_MODEL}" --device cuda \
    > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
trap 'echo "Deteniendo servidor ${MODEL} (PID $SERVER_PID)"; kill $SERVER_PID 2>/dev/null || true' EXIT

# --- 3) Esperar a que el modelo termine de cargar (endpoint /health) ------- #
echo "Esperando a que '${MODEL}' cargue el checkpoint..."
for i in $(seq 1 120); do              # hasta ~10 min
    if curl -sf "http://127.0.0.1:${PORT_MODEL}/health" >/dev/null 2>&1; then
        echo "Servidor '${MODEL}' LISTO en http://localhost:${PORT_MODEL}"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: el servidor '${MODEL}' murio al cargar. Ultimas lineas:"
        tail -n 40 "${SERVER_LOG}" || true
        exit 1
    fi
    sleep 5
done

# --- 4) Arrancar JUPYTER en el entorno del benchmark ----------------------- #
#     En el notebook:  from models.serving import RemoteModel
#                      model = RemoteModel(url="http://localhost:${PORT_MODEL}")
conda activate "${BENCH_ENV}"
which jupyter
echo ""
echo "Modelo '${MODEL}' servido en http://localhost:${PORT_MODEL}"
echo "Para conectarte, ejecuta en tu maquina local:"
echo "--------------------------------------------------------------------------------"
echo "  ssh -L localhost:8888:$(hostname):${PORT_JUPYTER} kraken"
echo "--------------------------------------------------------------------------------"
echo "Luego abre http://localhost:8888 y corre el notebook del modelo elegido."
echo "El token aparece mas abajo en este log."
echo ""
jupyter notebook --no-browser --ip="*" --port="${PORT_JUPYTER}"

echo "Trabajo ${SLURM_JOB_ID:-local} finalizado"
date
