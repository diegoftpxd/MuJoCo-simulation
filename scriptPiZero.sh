#!/bin/bash
#SBATCH --job-name=notebookPiZero      # pi0 (servidor) + LIBERO (notebook) en un nodo
#SBATCH --mail-type=END,FAIL           # Eventos al mail (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=diego.toledo@uc.cl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40gb                     # pi0 ~7GB + LIBERO/robosuite + jupyter
#SBATCH --partition=ialab              # particion con GPU (como script.sh)
#SBATCH --gres=gpu:2                   # compartidas: pi0 (inferencia) + EGL (render)
#SBATCH --output=slurm/logs/%x.log     # %x=nombre del trabajo, %j=ID del trabajo
#SBATCH --time=2:00:00                 # Tiempo limite del trabajo

pwd; hostname; date
set -euo pipefail

# ===========================================================================
# OPCION A: frontera de PROCESO. Dos entornos que NO comparten dependencias:
#
#   pizero  (MODEL_ENV) -> servidor de inferencia pi0  (lerobot, torch 2.7, numpy 2.x)
#   openvla (BENCH_ENV) -> benchmark LIBERO + notebook (robosuite 1.4.1, numpy<2)
#
# Se comunican por HTTP en localhost; por el cable solo cruzan arrays numpy
# (ver models/Pi_zero/wire.py). Asi lerobot y robosuite nunca chocan.
#
# Requisito: el entorno del benchmark (openvla) ya debe existir con LIBERO
# instalado (se crea con script.sh). El entorno del modelo (pizero) se crea aqui
# automaticamente la primera vez.
# ===========================================================================

MODEL_ENV=pizero                       # entorno del servidor pi0
BENCH_ENV=openvla                      # entorno del benchmark + notebook cliente
PORT_MODEL=9000                        # servidor pi0 (solo interno al nodo)
PORT_JUPYTER=2849                      # notebook (el que tuneleas por SSH)
export HF_HOME="${HF_HOME:-$PWD/hf_cache}"    # cache de pesos compartido entre envs

source ~/miniforge3/etc/profile.d/conda.sh
mkdir -p slurm/logs

# --- 1) Crear el entorno del MODELO solo si no existe (operacion de una vez) - #
if ! conda env list | grep -q "envs/${MODEL_ENV}"; then
    echo "Creando entorno ${MODEL_ENV} (servidor pi0)..."
    conda env create -f environment-pizero.yml
fi

# --- 2) Arrancar el SERVIDOR pi0 en background, en el entorno pizero -------- #
#     `conda run` lo ejecuta en pizero sin cambiar la activacion actual.
SERVER_LOG="slurm/logs/pizero_server_${SLURM_JOB_ID:-local}.log"
echo "Levantando servidor pi0 en el puerto ${PORT_MODEL} (log: ${SERVER_LOG})..."
HF_HOME="$HF_HOME" conda run --no-capture-output -n "${MODEL_ENV}" \
    python -m models.Pi_zero.server --host 127.0.0.1 --port "${PORT_MODEL}" --device cuda \
    > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
# Al terminar/cancelar el job, matar tambien el servidor.
trap 'echo "Deteniendo servidor pi0 (PID $SERVER_PID)"; kill $SERVER_PID 2>/dev/null || true' EXIT

# --- 3) Esperar a que el modelo termine de cargar (endpoint /health) ------- #
echo "Esperando a que pi0 cargue el checkpoint..."
for i in $(seq 1 120); do              # hasta ~10 min
    if curl -sf "http://127.0.0.1:${PORT_MODEL}/health" >/dev/null 2>&1; then
        echo "Servidor pi0 LISTO."
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: el servidor pi0 murio al cargar. Ultimas lineas de su log:"
        tail -n 40 "${SERVER_LOG}" || true
        exit 1
    fi
    sleep 5
done

# --- 4) Arrancar JUPYTER en el entorno del benchmark (openvla) -------------- #
#     En el notebook usa:  from models.Pi_zero import PiZeroClient
#                          model = PiZeroClient(url="http://localhost:9000")
conda activate "${BENCH_ENV}"
which jupyter
echo ""
echo "Para conectarte, ejecuta en tu maquina local:"
echo "--------------------------------------------------------------------------------"
echo "  ssh -L localhost:8888:$(hostname):${PORT_JUPYTER} kraken"
echo "--------------------------------------------------------------------------------"
echo "Luego abre http://localhost:8888 y corre pizero_experiments.ipynb"
echo "(kernel del entorno ${BENCH_ENV}). El token aparece mas abajo en este log."
echo ""
jupyter notebook --no-browser --ip="*" --port="${PORT_JUPYTER}"

echo "Trabajo ${SLURM_JOB_ID:-local} finalizado"
date
