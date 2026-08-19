#!/bin/bash
#SBATCH --job-name=run_openVLA
#SBATCH -t 0-01:00                    # tiempo maximo en el cluster (D-HH:MM)
#SBATCH -o c_job.out                 # STDOUT
#SBATCH -e c_job.err                 # STDERR
#SBATCH --mail-type=END,FAIL         # notificacion cuando el trabajo termine o falle
#SBATCH --mail-user=diego.toledo@uc.cl    # mail donde mandar las notificaciones
#SBATCH --chdir=/home/diegoftpxd/slurm_test      # direccion del directorio de trabajo
#
#SBATCH --nodes 1                    # numero de nodos a usar
#SBATCH --cpus-per-task=1            # numero de cpus (threads) por trabajo (proceso)
#SBATCH --partition=ialab            # partición donde correrá tu trabajo (proceso)

set -euo pipefail
mkdir -p logs

# --- entorno (ejemplo con conda; adapta a venv/module si corresponde) ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openvla

# --- render headless + cache de modelos + modo offline en el nodo ---
export MUJOCO_GL=egl                  # MuJoCo/robosuite sin pantalla
export HF_HOME="${SCRATCH:-$HOME}/hf_cache"   # pesos en scratch compartido
export HF_HUB_OFFLINE=1               # ya descargados en el login node
export TOKENIZERS_PARALLELISM=false

cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv    # log de la GPU asignada

python libero_test.py

# -------------------------------------------------------------------------
# Para 2 GPUs en fp16 (precision completa, ~8GB por tarjeta en 2x 2080Ti):
#   - cambia arriba:  #SBATCH --gres=gpu:2
#   - y en libero_test.py usa OpenVLAController(load_in_4bit=False,
#     device_map="auto")  (requiere el ajuste de device_map en el controller).
# -------------------------------------------------------------------------
