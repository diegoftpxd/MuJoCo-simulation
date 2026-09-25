#!/bin/bash
#SBATCH --job-name=train_dsrl          # entrena el actor SAC de DSRL-pi0 (LIBERO)
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=diego.toledo@uc.cl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40gb
#SBATCH --partition=ialab-low
#SBATCH --gres=gpu:1
#SBATCH --comment=Train_critic_of_dsrl
#SBATCH --output=slurm/logs/%x.log
#SBATCH --time=24:00:00                 
#SBATCH --nodelist=hydra
#SBATCH --qos=regular

# train_dsrl_libero.sh -- porta examples/scripts/run_libero.sh del repo dsrl_pi0 a
# un job SLURM de kraken. Entrena el actor SAC que DSRL usa como AGENT_CKPT para
# servir el modelo (ver scripts/experiment.sh, MODEL=dsrl_pi0).
#
# El entrenamiento vive en el repo dsrl_pi0 (usa su codigo: examples/launch_train_sim.py),
# corre en el entorno conda `dsrl` (ver scripts/setup_dsrl.sh). Este script solo lo
# envuelve: activa el env, exporta las variables del run_libero.sh original y lanza.
#
# Uso:
#     cd ~/MuJoCo-simulation && sbatch scripts/train_dsrl_libero.sh
#   sweep de semillas (cada uno en su GPU/job):
#     for s in 0 1 2; do SEED=$s sbatch --job-name=train_dsrl_s$s scripts/train_dsrl_libero.sh; done
#
# Variables opcionales (override por entorno):
#     DSRL_DIR   repo dsrl_pi0            (default: ~/dsrl_pi0)
#     ENV_NAME   entorno conda            (default: dsrl)
#     SEED       semilla                  (default: 0)
#     MAX_STEPS  pasos de entrenamiento   (default: 300000)
#     WANDB_MODE offline|online|disabled  (default: offline; sin login a W&B)

set -euo pipefail
pwd; hostname; date

# --- Config ---------------------------------------------------------------- #
DSRL_DIR="${DSRL_DIR:-$HOME/dsrl_pi0}"          # codigo del repo dsrl_pi0
# Donde van los DATOS pesados: checkpoints del actor + cache de la base pi0 (~10GB).
# Por defecto dentro del repo; si tu $HOME tiene cuota chica, apunta a scratch/proyecto:
#     DSRL_DATA=/mnt/scratch/$USER/dsrl sbatch scripts/train_dsrl_libero.sh
DSRL_DATA="${DSRL_DATA:-$DSRL_DIR}"
ENV_NAME="${ENV_NAME:-dsrl}"
SEED="${SEED:-0}"
MAX_STEPS="${MAX_STEPS:-300000}"
# OJO: --checkpoint_interval por defecto es -1 en el repo (= NO guarda checkpoints).
# Sin esto entrenas horas y no queda AGENT_CKPT. Lo forzamos (alineado con eval).
CKPT_INTERVAL="${CKPT_INTERVAL:-10000}"
export WANDB_MODE="${WANDB_MODE:-offline}"   # offline: no bloquea esperando login a W&B
proj_name="DSRL_pi0_Libero"
device_id=0                                   # bajo SLURM, la GPU asignada es el indice 0

mkdir -p slurm/logs
[ -d "$DSRL_DIR" ] || { echo "ERROR: no existe $DSRL_DIR (corre scripts/setup_dsrl.sh primero)."; exit 1; }

# --- Entorno conda `dsrl` -------------------------------------------------- #
source ~/miniforge3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"
echo "env: $ENV_NAME | python: $(which python)"

# --- Variables de entorno (del run_libero.sh original) --------------------- #
export DISPLAY=:0
export MUJOCO_GL=egl                     # render de LIBERO por GPU (EGL)
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=$device_id
export CUDA_VISIBLE_DEVICES=$device_id   # una sola GPU; evita que JAX pre-reserve otras
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # menos OOM (deja crecer la memoria de JAX)
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$DSRL_DATA/openpi}"   # cache de la base pi0 (openpi-assets)
export EXP="${EXP:-$DSRL_DATA/logs/$proj_name}"                    # <-- aqui quedan los checkpoints del actor
mkdir -p "$OPENPI_DATA_HOME" "$EXP"
echo "checkpoints -> $EXP"
echo "cache pi0   -> $OPENPI_DATA_HOME"

cd "$DSRL_DIR"

# LIBERO exige mujoco 3.3.1 (el run_libero.sh lo instala en caliente). Solo si falta:
if ! python -c "import mujoco,sys; sys.exit(0 if mujoco.__version__=='3.3.1' else 1)" 2>/dev/null; then
    echo "Instalando mujoco==3.3.1 (requerido por LIBERO)..."
    pip install mujoco==3.3.1
fi

echo "Lanzando entrenamiento DSRL (seed=$SEED, max_steps=$MAX_STEPS, wandb=$WANDB_MODE)"
echo "Checkpoints del actor -> $EXP"
echo "--------------------------------------------------------------------------------"

python3 examples/launch_train_sim.py \
    --algorithm pixel_sac \
    --env libero \
    --prefix dsrl_pi0_libero \
    --wandb_project "$proj_name" \
    --batch_size 256 \
    --discount 0.999 \
    --seed "$SEED" \
    --max_steps "$MAX_STEPS" \
    --eval_interval 10000 \
    --checkpoint_interval "$CKPT_INTERVAL" \
    --log_interval 500 \
    --eval_episodes 10 \
    --multi_grad_step 20 \
    --start_online_updates 500 \
    --resize_image 64 \
    --action_magnitude 1.0 \
    --query_freq 20 \
    --hidden_dims 128

echo "--------------------------------------------------------------------------------"
echo "Entrenamiento terminado. Usa el checkpoint como AGENT_CKPT para servir:"
echo "    cd ~/MuJoCo-simulation && sbatch --export=ALL,AGENT_CKPT=$EXP/... scripts/experiment.sh"
date
