#!/bin/bash
#SBATCH --job-name=notebook            # Nombre del trabajo
#SBATCH --mail-type=END,FAIL           # Enviar eventos al mail (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=diego.toledo@uc.cl             # El mail del usuario
#SBATCH --ntasks=1                     # Correr una sola tarea
#SBATCH --cpus-per-task=5              # Número de CPUs (threads) para el notebook
#SBATCH --mem=20gb                      # Memoria reservada para el trabajo
#SBATCH --partition=ialab              # Partición donde correr el trabajo
#SBATCH --nodelist=hydra              # Nodo donde correr el trabajo
#SBATCH --output=slurm/logs/%x.log  # Nombre del output (%x=nombre del trabajo, %j=ID del trabajo)
#SBATCH --time=1:00:00                 # Tiempo limite del trabajo. Importante definirlo para no
                                       # mantener recursos ocupados si olvidas cerrar el servidor
#SBATCH --gres=gpu:2                  # Solicitar una GPU de ser necesario (descomentar si se necesita)
pwd; hostname; date

# Activa tu entorno de conda
source ~/miniforge3/etc/profile.d/conda.sh
conda activate openvla

# Instalar dependencias (modo silencioso -q: no llena el log con "Requirement
# already satisfied" cuando la dependencia ya está instalada; solo muestra
# warnings y errores).
#
# ORDEN IMPORTANTE: LIBERO fija un transformers viejo, incompatible con el
# codigo remoto de OpenVLA. Por eso se instala PRIMERO LIBERO y el stack VLA va
# AL FINAL: asi los pines de requirements-vla.txt (transformers 4.40.1, etc.)
# tienen la ultima palabra y no quedan degradados.

# 1) torch + torchvision segun la GPU del cluster. Build cu130 (CUDA 13) para
#    GPUs nuevas (Blackwell). Ajusta el index-url si tu CUDA es otra.
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 2) LIBERO + robosuite (motor de simulación). requirements.txt trae robosuite;
#    -e registra el paquete libero del repo.
pip install -q -r benchmarks/libero/Libero-10-r/requirements.txt
pip install -q -e benchmarks/libero/Libero-10-r

# 3) Stack VLA (OpenVLA) — al final, para que sus versiones fijadas ganen.
pip install -q -r requirements-vla.txt

# 4) Sanidad: reporta conflictos de dependencias declaradas (no aborta el job).
pip check || echo "AVISO: pip check reporto conflictos (revisar arriba)."

# Iniciar jupyter.
PORT=2849
which jupyter
echo "Iniciando servidor de notebooks"
echo ""
echo "Para conectarte, ejecuta en tu máquina local:"
echo "--------------------------------------------------------------------------------"
echo "  ssh -L localhost:8888:$(hostname):${PORT} kraken"
echo "--------------------------------------------------------------------------------"
echo "Luego abre http://localhost:8888 en tu navegador."
echo "El token de acceso aparece más abajo en este mismo log."
echo ""
jupyter notebook --no-browser --ip="*" --port=${PORT}
echo "Trabajo $SLURM_JOBID finalizado"
date
