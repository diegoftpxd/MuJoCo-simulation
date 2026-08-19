#!/bin/bash
#SBATCH --job-name=notebook            # Nombre del trabajo
#SBATCH --mail-type=END,FAIL           # Enviar eventos al mail (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=diego.toledo@uc.cl             # El mail del usuario
#SBATCH --ntasks=1                     # Correr una sola tarea
#SBATCH --cpus-per-task=1              # Número de CPUs (threads) para el notebook
#SBATCH --mem=16gb                      # Memoria reservada para el trabajo
#SBATCH --partition=ialab              # Partición donde correr el trabajo
##SBATCH --nodelist=<node>              # Nodo donde correr el trabajo
#SBATCH --output=slurm/logs/%x_%j.log  # Nombre del output (%x=nombre del trabajo, %j=ID del trabajo)
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
pip install -q -r requirements-vla.txt
# LIBERO + robosuite (motor de simulación). requirements.txt fija las versiones
# compatibles (incluye robosuite); -e registra el paquete libero del repo.
pip install -q -r benchmarks/libero/Libero-10-r/requirements.txt
pip install -q -e benchmarks/libero/Libero-10-r

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
