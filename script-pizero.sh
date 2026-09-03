#!/bin/bash
#SBATCH --job-name=notebook-pizero     # Nombre del trabajo
#SBATCH --mail-type=END,FAIL           # Eventos al mail (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=diego.toledo@uc.cl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb                      # pi0 en fp16 son ~7GB; margen para carga
#SBATCH --partition=ialab
#SBATCH --gres=gpu:2                    # 2x RTX 2080 SUPER (8GB c/u); pi0 va justo en 8GB
#SBATCH --output=slurm/logs/%x_%j.log
#SBATCH --time=2:00:00
pwd; hostname; date

# Activa conda
source ~/miniforge3/etc/profile.d/conda.sh

# 1) Crear el entorno pizero SOLO si no existe (operacion de una vez).
if ! conda env list | grep -q "envs/pizero"; then
    echo "Creando entorno pizero..."
    conda env create -f environment-pizero.yml
fi
conda activate pizero

# 2) torch + torchvision para RTX 2080 SUPER (Turing, sm_75). torch 2.2 (<2.6)
#    no necesita el shim de torch.load.
pip install -q torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118

# 3) LIBERO editable (el repo clonado).
pip install -q -e benchmarks/libero/Libero-10-r

# 4) Jupyter + REGISTRO DEL KERNEL: esto es lo que hace que el env aparezca en
#    el menu de Jupyter (Kernel -> Change Kernel -> "Python (pizero)").
pip install -q ipykernel notebook
python -m ipykernel install --user --name pizero --display-name "Python (pizero)"

# 5) Sanidad de dependencias (no aborta el job).
pip check || echo "AVISO: pip check reporto conflictos (revisar arriba)."

# 6) Iniciar jupyter.
PORT=2849
which jupyter
echo "Iniciando servidor de notebooks"
echo ""
echo "Para conectarte, ejecuta en tu maquina local:"
echo "--------------------------------------------------------------------------------"
echo "  ssh -L localhost:8888:$(hostname):${PORT} kraken"
echo "--------------------------------------------------------------------------------"
echo "Luego abre http://localhost:8888 en tu navegador."
echo "El token de acceso aparece mas abajo en este mismo log."
echo ""
jupyter notebook --no-browser --ip="*" --port=${PORT}
echo "Trabajo $SLURM_JOBID finalizado"
date
