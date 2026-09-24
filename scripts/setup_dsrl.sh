#!/bin/bash
#SBATCH --job-name=setup_dsrl          # crea el entorno conda dsrl (Opcion A)
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=diego.toledo@uc.cl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=5gb
#SBATCH --partition=ialab-low
#SBATCH --qos=regular
#SBATCH --comment=Instala_dlrs
#SBATCH --output=slurm/logs/%x.log
#SBATCH --time=3:00:00

# setup_dsrl.sh -- crea el entorno conda `dsrl` para servir DSRL-pi0 (Opcion A).
#
# NECESITA INTERNET (clona el repo e instala con pip). Corre donde tu nodo tenga
# red: el LOGIN node, o un nodo de computo si tu cluster le da salida. Idempotente:
# se puede re-ejecutar (reutiliza el repo/env si ya existen).
#
# Formas de correrlo:
#   - Login node directo:  bash scripts/setup_dsrl.sh
#   - Batch (usa el #SBATCH de arriba):
#         mkdir -p slurm/logs && sbatch scripts/setup_dsrl.sh
#   - Interactivo con srun (srun NO lee los #SBATCH -> pasa los flags a mano):
#         srun --partition=ialab-low-unlimit --qos=debug --gres=gpu:2080_ti:1 \
#              --cpus-per-task=8 --mem=40gb --time=2:00:00 --nodelist=scylla \
#              --pty bash scripts/setup_dsrl.sh
#
# Variables opcionales (override por entorno):
#     DSRL_DIR   destino del clon de dsrl_pi0   (default: ~/dsrl_pi0)
#     ENV_NAME   nombre del entorno conda        (default: dsrl)
#     JAX_SPEC   build de JAX a instalar         (default: jax[cuda12])
#     DSRL_REPO  URL del repo                     (default: repo oficial)
#
# Al terminar deja el entorno `dsrl` listo; para servir el modelo:
#     cd <repo> && AGENT_CKPT=<ckpt> sbatch scripts/experiment.sh   (MODEL=dsrl_pi0)

set -euo pipefail
pwd; hostname; date
mkdir -p slurm/logs

# --- Config ---------------------------------------------------------------- #
DSRL_REPO="${DSRL_REPO:-https://github.com/nakamotoo/dsrl_pi0.git}"
DSRL_DIR="${DSRL_DIR:-$HOME/dsrl_pi0}"
ENV_NAME="${ENV_NAME:-dsrl}"
# openpi FIJA jax[cuda12]==0.5.0. NO subas de esa version: la ultima (0.10.x)
# arrastra numpy 2.x y rompe openpi/openpi-client (que exigen numpy<2).
JAX_SPEC="${JAX_SPEC:-jax[cuda12]==0.5.0}"

# Raiz del repo de simulacion. Bajo SLURM, `sbatch` copia el script a un spool
# (/tmp/slurmd/...), asi que BASH_SOURCE NO sirve; usamos, en orden:
#   1) SLURM_SUBMIT_DIR (dir desde donde lanzaste sbatch/srun = el repo),
#   2) el dir del script (ejecucion directa `bash scripts/setup_dsrl.sh`),
#   3) el cwd actual.
_has_yml() { [ -f "$1/requirements/environment-dsrl.yml" ]; }

SIM_ROOT=""
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && _has_yml "$SLURM_SUBMIT_DIR"; then
    SIM_ROOT="$SLURM_SUBMIT_DIR"
else
    _sd="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || true)"
    if [ -n "$_sd" ] && _has_yml "$(dirname "$_sd")"; then
        SIM_ROOT="$(dirname "$_sd")"
    elif _has_yml "$PWD"; then
        SIM_ROOT="$PWD"
    fi
fi
ENV_YML="${SIM_ROOT:-?}/requirements/environment-dsrl.yml"

echo "== setup_dsrl =="
echo "  repo DSRL : $DSRL_DIR"
echo "  env conda : $ENV_NAME"
echo "  env yml   : $ENV_YML"
echo "  jax       : $JAX_SPEC"
echo ""

# --- 0) conda disponible --------------------------------------------------- #
for c in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3"; do
    if [ -f "$c/etc/profile.d/conda.sh" ]; then
        source "$c/etc/profile.d/conda.sh"
        break
    fi
done
command -v conda >/dev/null || { echo "ERROR: no encuentro conda (revisa tu instalacion)."; exit 1; }
[ -f "$ENV_YML" ] || { echo "ERROR: no existe $ENV_YML (¿corres desde el repo correcto?)."; exit 1; }

# --- 1) Clonar dsrl_pi0 CON submodulos (openpi + LIBERO) ------------------- #
#     Los submodulos estan declarados con URL SSH (git@github.com:...) en
#     .gitmodules, pero el nodo no tiene clave SSH hacia GitHub. Reescribimos
#     SSH -> HTTPS para todas las operaciones de git (idempotente).
git config --global url."https://github.com/".insteadOf "git@github.com:"

if [ -d "$DSRL_DIR/.git" ]; then
    echo "[1/5] repo ya existe -> sincronizando y actualizando submodulos"
    git -C "$DSRL_DIR" submodule sync --recursive
    git -C "$DSRL_DIR" submodule update --init --recursive
else
    echo "[1/5] clonando $DSRL_REPO -> $DSRL_DIR"
    git clone --recurse-submodules "$DSRL_REPO" "$DSRL_DIR"
fi

# --- 2) Crear (o reutilizar) el entorno conda ------------------------------ #
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[2/5] env '$ENV_NAME' ya existe -> lo reutilizo"
else
    echo "[2/5] creando env '$ENV_NAME' desde el yml"
    conda env create -f "$ENV_YML"
fi
conda activate "$ENV_NAME"
echo "      python: $(which python)"

# --- 3) openpi (FORK del submodulo) + jaxrl2 (repo) + deps ----------------- #
#     IMPORTANTE: openpi va desde el submodulo (su Policy.infer acepta noise=),
#     no desde PyPI.
echo "[3/5] instalando openpi (submodulo), jaxrl2 (repo) y requirements"
pip install -e "$DSRL_DIR/openpi"
pip install -e "$DSRL_DIR"
if [ -f "$DSRL_DIR/requirements.txt" ]; then
    pip install -r "$DSRL_DIR/requirements.txt"
fi

# --- 4) Reconciliar las versiones que openpi FIJA -------------------------- #
#     Los pasos anteriores (requirements/jaxrl2) pueden subir jax/numpy y romper
#     los pines de openpi. Los reinstalamos EXACTOS AL FINAL, para que queden:
#       jax[cuda12]==0.5.0  (con CUDA)   numpy<2   pillow>=11
echo "[4/5] fijando versiones de openpi ($JAX_SPEC, numpy<2, pillow>=11)"
pip install "$JAX_SPEC" "numpy>=1.26,<2.0" "pillow>=11.0.0"

# --- 5) Verificacion ------------------------------------------------------- #
#     Importa las 3 libs. Si este nodo NO tiene GPU (instalar no la necesita),
#     jax cae a CPU y avisa "CUDA_ERROR_UNKNOWN": es ESPERADO aqui; la GPU se usa
#     al SERVIR (scripts/experiment.sh pide la GPU). Lo que importa es que los
#     imports funcionen y que jax quede en la version 0.5.0.
echo "[5/5] verificando imports (CPU es normal si este nodo no tiene GPU)"
python -c "import jax, openpi, jaxrl2; print('OK  jax', jax.__version__, '| devices:', jax.devices())"

echo ""
echo "Entorno '$ENV_NAME' listo."
echo "Congela cuando funcione:  conda run -n $ENV_NAME pip freeze > $SIM_ROOT/requirements/requirements-dsrl-lock.txt"
echo "Para servir el modelo:"
echo "    cd $SIM_ROOT && AGENT_CKPT=<dir_del_actor_SAC> sbatch scripts/experiment.sh   # MODEL=dsrl_pi0"
