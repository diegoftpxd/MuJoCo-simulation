#!/bin/bash
# setup_dsrl.sh -- crea el entorno conda `dsrl` para servir DSRL-pi0 (Opcion A).
#
# CORRER EN EL LOGIN NODE de kraken (necesita internet: clona el repo e instala
# con pip; los nodos de computo normalmente no tienen red). Idempotente: se puede
# re-ejecutar (reutiliza el repo/env si ya existen).
#
# Uso:
#     bash scripts/setup_dsrl.sh
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

# --- Config ---------------------------------------------------------------- #
DSRL_REPO="${DSRL_REPO:-https://github.com/nakamotoo/dsrl_pi0.git}"
DSRL_DIR="${DSRL_DIR:-$HOME/dsrl_pi0}"
ENV_NAME="${ENV_NAME:-dsrl}"
JAX_SPEC="${JAX_SPEC:-jax[cuda12]}"

# Raiz del repo de simulacion (este script vive en <repo>/scripts/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_YML="$SIM_ROOT/requirements/environment-dsrl.yml"

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
if [ -d "$DSRL_DIR/.git" ]; then
    echo "[1/5] repo ya existe -> actualizando submodulos"
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
#     no desde PyPI. Instalamos las deps del repo ANTES que jax[cuda12] para que
#     el build con CUDA sea el que quede al final (paso 4).
echo "[3/5] instalando openpi (submodulo), jaxrl2 (repo) y requirements"
pip install -e "$DSRL_DIR/openpi"
pip install -e "$DSRL_DIR"
if [ -f "$DSRL_DIR/requirements.txt" ]; then
    pip install -r "$DSRL_DIR/requirements.txt"
fi

# --- 4) JAX con CUDA (ultimo, para fijar el build con GPU) ----------------- #
echo "[4/5] instalando $JAX_SPEC"
pip install --upgrade "$JAX_SPEC"

# --- 5) Verificacion ------------------------------------------------------- #
echo "[5/5] verificando imports"
python -c "import jax, openpi, jaxrl2; print('OK  jax', jax.__version__, '| devices:', jax.devices())"

echo ""
echo "Entorno '$ENV_NAME' listo."
echo "Congela cuando funcione:  conda run -n $ENV_NAME pip freeze > $SIM_ROOT/requirements/requirements-dsrl-lock.txt"
echo "Para servir el modelo:"
echo "    cd $SIM_ROOT && AGENT_CKPT=<dir_del_actor_SAC> sbatch scripts/experiment.sh   # MODEL=dsrl_pi0"
