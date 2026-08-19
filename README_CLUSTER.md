# Correr OpenVLA + LIBERO en el cluster

Este bundle contiene **todo el código necesario** para correr OpenVLA sobre
LIBERO-10 en el cluster. Incluye el framework propio (`core`, `benchmarks`,
`models`, `robots`) y el repo de **LIBERO vendorizado** en `benchmarks/libero/repo`
(sin `.git`, listo para `pip install -e`).

## 0. Contenido

```
core/                     estructuras comunes + bucle de ejecucion (run_episode, run_experiments)
benchmarks/               interfaz BenchMark + LiberoController + ejemplo
  libero/repo/            LIBERO vendorizado (robosuite, bddl, assets, init states)
models/                   interfaz Model + OpenVLAController (+ Random, Pi_zero)
robots/, objects/, MJCF/  simulacion propia (para smoke test sin GPU)
libero_test.py            entrypoint: OpenVLA sobre 3 tareas de LIBERO, graba video
openvla_experiments.ipynb notebook equivalente (tabla + tasa de exito + videos)
requirements-vla.txt      dependencias de GPU
run_openvla.slurm         plantilla de job (ajustala a tu cluster)
```

## 1. Crear el entorno (login node, con internet)

```bash
conda create -n openvla python=3.10 -y
conda activate openvla

# torch segun la CUDA del cluster (1080Ti/2080Ti -> cu118)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# LIBERO (robosuite, bddl, ...) desde el repo incluido
pip install -e benchmarks/libero/repo

# resto del stack OpenVLA + simulacion
pip install -r requirements-vla.txt
```

## 2. Pre-descargar los pesos (login node)

Los nodos de cómputo suelen estar **offline**. Baja los pesos en el login node
hacia un cache en scratch, y luego corre en modo offline:

```bash
export HF_HOME="${SCRATCH:-$HOME}/hf_cache"
```

## 3. Correr

Con GPU asignada (ver `run_openvla.slurm`):

```bash
export MUJOCO_GL=egl                 # render headless
export HF_HOME="${SCRATCH:-$HOME}/hf_cache"
export HF_HUB_OFFLINE=1              # ya descargados en el paso 2
python libero_test.py               # o abrir openvla_experiments.ipynb
```

Los videos quedan en `output/libero/` (o `output/experiments/` desde el notebook).

## Notas de GPU

- **4-bit** (`load_in_4bit=True`, por defecto) entra en **una** tarjeta de 11 GB
  (2080Ti recomendada; el 1080Ti es Pascal y va lento en fp16).
- Para **fp16 en 2 GPUs**: `#SBATCH --gres=gpu:2` y
  `OpenVLAController(load_in_4bit=False, device_map="auto")`.

## Cómo se expande el código

- **Nuevo benchmark**: subclase de `BenchMark` en `benchmarks/` que produzca
  `Observation` y consuma `Action`. No toca el modelo.
- **Nuevo modelo**: subclase de `Model` en `models/` con `act(obs) -> Action`.
  No toca el benchmark.
- `run_experiments(model, benchmarks, ...)` corre cualquier combinación
  modelo × lista-de-benchmarks; el notebook solo configura y muestra.
