"""Exporta a un JSON las estadisticas de des-normalizacion de acciones de pi0-LIBERO.

POR QUE: el CSV de denoising (output/pi0_denoise/*.csv) guarda el x_t del
flow-matching en espacio NORMALIZADO (antes de des-normalizar). Para leerlo en
unidades fisicas (cm / grados) hay que aplicar el des-normalizador afin del
checkpoint y luego la escala del controlador OSC_POSE de LIBERO. Este script
recupera ese des-normalizador y lo guarda; el notebook `flow_matching_analisis`
lo usa para escribir el CSV en cm (ver FlowMatchingAnalyzer.std_report_cm).

COMO: el postprocess de lerobot (self._postprocess del controlador) es afin por
dimension: post(x) = x * unnorm_std + unnorm_mean. Lo recuperamos SIN depender de
la API interna, probando con los vectores 0 y 1:
    unnorm_mean = post(0)
    unnorm_std  = post(1) - post(0)

DONDE CORRERLO: en el entorno conda `pizero` (con GPU y el checkpoint pi0-LIBERO),
el mismo que levanta el servidor del modelo. Es un one-shot:

    conda run -n pizero python scripts/export_action_stats.py

Variables opcionales:
    PI0_DEVICE      : "cuda" (default) o "cpu".
    PI0_STATS_JSON  : ruta de salida (default output/pi0_denoise/action_norm_stats.json).
"""

import json
import os

import numpy as np
import torch

from models.Pi_zero.Pi_zero_controller import PiZeroController

DEVICE = os.environ.get("PI0_DEVICE", "cuda")
OUT = os.environ.get("PI0_STATS_JSON", "output/pi0_denoise/action_norm_stats.json")

# Escala del controlador OSC_POSE de LIBERO (fija, ver env_wrapper.py / config del
# controlador): el comando en [-1,1] se mapea a delta fisico con estos maximos.
#   posicion (dx,dy,dz)      -> 0.05 m
#   orientacion (dr,dp,dy)   -> 0.5 rad
CONTROLLER_OUTPUT_MAX = [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]


def _probe(post, dim, device):
    """Recupera (mean, std) del des-normalizador afin probando post(0) y post(1)."""
    z = torch.zeros(1, dim, dtype=torch.float32, device=device)
    o = torch.ones(1, dim, dtype=torch.float32, device=device)
    mean = np.asarray(post(z).squeeze(0).float().cpu().numpy(), dtype=float)
    std = np.asarray(post(o).squeeze(0).float().cpu().numpy(), dtype=float) - mean
    return mean, std


def main():
    print("Cargando pi0-LIBERO (esto baja/carga el checkpoint)...")
    ctrl = PiZeroController(device=DEVICE)
    post = ctrl._postprocess
    cfg = ctrl.policy.config

    # Dimension de accion a probar: intenta la del config, luego candidatos comunes.
    candidates = [getattr(cfg, "max_action_dim", None),
                  getattr(cfg, "action_feature", None), 7, 8, 32]
    candidates = [int(d) for d in candidates if isinstance(d, int) and d > 0]

    mean = std = None
    for dim in candidates:
        try:
            mean, std = _probe(post, dim, DEVICE)
            print(f"OK: postprocess acepta dim={dim} -> salida de {len(mean)} dims.")
            break
        except Exception as e:                                    # noqa: BLE001
            print(f"  probe dim={dim} fallo: {type(e).__name__}: {e}")

    if mean is None:
        raise SystemExit(
            "No pude recuperar las stats via postprocess. Revisa la API de lerobot y "
            "comparte el error; ajusto el probe.")

    stats = {
        "model_id": ctrl.model_id,
        "action_dim": int(len(mean)),
        "unnorm_mean": mean.tolist(),
        "unnorm_std": std.tolist(),
        "controller_output_max": CONTROLLER_OUTPUT_MAX,
        "pos_dims": [0, 1, 2],
        "ori_dims": [3, 4, 5],
        "gripper_dim": 6,
        "note": ("post(x)=x*unnorm_std+unnorm_mean (comando OSC en ~[-1,1]); "
                 "pos_cm = comando*0.05*100 ; ori_grados = comando*0.5*180/pi"),
    }
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(stats, f, indent=2)

    print("\nStats guardadas en:", os.path.abspath(OUT))
    print("action_dim:", stats["action_dim"])
    print("unnorm_std[:7]:", np.round(std[:7], 6).tolist())
    print("unnorm_mean[:7]:", np.round(mean[:7], 6).tolist())


if __name__ == "__main__":
    main()
