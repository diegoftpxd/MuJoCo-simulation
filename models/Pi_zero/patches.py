"""
Punto UNICO para modificar el algoritmo interno de pi0 (flow-matching) SIN tocar
el paquete `lerobot` instalado en site-packages.

Aqui defines tus propias versiones de metodos del modelo pi0 y `apply_patches()`
los engancha (monkeypatch) sobre la politica ya cargada. Ventajas frente a editar
site-packages:
  - Tus cambios viven en el repo (versionados con git).
  - Sobreviven a reinstalar/reconstruir el entorno `pizero`.
  - Se activan/desactivan en un solo sitio (variable de entorno PI0_PATCH).

IMPORTANTE: este modulo se importa SOLO desde `PiZeroController._load` (entorno
`pizero`), por eso aqui SI se puede importar torch/lerobot directamente.

--------------------------------------------------------------------------------
COMO USARLO
  1. Edita `sample_actions` abajo. Su cuerpo es una COPIA del original de lerobot
     v0.4.0 (mas el logging a CSV); modifica lo que quieras (nº de pasos,
     integrador de Euler -> Heun/midpoint, guidance, etc.).
  2. El controlador llama a `apply_patches(self.policy)` al cargar el modelo, asi
     que basta relanzar el job (`scancel` + `sbatch`) para que tome tus cambios.
  3. Para volver al pi0 ORIGINAL sin borrar nada, lanza con  PI0_PATCH=0.

LOGGING DEL DENOISING (flow-matching)
  `sample_actions` escribe, en cada paso, el vector x_t que se va prediciendo
  hacia la accion, EMPEZANDO por el ruido inicial (iteracion 0). Cada fila trae:
    - call      : nº de llamado a sample_actions en este proceso (1 = primera vez).
    - batch     : indice de la muestra dentro del batch (normalmente 0).
    - iteration : 0 = ruido inicial; 1..num_steps = x_t tras cada paso de Euler.
    - time      : el tiempo t del flow-matching (1.0 en el ruido, ~0 al final).
    - num_steps : nº total de pasos de denoising de esta llamada.
    - v0..vK    : el vector x_t APLANADO (chunk_size * max_action_dim valores).
  Ruta del CSV: variable de entorno PI0_DENOISE_CSV (default:
  output/pi0_denoise/denoise_log.csv). Se reinicia (trunca) al arrancar cada
  servidor; para desactivar el log, pon PI0_DENOISE_CSV="" (vacio).
  NOTA: hoy se registra x_t (la estimacion de la accion). Si prefieres el "campo
  de velocidad" v_t, cambia el tensor que se pasa a `_log_denoise` dentro del bucle.
--------------------------------------------------------------------------------
"""

import csv
import os
import types
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

# Funcion de modulo de lerobot que usan tanto el original como esta copia.
from lerobot.policies.pi0.modeling_pi0 import make_att_2d_masks


# --------------------------------------------------------------------------- #
#  Restriccion de rango del gripper (pinza) en el flow-matching
# --------------------------------------------------------------------------- #
# Indice de la dimension de la PINZA dentro de cada paso de accion (el ultimo eje
# de x_t, de tamano max_action_dim). El vector de accion de pi0-LIBERO es
# [x, y, z, roll, pitch, yaw, gripper, ...padding], asi que el gripper es la
# dim 6 (mismo valor que "gripper_dim" en scripts/export_action_stats.py).
GRIPPER_DIM = 6
# Rango valido de la pinza. Un gripper fuera de [0, 1] es un estado "irreal":
# lo forzamos a este intervalo SOLO en el VECTOR INICIAL del flow-matching (el
# ruido de partida), para no arrancar el muestreo desde una magnitud de gripper
# sin sentido. El paso de Euler NO se toca: la dinamica del denoising queda igual.
GRIPPER_MIN, GRIPPER_MAX = 0.0, 1.0


def _clamp_gripper(x_t):
    """Fuerza in-place la dim del gripper de `x_t` (bsize, chunk, max_action_dim)
    al rango [GRIPPER_MIN, GRIPPER_MAX]. Solo toca la coordenada de la pinza; el
    resto del vector queda intacto."""
    if x_t.shape[-1] > GRIPPER_DIM:
        x_t[..., GRIPPER_DIM].clamp_(GRIPPER_MIN, GRIPPER_MAX)
    return x_t


# --------------------------------------------------------------------------- #
#  Logging del denoising a CSV
# --------------------------------------------------------------------------- #
_CALL_COUNT = {"n": 0}      # nº de llamados a sample_actions en este proceso
_FRESH = set()              # rutas de CSV ya truncadas en este proceso


def _next_call_index() -> int:
    _CALL_COUNT["n"] += 1
    return _CALL_COUNT["n"]


def _denoise_csv_path() -> str:
    # Ruta del CSV; override con PI0_DENOISE_CSV. Vacio ("") desactiva el log.
    return os.environ.get("PI0_DENOISE_CSV", "output/pi0_denoise/denoise_log.csv")


def _log_denoise(path, step, call_idx, iteration, t_value, num_steps, x_t, dist=0):
    """
    Agrega al CSV `path` una fila por muestra del batch con el vector `x_t`
    aplanado y los metadatos del paso. La PRIMERA escritura a cada ruta en este
    proceso la TRUNCA (empieza limpia) y escribe la cabecera; luego anexa.

    `step` = paso del entorno (lo fija el cliente con model.set_context(step=...);
    -1 si no se seteo). Permite agrupar corridas por estado del entorno.
    `dist` = etiqueta de la distribucion normal del ruido inicial (0 = sin marcar,
    1 o 2 en el experimento de dos gaussianas; model.set_context(dist=...)). Permite
    saber, en el analisis, de que cluster inicial vino cada corrida.
    """
    if not path:                                  # ruta vacia -> sin log
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    arr = x_t.detach().to(torch.float32).cpu().numpy()
    flat = arr.reshape(arr.shape[0], -1)          # (bsize, chunk_size*max_action_dim)

    first = path not in _FRESH
    mode = "w" if first else "a"
    write_header = first or (not p.exists()) or p.stat().st_size == 0
    if first:
        _FRESH.add(path)
        print(f"[pi0] logging denoise -> {p.resolve()}", flush=True)

    with p.open(mode, newline="") as f:
        w = csv.writer(f)
        if write_header:
            cols = ["step", "dist", "call", "batch", "iteration", "time", "num_steps"]
            cols += [f"v{i}" for i in range(flat.shape[1])]
            w.writerow(cols)
        for b in range(flat.shape[0]):
            w.writerow([int(step), int(dist), call_idx, b, iteration, float(t_value),
                        int(num_steps), *flat[b].tolist()])


# --------------------------------------------------------------------------- #
#  Flow-matching: copia editable de PI0Pytorch.sample_actions
#  `self` es el MODELO interno de pi0 (PiZeroController.policy.model), asi que
#  self.config, self.denoise_step, self.embed_prefix, self.sample_noise,
#  self._prepare_attention_masks_4d, self.paligemma_with_expert y
#  self.action_out_proj estan todos disponibles.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state,
                   noise=None, num_steps=None) -> Tensor:
    if num_steps is None:
        num_steps = self.config.num_inference_steps       # nº de pasos de denoising

    bsize = state.shape[0]
    device = state.device

    if noise is None:
        actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
        noise = self.sample_noise(actions_shape, device)
        # Ruido AMPLIADO (opcional): si el cliente envio una escala por dimension
        # con model.set_context(noise_scale=[...]), multiplicamos el N(0,1) por ella
        # -> cada dim cubre un area comparable a la magnitud de los datos. Sin escala,
        # ruido N(0,1) normal. La escala se guarda en el modelo (set_context).
        scale = getattr(self, "_pi0_noise_scale", None)
        if scale is not None:
            s = torch.as_tensor(np.asarray(scale, dtype="float32"), device=device)
            expected = int(actions_shape[1]) * int(actions_shape[2])
            if s.numel() != expected:
                raise ValueError(
                    f"noise_scale tiene {s.numel()} dims != esperado {expected} "
                    "(chunk_size*max_action_dim).")
            noise = noise * s.reshape(1, actions_shape[1], actions_shape[2])
        # Media DESPLAZADA (opcional): si el cliente envio una media por dimension
        # con model.set_context(noise_mean=[...]), la SUMAMOS al ruido -> el ruido
        # inicial pasa de N(0, scale) a N(mean, scale). Con dos medias distintas
        # (dos llamadas set_context con noise_mean diferentes) se generan dos
        # clusters de ruido inicial "super disjuntos". Sin media, ruido centrado.
        mean = getattr(self, "_pi0_noise_mean", None)
        if mean is not None:
            m = torch.as_tensor(np.asarray(mean, dtype="float32"), device=device)
            expected = int(actions_shape[1]) * int(actions_shape[2])
            if m.numel() != expected:
                raise ValueError(
                    f"noise_mean tiene {m.numel()} dims != esperado {expected} "
                    "(chunk_size*max_action_dim).")
            noise = noise + m.reshape(1, actions_shape[1], actions_shape[2])
    # Prefijo (imagenes + lenguaje): se computa UNA vez y se cachea en KV.
    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
        images, img_masks, lang_tokens, lang_masks)
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
    self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

    _, past_key_values = self.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )

    # ---- Bucle de integracion (Euler). Este es el corazon a modificar. ------ #
    dt = -1.0 / num_steps
    dt = torch.tensor(dt, dtype=torch.float32, device=device)

    x_t = noise
    # Gripper realista en el VECTOR INICIAL: por muy desplazado/escalado que quede
    # el ruido (noise_scale / noise_mean) o el noise que envie el cliente, la pinza
    # no puede PARTIR de un valor imposible. La forzamos a [0, 1] antes de empezar
    # el denoising (y el log de la iteracion 0 ya registra el valor corregido).
    _clamp_gripper(x_t)
    call_idx = _next_call_index()
    step = int(getattr(self, "_pi0_step", -1))         # paso del entorno (set_context)
    dist = int(getattr(self, "_pi0_dist", 0))          # distribucion del ruido (set_context)
    # Ruta del CSV: override por set_context(csv_path=...), si no la de PI0_DENOISE_CSV.
    csv_path = getattr(self, "_pi0_csv", None) or _denoise_csv_path()
    _log_denoise(csv_path, step, call_idx, 0, 1.0, num_steps, x_t, dist)   # iteracion 0 = ruido

    time = torch.tensor(1.0, dtype=torch.float32, device=device)
    iteration = 0
    while time >= -dt / 2:
        expanded_time = time.expand(bsize)
        v_t = self.denoise_step(state, prefix_pad_masks, past_key_values,
                                x_t, expanded_time)
        x_t = x_t + dt * v_t          # paso de Euler
        time += dt
        iteration += 1
        # Registra el x_t predicho tras este paso (tiempo t ya actualizado).
        _log_denoise(csv_path, step, call_idx, iteration, float(time), num_steps, x_t, dist)

    return x_t


# --------------------------------------------------------------------------- #
#  Enganche de los patches sobre la politica cargada.
# --------------------------------------------------------------------------- #
#  Registra aqui cada metodo que quieras sobreescribir: (funcion, "nombre").
_PATCHES = [
    (sample_actions, "sample_actions"),
    # (denoise_step, "denoise_step"),   # <- si algun dia quieres modificar el paso
]


def apply_patches(policy):
    """
    Engancha los metodos de `_PATCHES` sobre el modelo interno de la politica pi0.
    Idempotente. Controlado por la variable de entorno PI0_PATCH:
      - PI0_PATCH=1 (default): aplica los patches de este archivo.
      - PI0_PATCH=0:           no hace nada (usa el pi0 ORIGINAL de lerobot).
    Devuelve la misma `policy` (para encadenar).
    """
    if os.environ.get("PI0_PATCH", "1") == "0":
        print("[pi0] PI0_PATCH=0 -> sin monkeypatch (pi0 original)", flush=True)
        return policy
    model = policy.model                       # PI0Pytorch (el modelo interno)
    for fn, name in _PATCHES:
        setattr(model, name, types.MethodType(fn, model))
    print("[pi0] patches aplicados desde models/Pi_zero/patches.py: "
          + ", ".join(name for _, name in _PATCHES), flush=True)
    return policy
