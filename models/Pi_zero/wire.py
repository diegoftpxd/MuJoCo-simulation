"""
Serializacion minima para la frontera de proceso de pi0 (Opcion A).

El modelo (lerobot/pi0) corre en su propio proceso/entorno y el benchmark
(LIBERO) lo consume por red. Nada de torch/lerobot cruza el cable: SOLO arrays.
Por eso este modulo depende UNICAMENTE de numpy (+ stdlib), disponible tanto en
el entorno `pizero` como en el `openvla`. Asi ninguno de los dos entornos hereda
las dependencias del otro.

Formato del cable: un unico .npz comprimido con claves "mangladas":
    img::<vista>     -> imagen RGB (H,W,3) uint8
    st::<nombre>     -> vector de estado (propriocepcion)
    __instruction__  -> instruccion en lenguaje natural (bytes UTF-8 como uint8)

No se usa pickle (allow_pickle=False, el default) a proposito: el payload es
100% arrays numericos, seguro de deserializar y estable entre versiones.
"""

import io

import numpy as np

from core import Observation


# --------------------------- Observation <-> bytes -------------------------- #
def dump_observation(obs: Observation) -> bytes:
    """Serializa una Observation (imagenes + estado + instruccion) a bytes npz."""
    flat = {}
    for name, arr in obs.images.items():
        flat[f"img::{name}"] = np.asarray(arr)
    for name, arr in obs.state.items():
        flat[f"st::{name}"] = np.asarray(arr)
    # La instruccion viaja como sus bytes UTF-8 (array uint8): sin pickle.
    flat["__instruction__"] = np.frombuffer(
        (obs.instruction or "").encode("utf-8"), dtype=np.uint8)
    buf = io.BytesIO()
    np.savez_compressed(buf, **flat)
    return buf.getvalue()


def load_observation(blob: bytes) -> Observation:
    """Reconstruye una Observation a partir de los bytes npz."""
    images, state, instruction = {}, {}, None
    with np.load(io.BytesIO(blob)) as data:      # allow_pickle=False (default)
        for key in data.files:
            if key.startswith("img::"):
                images[key[len("img::"):]] = data[key]
            elif key.startswith("st::"):
                state[key[len("st::"):]] = data[key]
            elif key == "__instruction__":
                instruction = bytes(data[key]).decode("utf-8")
    return Observation(images=images, state=state, instruction=instruction)


# ------------------------------ chunk <-> bytes ----------------------------- #
def dump_chunk(chunk) -> bytes:
    """Serializa el chunk de acciones (H, 7) a bytes npz."""
    buf = io.BytesIO()
    np.savez_compressed(buf, chunk=np.asarray(chunk, dtype=np.float64))
    return buf.getvalue()


def load_chunk(blob: bytes) -> np.ndarray:
    """Reconstruye el chunk de acciones (H, 7)."""
    with np.load(io.BytesIO(blob)) as data:
        return data["chunk"]
