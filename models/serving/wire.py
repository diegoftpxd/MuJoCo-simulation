"""
Serializacion minima para la frontera de proceso modelo <-> benchmark.

Capa GENERICA (no sabe de pi0 ni de OpenVLA): por el cable solo cruzan
`Observation` (benchmark -> modelo) y `List[Action]` (modelo -> benchmark),
ambos serializados con numpy. Por eso este modulo depende UNICAMENTE de numpy
(+ stdlib), disponible en todos los entornos. Asi ningun entorno hereda las
dependencias de otro.

No se usa pickle (allow_pickle=False, el default): el payload son arrays
numericos (+ un poco de JSON para `Action.extra`), seguro y estable entre
versiones.
"""

import io
import json

import numpy as np

from core import Action, Observation


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


# ----------------------------- Action <-> bytes ----------------------------- #
def _json_default(o):
    """Permite serializar arrays/escalares numpy dentro de Action.extra."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(f"no serializable a JSON: {type(o)}")


def dump_actions(actions) -> bytes:
    """
    Serializa una lista de `Action` (el "chunk" que devuelve `Model.act`) a npz.

    Es GENERICA: guarda solo los campos presentes de cada accion
    (cartesian_delta, gripper, joint_targets, raw, extra), asi sirve tanto para
    un modelo de accion unica (OpenVLA) como de horizonte (pi0), sin que el
    servidor tenga que saber la forma del vector.
    """
    flat = {"__n__": np.array([len(actions)], dtype=np.int64)}
    for i, a in enumerate(actions):
        if a.cartesian_delta is not None:
            flat[f"{i}::cd"] = np.asarray(a.cartesian_delta, dtype=np.float64)
        if a.gripper is not None:
            flat[f"{i}::gr"] = np.array([a.gripper], dtype=np.float64)
        if a.joint_targets is not None:
            flat[f"{i}::jt"] = np.asarray(a.joint_targets, dtype=np.float64)
        if a.raw is not None:
            flat[f"{i}::rw"] = np.asarray(a.raw, dtype=np.float64)
        if a.extra:
            payload = json.dumps(a.extra, default=_json_default).encode("utf-8")
            flat[f"{i}::ex"] = np.frombuffer(payload, dtype=np.uint8)
    buf = io.BytesIO()
    np.savez_compressed(buf, **flat)
    return buf.getvalue()


def load_actions(blob: bytes) -> list:
    """Reconstruye la lista de `Action`."""
    with np.load(io.BytesIO(blob)) as data:
        keys = set(data.files)
        n = int(data["__n__"][0])
        out = []
        for i in range(n):
            cd = data[f"{i}::cd"] if f"{i}::cd" in keys else None
            gr = float(data[f"{i}::gr"][0]) if f"{i}::gr" in keys else None
            jt = data[f"{i}::jt"] if f"{i}::jt" in keys else None
            rw = data[f"{i}::rw"] if f"{i}::rw" in keys else None
            ex = (json.loads(bytes(data[f"{i}::ex"]).decode("utf-8"))
                  if f"{i}::ex" in keys else {})
            out.append(Action(cartesian_delta=cd, gripper=gr,
                              joint_targets=jt, raw=rw, extra=ex))
        return out
