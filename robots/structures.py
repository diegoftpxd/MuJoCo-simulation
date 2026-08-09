"""
Estructuras de datos (objetos de valor) para describir poses, incrementos de
pose y acciones cartesianas de forma legible.

En vez de pasar seis floats sueltos o arrays anonimos, el codigo usa tipos con
nombre: `PoseDelta(dz=-0.05)` se lee solo, y `CartesianAction.from_vector(v)`
deja claro que un vector de 7 numeros es "6 deltas + pinza".
"""

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple, Optional, Sequence

import numpy as np


class Frame(str, Enum):
    """Marco de referencia en el que se interpreta un incremento de pose."""
    TOOL = "tool"    # ejes del efector final (TCP)
    WORLD = "world"  # ejes del mundo


class Pose(NamedTuple):
    """
    Pose de un cuerpo en el mundo: posicion (3,) y rotacion (3x3).

    Es una tupla con nombre, asi que se puede desempaquetar
    (`position, rotation = pose`) o acceder por campo (`pose.position`).
    """
    position: np.ndarray
    rotation: np.ndarray


@dataclass(frozen=True)
class PoseDelta:
    """
    Incremento de pose del TCP: traslacion (m) y giro (rad, roll/pitch/yaw).

    Todos los campos valen 0 por defecto, para escribir p. ej.
    `PoseDelta(dz=-0.05)` o `PoseDelta(dyaw=0.3)`.
    """
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    droll: float = 0.0
    dpitch: float = 0.0
    dyaw: float = 0.0

    @property
    def translation(self) -> np.ndarray:
        """Parte de traslacion [dx, dy, dz]."""
        return np.array([self.dx, self.dy, self.dz], dtype=float)

    @property
    def rotation_rpy(self) -> np.ndarray:
        """Parte de rotacion [droll, dpitch, dyaw]."""
        return np.array([self.droll, self.dpitch, self.dyaw], dtype=float)

    def as_twist(self) -> np.ndarray:
        """Vector (6,): [dx, dy, dz, droll, dpitch, dyaw]."""
        return np.array([self.dx, self.dy, self.dz,
                         self.droll, self.dpitch, self.dyaw], dtype=float)

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> "PoseDelta":
        """Crea un PoseDelta desde una secuencia de 6 numeros."""
        v = np.asarray(values, dtype=float).reshape(6)
        return cls(*(float(x) for x in v))

    @classmethod
    def coerce(cls, value) -> "PoseDelta":
        """Devuelve un PoseDelta a partir de un PoseDelta o una secuencia."""
        return value if isinstance(value, cls) else cls.from_sequence(value)


@dataclass(frozen=True)
class CartesianAction:
    """
    Accion cartesiana: un incremento de pose mas el estado de la pinza.

    Es la accion tipica de un agente de RL o de un modelo tipo OpenVLA:
    6 deltas de pose + 1 valor de pinza.
    """
    delta: PoseDelta
    gripper_open: Optional[bool] = None

    @classmethod
    def from_vector(cls, values, gripper_threshold: float = 0.5) -> "CartesianAction":
        """
        Construye la accion desde un vector
        `[dx, dy, dz, droll, dpitch, dyaw, (pinza)]`.

        El 7º valor es opcional y se umbraliza: `> gripper_threshold` = abierta.
        Si no viene, la pinza se deja como esta (`None`).
        """
        v = np.asarray(values, dtype=float).reshape(-1)
        delta = PoseDelta.from_sequence(v[:6])
        gripper_open = bool(v[6] > gripper_threshold) if v.size >= 7 else None
        return cls(delta, gripper_open)
