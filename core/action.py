"""
Action: la estructura de datos que TODOS los modelos producen.

Tambien es un contenedor rico: el modelo rellena lo que sabe generar (p. ej.
un VLA da un delta cartesiano + pinza; un controlador articular daria
`joint_targets`), y cada benchmark toma solo los campos que su robot soporta.
Asi el modelo no necesita conocer el espacio de accion exacto del benchmark.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Action:
    """Accion que entrega un modelo. Los campos no usados quedan en None."""

    # Control cartesiano del efector: [dx, dy, dz, droll, dpitch, dyaw]
    cartesian_delta: Optional[np.ndarray] = None
    # Apertura de la pinza en la semantica del modelo (p. ej. [0,1]).
    gripper: Optional[float] = None
    # Control articular: objetivos de posicion de las articulaciones.
    joint_targets: Optional[np.ndarray] = None
    # Vector crudo tal cual salio del modelo (escape hatch para el benchmark).
    raw: Optional[np.ndarray] = None
    # Cualquier dato adicional.
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_cartesian(cls, delta6, gripper=None, raw=None):
        """Atajo: acción cartesiana (6 deltas) + pinza."""
        return cls(cartesian_delta=np.asarray(delta6, dtype=float),
                   gripper=None if gripper is None else float(gripper),
                   raw=None if raw is None else np.asarray(raw, dtype=float))
