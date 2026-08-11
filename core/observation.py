"""
Observacion: la estructura de datos que TODOS los benchmarks producen.

Es un superconjunto: el benchmark rellena todas las vistas de camara y estados
que tenga (`side_1`, `side_2`, `wrist`, ...), y cada modelo toma solo lo que
necesita por nombre. Asi benchmark y modelo quedan desacoplados: un benchmark
puede ofrecer 3 camaras y un modelo usar solo una, sin que ninguno conozca al
otro.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


class View(str, Enum):
    """Nombres canonicos de vistas de camara (evita errores de tipeo)."""
    FRONT = "front"
    SIDE_1 = "side_1"       # vista lateral 1
    SIDE_2 = "side_2"       # vista lateral 2
    WRIST = "wrist"         # vista del efector / accionador
    AGENT = "agentview"     # vista tercera-persona (convencion LIBERO)
    TOP = "top"


def _key(name):
    """Normaliza un nombre de vista (View o str) a su string canonico."""
    return name.value if isinstance(name, View) else str(name)


@dataclass
class Observation:
    """Observacion que entrega un benchmark en cada paso."""

    images: dict = field(default_factory=dict)   # {nombre_vista: array RGB (H,W,3)}
    state: dict = field(default_factory=dict)     # {nombre: array} (propriocepcion)
    instruction: Optional[str] = None             # instruccion en lenguaje natural
    extra: dict = field(default_factory=dict)     # cualquier dato adicional

    # -------- acceso a imagenes (el modelo elige lo que necesita) -------- #
    def image(self, name) -> np.ndarray:
        """Devuelve la vista `name`; error claro si el benchmark no la ofrece."""
        key = _key(name)
        if key not in self.images:
            raise KeyError(
                f"La observacion no tiene la vista '{key}'. "
                f"Disponibles: {list(self.images)}"
            )
        return self.images[key]

    def get_image(self, name, default=None):
        """Como `image`, pero devuelve `default` si la vista no existe."""
        return self.images.get(_key(name), default)

    def has_image(self, name) -> bool:
        return _key(name) in self.images

    def get_state(self, name, default=None):
        return self.state.get(_key(name), default)

    @property
    def view_names(self):
        return list(self.images)
