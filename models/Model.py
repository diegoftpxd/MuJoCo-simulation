"""
Interfaz comun de los modelos (politicas).

Un modelo recibe `Observation` (elige la vista/estado que necesita) y produce
una LISTA de `Action` (un "chunk"). Mientras respete esta interfaz, se puede
intercambiar por cualquier otro (OpenVLA, Pi-zero, un modelo aleatorio, ...) sin
tocar el benchmark.
"""

from abc import ABC, abstractmethod
from typing import List

from core import Action, Observation


class Model(ABC):
    """Politica. Subclasear e implementar `act`."""

    @abstractmethod
    def act(self, observation: Observation) -> List[Action]:
        """
        Devuelve una LISTA de acciones (un "chunk") a partir de la observacion.

        El runner (`run_episode`) las ejecuta una a una y vuelve a pedir cuando
        se agota el chunk. Debe devolver al menos una accion. Un modelo de accion
        unica devuelve una lista de un solo elemento.
        """

    def reset(self):
        """Reinicia el estado interno del modelo entre episodios (opcional)."""
