"""
Interfaz comun de los modelos (politicas).

Un modelo recibe `Observation` (elige la vista/estado que necesita) y produce
`Action`. Mientras respete esta interfaz, se puede intercambiar por cualquier
otro (OpenVLA, Pi-zero, un modelo aleatorio, ...) sin tocar el benchmark.
"""

from abc import ABC, abstractmethod

from core import Action, Observation


class Model(ABC):
    """Politica. Subclasear e implementar `act`."""

    @abstractmethod
    def act(self, observation: Observation) -> Action:
        """Elige una accion a partir de la observacion."""

    def reset(self):
        """Reinicia el estado interno del modelo entre episodios (opcional)."""
