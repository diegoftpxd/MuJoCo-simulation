"""
Interfaz comun de los benchmarks (entornos de prueba).

Un benchmark produce `Observation` (multi-vista) y consume `Action`. Mientras
respete esta interfaz, se puede intercambiar por cualquier otro sin tocar el
modelo ni el runner.
"""

from abc import ABC, abstractmethod

from core import Action, Observation, StepResult


class BenchMark(ABC):
    """Entorno de prueba. Subclasear e implementar reset/step/instruction."""

    @abstractmethod
    def reset(self) -> Observation:
        """Reinicia el entorno y devuelve la observacion inicial."""

    @abstractmethod
    def step(self, action: Action) -> StepResult:
        """Aplica `action` y devuelve (observacion, reward, done, info)."""

    @property
    @abstractmethod
    def instruction(self) -> str:
        """Instruccion en lenguaje natural de la tarea actual."""

    def close(self):
        """Libera recursos (opcional)."""
