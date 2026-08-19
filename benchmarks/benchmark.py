"""
Interfaz comun de los benchmarks (entornos de prueba).

Dos niveles de granularidad:
- **Escenario/tarea**: una instancia de `BenchMark` representa un escenario (en
  LIBERO, una tarea con su bddl). Cambiar de tarea = crear otra instancia; las
  tareas de una suite se enumeran aparte (p. ej. `LiberoController.tasks(...)`).
- **Episodio**: dentro de un escenario puede haber varias configuraciones
  iniciales. `reset(episode=i)` selecciona una; `num_episodes` dice cuantas hay.

Un benchmark produce `Observation` (multi-vista) y consume `Action`.
"""

from abc import ABC, abstractmethod

from core import Action, Observation, StepResult


class BenchMark(ABC):
    """Entorno de prueba. Subclasear e implementar reset/step/instruction."""

    @abstractmethod
    def reset(self, episode=None) -> Observation:
        """
        Reinicia el entorno y devuelve la observacion inicial.
        `episode` elige la configuracion inicial (None = la siguiente por defecto).
        """

    @abstractmethod
    def step(self, action: Action) -> StepResult:
        """Aplica `action` y devuelve (observacion, reward, done, info)."""

    @property
    @abstractmethod
    def instruction(self) -> str:
        """Instruccion en lenguaje natural de la tarea actual."""

    @property
    def num_episodes(self) -> int:
        """Cuantas configuraciones iniciales (episodios) ofrece el escenario."""
        return 1

    def close(self):
        """Libera recursos (opcional)."""
