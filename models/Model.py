"""
Interfaz comun de los modelos (politicas).

Un modelo recibe `Observation` (elige la vista/estado que necesita) y produce
una LISTA de `Action` (un "chunk"). Mientras respete esta interfaz, se puede
intercambiar por cualquier otro (OpenVLA, Pi-zero, un modelo aleatorio, ...) sin
tocar el benchmark.
"""

from abc import ABC, abstractmethod
from typing import List

from core import Action, Capabilities, Observation


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

    def requirements(self) -> Capabilities:
        """
        Vistas / estados / instruccion que este modelo NECESITA de la
        `Observation`. El runner las compara con lo que ofrece el benchmark
        (`BenchMark.capabilities()`) ANTES de correr, para fallar temprano y
        claro si algo falta (ver core/capabilities.py).

        Por defecto no requiere nada (retrocompatible). Cada modelo la
        sobreescribe declarando lo que consume; puede depender de su config
        (p. ej. la vista elegida), por eso es un metodo de instancia.
        """
        return Capabilities()
