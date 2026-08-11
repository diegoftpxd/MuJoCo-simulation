"""
Controlador aleatorio: el wrapper mas simple que implementa `Model`.

Elige deltas cartesianos pequeños al azar. Sirve para probar la arquitectura
(benchmark + runner + grabado) SIN GPU ni pesos, y como plantilla de un modelo.
"""

import numpy as np

from core import Action
from models.Model import Model


class RandomController(Model):
    """Politica de prueba que implementa la interfaz `Model`."""

    def __init__(self, scale=0.02, seed=0):
        self.scale = scale
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.rng = np.random.default_rng(self.seed)

    def act(self, observation) -> Action:
        delta = self.rng.uniform(-1.0, 1.0, size=6) * self.scale
        delta[3:] *= 0.5   # rotaciones más suaves
        return Action.from_cartesian(delta, gripper=1.0)
