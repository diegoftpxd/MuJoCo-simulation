"""
Controlador de Pi-zero: esqueleto del wrapper que implementa `Model`.

Sigue el mismo patron que `OpenVLAController`: encapsula el modelo Pi-zero,
elige la vista que necesita de la `Observation`, predice, y devuelve un
`core.Action`. Rellena `_load` y `predict_action` con la carga/inferencia real.
"""

import numpy as np

from core import Action, View
from models.Model import Model


class PiZeroController(Model):
    """Wrapper de Pi-zero que implementa la interfaz `Model` (a completar)."""

    def __init__(self, view=View.AGENT):
        self.view = view
        self._load()

    # ------------------------------------------------------------------ #
    #  Interfaz comun (Model)
    # ------------------------------------------------------------------ #
    def act(self, observation) -> Action:
        image = observation.image(self.view)
        vec = self.predict_action(image, observation.instruction)
        return Action.from_cartesian(vec[:6], gripper=float(vec[6]), raw=vec)

    # ------------------------------------------------------------------ #
    #  Bajo nivel (a implementar con el modelo real Pi-zero)
    # ------------------------------------------------------------------ #
    def _load(self):
        raise NotImplementedError(
            "Carga el modelo Pi-zero aqui (imports perezosos de torch/etc.).")

    def predict_action(self, image, instruction):
        """Debe devolver un vector (7,): [dx,dy,dz,droll,dpitch,dyaw, pinza]."""
        raise NotImplementedError("Implementa la inferencia de Pi-zero.")
