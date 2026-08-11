"""Cubo: una caja de lado `2*size`."""

import mujoco

from .base import SceneObject


class Cube(SceneObject):
    """Un cubo (caja) manipulable o fijo."""

    def __init__(self, name="cube", size=0.025, **kwargs):
        super().__init__(name, **kwargs)
        self.size = float(size)

    def _add_geom(self, body):
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[self.size, self.size, self.size],
                      rgba=self.rgba, mass=self.mass)
