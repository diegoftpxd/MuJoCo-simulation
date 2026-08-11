"""Esfera de radio `radius`."""

import mujoco

from .base import SceneObject


class Sphere(SceneObject):
    """Una esfera manipulable o fija."""

    def __init__(self, name="sphere", radius=0.03, **kwargs):
        super().__init__(name, **kwargs)
        self.radius = float(radius)

    def _add_geom(self, body):
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE,
                      size=[self.radius, 0.0, 0.0],
                      rgba=self.rgba, mass=self.mass)
