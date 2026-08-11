"""
Franka Emika Panda, del repositorio MuJoCo Menagerie clonado en `MJCF/`.
"""

import os

from .base import MJCF_DIR, RobotArm
from .factory import RobotFactory


@RobotFactory.register("panda")
class Panda(RobotArm):
    """
    Franka Emika Panda (7 GDL) del repositorio MuJoCo Menagerie.

    Su modelo no trae un site de TCP, asi que se usa el cuerpo `hand` de la
    pinza con un offset a lo largo de su eje de aproximacion (~0.1034 m, la
    posicion tipica del "hand_tcp" del Panda). La pinza se acciona por tendon.
    """

    XML = os.path.join(MJCF_DIR, "panda", "scene.xml")

    def __init__(self, objects=None):
        super().__init__(self.XML, end_effector_body="hand",
                         end_effector_offset=(0, 0, 0.1034), home_key="home",
                         objects=objects)
