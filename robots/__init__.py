"""
Paquete de robots.

Estructura:
    base.py       -> RobotArm (clase base generica) + MJCF_DIR
    control.py    -> CartesianController (IK, movimiento por Δpose)
    structures.py -> PoseDelta / Pose / Frame / CartesianAction
    factory.py    -> RobotFactory (registro y creacion por nombre) + make_robot
    tm5.py        -> familia TM5 (TM5_700, ...)
    panda.py      -> Franka Emika Panda

Los objetos del entorno (cubos, esferas, ...) viven en el paquete `objects`.

Al importar este paquete se cargan los modulos de cada robot, lo que dispara su
registro en `RobotFactory`. Para agregar un robot: crea su archivo con una clase
decorada con `@RobotFactory.register(...)` e impórtalo aqui abajo.
"""

from .base import MJCF_DIR, RobotArm
from .control import CartesianController
from .factory import RobotFactory, make_robot
from .structures import CartesianAction, Frame, Pose, PoseDelta

# Importar cada robot registra su clase en la fabrica.
from .tm5 import TM5, TM5_700
from .panda import Panda

__all__ = [
    "RobotArm",
    "CartesianController",
    "RobotFactory",
    "make_robot",
    "MJCF_DIR",
    # Estructuras de datos
    "PoseDelta",
    "Pose",
    "Frame",
    "CartesianAction",
    # Robots
    "TM5",
    "TM5_700",
    "Panda",
]
