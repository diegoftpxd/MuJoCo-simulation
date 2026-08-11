"""
Objetos que se pueden agregar al entorno (cubos, esferas, ...).

Estructura (una clase por archivo):
    base.py    -> SceneObject (clase base abstracta)
    cube.py    -> Cube
    sphere.py  -> Sphere

Para agregar un tipo nuevo: crea `objects/<tipo>.py` con una subclase de
`SceneObject` que implemente `_add_geom`, e impórtala aquí abajo.

Uso:
    from robots import RobotFactory
    from objects import Cube
    robot = RobotFactory.create("panda", objects=[Cube(position=(0.5, 0, 0.05))])
"""

from .base import SceneObject
from .cube import Cube
from .sphere import Sphere

__all__ = ["SceneObject", "Cube", "Sphere"]
