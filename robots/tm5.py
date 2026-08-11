"""
Familia de brazos colaborativos Techman TM5 (modelos propios en `MJCF/`).
"""

import os

from .base import MJCF_DIR, RobotArm
from .factory import RobotFactory


class TM5(RobotArm):
    """
    Familia de brazos colaborativos Techman TM5 (6 GDL).

    Las variantes concretas (TM5-700, TM5-900, ...) solo definen `MODEL`, que
    es el nombre de la carpeta y del archivo dentro de `MJCF/`. La ubicacion del
    TCP y el keyframe de arranque son comunes a la familia.
    """

    MODEL = None          # p. ej. "tm5-700"  (subcarpeta y archivo en MJCF/)
    END_EFFECTOR_SITE = "tcp"       # sitio del efector final (punto de agarre de la pinza)
    HOME_KEY = "home"     # keyframe de arranque

    def __init__(self, objects=None):
        if self.MODEL is None:
            raise NotImplementedError(
                "Define el atributo MODEL en la subclase concreta de TM5."
            )
        xml = os.path.join(MJCF_DIR, self.MODEL, f"{self.MODEL}.xml")
        super().__init__(xml, end_effector_site=self.END_EFFECTOR_SITE,
                         home_key=self.HOME_KEY, objects=objects)


@RobotFactory.register("tm5-700")
class TM5_700(TM5):
    """Techman TM5-700 (alcance 700 mm)."""
    MODEL = "tm5-700"
