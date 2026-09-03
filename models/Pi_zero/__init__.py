"""
Paquete pi0. Expone `PiZeroController` (carga pi0 EN PROCESO; requiere
lerobot/torch en el entorno). Se sirve por HTTP con la capa generica
`models.serving` (server.py --model pi0); el benchmark lo consume con
`RemoteModel`, sin arrastrar lerobot.

Import perezoso (PEP 562): importar el paquete no importa lerobot.
"""

__all__ = ["PiZeroController"]


def __getattr__(name):
    if name == "PiZeroController":
        from .Pi_zero_controller import PiZeroController
        return PiZeroController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
