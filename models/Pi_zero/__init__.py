"""
Paquete pi0. Dos formas de usar el modelo:

  - PiZeroController : carga pi0 EN PROCESO (requiere lerobot/torch en el mismo
                       entorno). Es lo que usa el servidor.
  - PiZeroClient     : proxy por HTTP contra un servidor de inferencia (Opcion A).
                       Solo necesita numpy + stdlib; vive en el entorno del
                       benchmark, sin lerobot ni torch.

Los imports son PEREZOSOS (PEP 562): pedir `PiZeroClient` NO importa
`Pi_zero_controller` (que necesita lerobot), y viceversa. Asi cada entorno
importa solo lo que sus dependencias permiten.
"""

__all__ = ["PiZeroController", "PiZeroClient"]


def __getattr__(name):
    if name == "PiZeroController":
        from .Pi_zero_controller import PiZeroController
        return PiZeroController
    if name == "PiZeroClient":
        from .client import PiZeroClient
        return PiZeroClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
