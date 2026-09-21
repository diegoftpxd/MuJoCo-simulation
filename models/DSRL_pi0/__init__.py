"""
Paquete DSRL-pi0. Expone `DSRLPi0Controller` (carga openpi pi0 + el actor SAC de
DSRL EN PROCESO; requiere JAX/openpi/jaxrl2 en el entorno). Se sirve por HTTP con
la capa generica `models.serving` (server.py --model dsrl_pi0); el benchmark lo
consume con `RemoteModel`, sin arrastrar JAX.

Import perezoso (PEP 562): importar el paquete no importa JAX/openpi.
"""

__all__ = ["DSRLPi0Controller"]


def __getattr__(name):
    if name == "DSRLPi0Controller":
        from .dsrl_pi0_controller import DSRLPi0Controller
        return DSRLPi0Controller
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
