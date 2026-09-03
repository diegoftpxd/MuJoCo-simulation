"""
Paquete OpenVLA. Expone `OpenVLAController` (carga OpenVLA EN PROCESO; requiere
torch/transformers en el entorno). Se sirve por HTTP con la capa generica
`models.serving` (server.py --model openvla); el benchmark lo consume con
`RemoteModel`, sin arrastrar torch/transformers.

Import perezoso (PEP 562): importar el paquete no importa torch/transformers.
"""

__all__ = ["OpenVLAController"]


def __getattr__(name):
    if name == "OpenVLAController":
        from .openvla_controller import OpenVLAController
        return OpenVLAController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
