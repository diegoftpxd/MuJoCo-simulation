"""
Capa de serving GENERICA: cualquier `Model` como servidor HTTP, consumible por
un benchmark que no comparte sus dependencias.

  - RemoteModel : cliente (implementa `Model`); solo numpy + stdlib. Vive en el
                  entorno del benchmark.
  - serve       : levanta el servidor HTTP para un `Model` cargado. Vive en el
                  entorno del modelo.
  - wire        : serializacion Observation/Action (solo numpy).

Imports perezosos (PEP 562): pedir `RemoteModel` NO importa el servidor ni los
controllers de los modelos.
"""

__all__ = ["RemoteModel", "serve", "wire"]


def __getattr__(name):
    if name == "RemoteModel":
        from .client import RemoteModel
        return RemoteModel
    if name == "serve":
        from .server import serve
        return serve
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
