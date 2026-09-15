"""
Fabrica y registro de modelos (politicas) por nombre.

Analogo a `RobotFactory` (robots/factory.py), pero PEREZOSO (ver
`core.registry.LazyRegistry`): cada modelo se registra como una cadena
"modulo:Clase" que solo se importa al crearlo. Asi `import models` no arrastra
torch/lerobot/transformers, y cada controlador carga su stack unicamente en su
propio entorno (ver models/serving/).

Para agregar un modelo:
  1. Crea su controlador en `models/TuModelo/` (subclase de `Model`).
  2. Registra su ruta AQUI con una linea:
         ModelFactory.register("tu-modelo",
                               "models.TuModelo.tu_controller:TuController")
  3. Listo: el servidor generico (models/serving/server.py) ya lo puede servir
     por nombre y `ModelFactory.available()` lo lista. No hay que tocar nada mas.

Uso:
    from models import ModelFactory
    ModelFactory.available()                 # ["openvla", "pi0", "random"]
    model = ModelFactory.create("random")    # en el entorno del modelo
"""

from core.registry import LazyRegistry

ModelFactory = LazyRegistry("modelo")

# Registro perezoso: estas cadenas NO importan torch/lerobot hasta `create`.
ModelFactory.register("openvla", "models.OpenVLA.openvla_controller:OpenVLAController")
ModelFactory.register("pi0", "models.Pi_zero.Pi_zero_controller:PiZeroController")
ModelFactory.register("random", "models.Random.random_controller:RandomController")
