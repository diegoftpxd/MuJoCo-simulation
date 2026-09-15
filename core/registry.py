"""
Registro GENERICO y PEREZOSO por nombre (base de `ModelFactory` y
`BenchmarkFactory`).

Igual que `RobotFactory` (robots/factory.py) permite crear un robot por nombre,
esta clase permite registrar y crear modelos o benchmarks por nombre. La
diferencia es que aqui el registro es PEREZOSO: cada entrada puede ser una
CADENA "paquete.modulo:Clase" que solo se importa al momento de crear la
instancia (`create`/`get`), no al registrarse.

Por que perezoso: los controladores de modelos importan stacks pesados e
INCOMPATIBLES entre si (torch/lerobot para pi0, torch/transformers para OpenVLA)
que ademas no estan instalados en el entorno del benchmark. Si el registro
importara todas las clases al construirse, romperia el aislamiento por proceso
(ver models/serving/). Con cadenas perezosas, `import models` solo carga este
registro (stdlib), y torch/lerobot se importan unicamente cuando se pide ese
modelo en su propio entorno.

Uso:
    reg = LazyRegistry("modelo")

    # Registro perezoso (no importa nada todavia):
    reg.register("openvla", "models.OpenVLA.openvla_controller:OpenVLAController")

    # O como decorador, si la clase ya esta importada (registro ansioso):
    @reg.register("random")
    class RandomController(Model): ...

    reg.available()          # ["openvla", "random"]
    reg.create("openvla", device="cuda")   # importa y construye
"""

import importlib


class LazyRegistry:
    """Registro por nombre con importacion perezosa de las entradas."""

    def __init__(self, kind="item"):
        """
        Parametros
        ----------
        kind : str
            Etiqueta para los mensajes de error (p. ej. "modelo", "benchmark").
        """
        self._kind = kind
        self._registry = {}      # nombre -> spec ("modulo:attr" | clase | callable)

    def register(self, name, target=None):
        """
        Registra `name`. Dos formas de uso:

        - Perezosa (recomendada): `register("x", "paquete.modulo:Clase")`. La
          clase NO se importa hasta `create`/`get`.
        - Como decorador: `@register("x")` sobre una clase/funcion ya importada
          (registro ansioso; util para modelos definidos por el usuario).

        Devuelve `target` (o el decorador), para poder encadenar/decorar.
        """
        if target is not None:
            self._add(name, target)
            return target

        def decorator(obj):
            self._add(name, obj)
            return obj
        return decorator

    def _add(self, name, target):
        if name in self._registry:
            raise ValueError(f"Ya existe un {self._kind} registrado como '{name}'.")
        self._registry[name] = target

    def get(self, name):
        """
        Devuelve el CALLABLE registrado como `name` (clase o funcion), importandolo
        de forma perezosa si se registro como cadena. No lo instancia.
        """
        if name not in self._registry:
            raise KeyError(
                f"{self._kind.capitalize()} '{name}' desconocido. "
                f"Disponibles: {self.available()}")
        target = self._registry[name]
        if isinstance(target, str):
            target = self._resolve(target)
            self._registry[name] = target      # cachea la clase ya importada
        return target

    def create(self, name, **kwargs):
        """Crea una instancia del `name` registrado, pasandole `**kwargs`."""
        return self.get(name)(**kwargs)

    def available(self):
        """Lista los nombres registrados."""
        return sorted(self._registry)

    @staticmethod
    def _resolve(spec):
        """Importa 'paquete.modulo:atributo' y devuelve el atributo."""
        module_path, sep, attr = spec.partition(":")
        if not sep:
            raise ValueError(
                f"Spec de registro invalida: '{spec}'. "
                "Usa el formato 'paquete.modulo:Clase'.")
        module = importlib.import_module(module_path)
        return getattr(module, attr)
