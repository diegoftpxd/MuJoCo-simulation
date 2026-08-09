"""
Fabrica y registro de robots.

Cada robot concreto (en su propio archivo del paquete) se registra decorando su
clase con `@RobotFactory.register("nombre")`, y se crea por nombre con
`RobotFactory.create("nombre")`. Asi, agregar un robot NO requiere editar ningun
diccionario central: basta con definir su clase, decorarla e importarla en
`robots/__init__.py`.
"""


class RobotFactory:
    """Fabrica y registro de robots por nombre."""

    _registry = {}

    @classmethod
    def register(cls, name):
        """Decorador que registra una clase de robot bajo `name`."""
        def decorator(robot_cls):
            if name in cls._registry:
                raise ValueError(f"Ya existe un robot registrado como '{name}'.")
            cls._registry[name] = robot_cls
            robot_cls.robot_id = name
            return robot_cls
        return decorator

    @classmethod
    def create(cls, name):
        """Crea una instancia del robot registrado como `name`."""
        if name not in cls._registry:
            raise KeyError(
                f"Robot '{name}' desconocido. Disponibles: {cls.available()}"
            )
        return cls._registry[name]()

    @classmethod
    def available(cls):
        """Lista los nombres de los robots registrados."""
        return sorted(cls._registry)


def make_robot(name="tm5-700"):
    """Atajo para `RobotFactory.create(name)`."""
    return RobotFactory.create(name)
