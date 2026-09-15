"""
Contrato de capacidades entre benchmark y modelo.

El desacople benchmark <-> modelo (ver core/observation.py y core/action.py) es
comodo pero tiene un riesgo: nada garantiza que el benchmark OFREZCA lo que el
modelo NECESITA. Si pi0 espera el estado `eef_quat` y el benchmark no lo pone en
la `Observation`, el error salta recien en la primera inferencia, como un
mensaje cripto de tamano de tensor ("tensor a (5) must match tensor b (8)").

`Capabilities` hace ese contrato EXPLICITO:
  - Un MODELO declara lo que REQUIERE  -> `Model.requirements()`.
  - Un BENCHMARK declara lo que OFRECE -> `BenchMark.capabilities()`.
Y `check_compatibility` los compara ANTES de correr el episodio, fallando con un
mensaje claro que dice exactamente que falta.

El mismo tipo describe "lo requerido" y "lo ofrecido": son conjuntos de nombres
de vistas y de estados, mas si hace falta (o hay) instruccion en lenguaje natural.
"""

from dataclasses import dataclass, field

from .observation import Observation, View


def _norm(name) -> str:
    """Normaliza un nombre de vista/estado (View o str) a su string canonico."""
    return name.value if isinstance(name, View) else str(name)


@dataclass(frozen=True)
class Capabilities:
    """Vistas + estados (+ instruccion) que un modelo REQUIERE o un benchmark OFRECE."""

    views: frozenset = field(default_factory=frozenset)
    state: frozenset = field(default_factory=frozenset)
    instruction: bool = False

    # ------------------------------------------------------------------ #
    #  Construccion
    # ------------------------------------------------------------------ #
    @classmethod
    def of(cls, views=(), state=(), instruction=False) -> "Capabilities":
        """Atajo legible: `Capabilities.of(views=[View.AGENT], state=["eef_pos"])`."""
        return cls(frozenset(_norm(v) for v in views),
                   frozenset(_norm(s) for s in state),
                   bool(instruction))

    @classmethod
    def from_observation(cls, obs: Observation) -> "Capabilities":
        """
        Capacidades que una `Observation` REAL ofrece. Sirve de red de seguridad
        cuando un benchmark no declara `capabilities()`: se valida contra lo que
        de verdad trae la primera observacion (imposible que se desincronice).
        """
        return cls(frozenset(obs.images), frozenset(obs.state),
                   obs.instruction is not None)

    # ------------------------------------------------------------------ #
    #  Comparacion
    # ------------------------------------------------------------------ #
    def missing_from(self, provided: "Capabilities") -> "Capabilities":
        """Lo que `self` (requisitos) exige y `provided` (oferta) NO cubre."""
        return Capabilities(
            views=self.views - provided.views,
            state=self.state - provided.state,
            instruction=self.instruction and not provided.instruction,
        )

    def is_empty(self) -> bool:
        return not self.views and not self.state and not self.instruction

    def describe(self) -> str:
        parts = []
        if self.views:
            parts.append(f"vistas={sorted(self.views)}")
        if self.state:
            parts.append(f"estado={sorted(self.state)}")
        if self.instruction:
            parts.append("instruccion")
        return ", ".join(parts) if parts else "(nada)"

    # ------------------------------------------------------------------ #
    #  Serializacion (cruza la frontera de proceso modelo <-> benchmark)
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {"views": sorted(self.views), "state": sorted(self.state),
                "instruction": self.instruction}

    @classmethod
    def from_dict(cls, d: dict) -> "Capabilities":
        return cls(frozenset(d.get("views", ())),
                   frozenset(d.get("state", ())),
                   bool(d.get("instruction", False)))


class IncompatibleCapabilities(Exception):
    """El benchmark no ofrece algo que el modelo requiere."""


def check_compatibility(required: Capabilities, provided: Capabilities,
                        model_name="modelo", benchmark_name="benchmark") -> None:
    """
    Verifica que `provided` (lo que ofrece el benchmark) cubra `required` (lo que
    exige el modelo). Si falta algo, lanza `IncompatibleCapabilities` con un
    mensaje que dice requiere / ofrece / falta. No devuelve nada si todo cuadra.
    """
    missing = required.missing_from(provided)
    if not missing.is_empty():
        raise IncompatibleCapabilities(
            f"El benchmark '{benchmark_name}' no ofrece lo que el modelo "
            f"'{model_name}' requiere.\n"
            f"  requiere : {required.describe()}\n"
            f"  ofrece   : {provided.describe()}\n"
            f"  falta    : {missing.describe()}")
