"""
Resultado de un paso/episodio y el runner que conecta benchmark <-> modelo.

`run_episode` es el bucle agente-entorno estandar: pide una observacion al
benchmark, se la pasa al modelo, aplica la accion, y repite. No sabe nada de
OpenVLA ni de LIBERO en particular: solo habla `Observation`/`Action`.
"""

from dataclasses import dataclass, field


@dataclass
class StepResult:
    """Lo que devuelve `BenchMark.step`: observacion + señal de progreso."""
    observation: object            # Observation
    reward: float = 0.0
    done: bool = False
    info: dict = field(default_factory=dict)


@dataclass
class EpisodeResult:
    """Resumen de un episodio."""
    success: bool
    steps: int
    total_reward: float


class VideoRecorder:
    """Acumula una vista de la observacion cuadro a cuadro y la guarda a video."""

    def __init__(self, view=None):
        self.view = view          # nombre de vista a grabar (None = la primera)
        self.frames = []

    def record(self, observation):
        if self.view is not None and observation.has_image(self.view):
            self.frames.append(observation.image(self.view))
        elif observation.images:
            self.frames.append(next(iter(observation.images.values())))

    def save(self, path, fps=20):
        import os
        import imageio
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        imageio.mimsave(path, self.frames, fps=fps)
        print(f"video guardado: {path} ({len(self.frames)} cuadros)")
        return path


def run_episode(benchmark, model, max_steps=300, recorder=None, episode=None,
                validate=True):
    """
    Corre un episodio: `benchmark` y `model` solo se comunican via
    `Observation`/`Action`, por lo que se pueden intercambiar libremente.
    `episode` elige la configuracion inicial del escenario (ver `BenchMark`).

    El modelo devuelve una LISTA de acciones (un "chunk"): el runner las ejecuta
    una a una contra el benchmark y, cuando se agota el chunk, vuelve a pedir con
    la observacion actual. Un modelo de accion unica devuelve una lista de 1.
    `benchmark.step` sigue siendo atomico (una accion -> una observacion), asi
    que el reward, el `done` y la grabacion siguen a nivel de accion individual.

    Si `validate` (por defecto), comprueba el CONTRATO DE CAPACIDADES antes del
    primer paso: que el benchmark ofrezca las vistas/estados/instruccion que el
    modelo requiere (ver core/capabilities.py). Falla temprano y claro en vez de
    reventar en la primera inferencia con un error cripto de tamano de tensor.
    """
    model.reset()
    observation = benchmark.reset(episode)
    if validate:
        _check_capabilities(benchmark, model, observation)
    total_reward, done, step = 0.0, False, 0
    actions = []          # buffer del chunk actual (acciones aun no ejecutadas)

    for step in range(max_steps):
        if recorder is not None:
            recorder.record(observation)
        if not actions:                       # chunk agotado -> pedir mas al modelo
            actions = list(model.act(observation))
        action = actions.pop(0)
        result = benchmark.step(action)
        observation = result.observation
        total_reward += result.reward
        done = result.done
        if done:
            break

    return EpisodeResult(success=done, steps=step + 1, total_reward=total_reward)


def _check_capabilities(benchmark, model, observation):
    """
    Valida el contrato modelo <-> benchmark. Usa lo que el benchmark DECLARA
    (`capabilities()`); si no declara nada (None), cae a lo que la primera
    observacion realmente ofrece. Compara contra lo que el modelo REQUIERE.
    """
    from .capabilities import Capabilities, check_compatibility
    provided = benchmark.capabilities()
    if provided is None:
        provided = Capabilities.from_observation(observation)
    check_compatibility(model.requirements(), provided,
                        model_name=type(model).__name__,
                        benchmark_name=type(benchmark).__name__)
