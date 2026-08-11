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


def run_episode(benchmark, model, max_steps=300, recorder=None):
    """
    Corre un episodio: `benchmark` y `model` solo se comunican via
    `Observation`/`Action`, por lo que se pueden intercambiar libremente.
    """
    model.reset()
    observation = benchmark.reset()
    total_reward, done, step = 0.0, False, 0

    for step in range(max_steps):
        if recorder is not None:
            recorder.record(observation)
        action = model.act(observation)
        result = benchmark.step(action)
        observation = result.observation
        total_reward += result.reward
        done = result.done
        if done:
            break

    return EpisodeResult(success=done, steps=step + 1, total_reward=total_reward)
