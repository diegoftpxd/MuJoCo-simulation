"""
Capa comun: estructuras de datos e interfaz de ejecucion compartidas entre
benchmarks y modelos.

    Observation / View  -> lo que produce un benchmark (el modelo elige que usa)
    Action              -> lo que produce un modelo (el benchmark elige que usa)
    StepResult          -> salida de BenchMark.step
    run_episode         -> bucle benchmark <-> modelo
    VideoRecorder       -> graba una vista del episodio
"""

from .action import Action
from .episode import EpisodeResult, StepResult, VideoRecorder, run_episode
from .experiment import run_experiments, summarize
from .observation import Observation, View

__all__ = [
    "Observation",
    "View",
    "Action",
    "StepResult",
    "EpisodeResult",
    "run_episode",
    "VideoRecorder",
    "run_experiments",
    "summarize",
]
