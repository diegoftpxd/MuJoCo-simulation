"""
Corre un episodio de un benchmark con un modelo y graba el video.

Benchmark y modelo (ambos "controladores" que mantienen la interfaz común) solo
se comunican via `Observation`/`Action`, así que se intercambian cambiando una
línea.

    python run.py     # ExampleController (Panda) + RandomController -> output/episode.mp4
"""

from benchmarks.example import ExampleController
from core import VideoRecorder, View, run_episode
from models.Random import RandomController


def main():
    benchmark = ExampleController(robot="panda")
    model = RandomController(scale=0.03)

    recorder = VideoRecorder(view=View.SIDE_1)      # graba una de las vistas
    result = run_episode(benchmark, model, max_steps=60, recorder=recorder)
    recorder.save("output/episode.mp4")

    print(f"episodio: exito={result.success} | pasos={result.steps}")
    print("instruccion:", benchmark.instruction)


if __name__ == "__main__":
    main()
