"""
Test de integración: OpenVLA sobre 3 tareas reales de LIBERO-10, grabando video.

Usa la arquitectura común: un `Model` (OpenVLA) y un `BenchMark` (LIBERO) que
solo se comunican por `Observation`/`Action`. El modelo se carga UNA vez y se
reutiliza en las 3 tareas (se intercambia el benchmark, no el modelo).

Requisitos: paquete `libero` instalado + GPU + ~29 GB RAM (usar Kaggle).

    python libero_test.py
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from benchmarks.libero_controller import LiberoController
from core import VideoRecorder, View, run_episode
from models.OpenVLA import OpenVLAController

NUM_TASKS = 3
MAX_STEPS = 220          # libero_10 es de horizonte largo; el eval oficial usa ~520
OUT_DIR = "output/libero"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # El modelo se carga una sola vez (la parte cara).
    model = OpenVLAController(view=View.AGENT, center_crop=True,
                             load_in_4bit=True, device="cuda")
    print("modelo cargado | unnorm_key =", model.unnorm_key)

    resultados = []
    for task_id in range(NUM_TASKS):
        benchmark = LiberoController(task_id=task_id, suite="libero_10")
        print(f"\n[tarea {task_id}] {benchmark.instruction}")

        recorder = VideoRecorder(view=View.AGENT)
        result = run_episode(benchmark, model, max_steps=MAX_STEPS, recorder=recorder)
        recorder.save(os.path.join(OUT_DIR, f"task{task_id}.mp4"))
        benchmark.close()

        print(f"  exito={result.success} | pasos={result.steps}")
        resultados.append((task_id, benchmark.instruction, result.success))

    print("\n===== RESUMEN =====")
    for tid, instr, ok in resultados:
        print(f"  tarea {tid}: {'OK' if ok else 'no completada'} | {instr}")


if __name__ == "__main__":
    main()
