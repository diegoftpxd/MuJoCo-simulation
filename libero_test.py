"""
Test de integración: OpenVLA sobre tareas reales de LIBERO-10, grabando video.

Muestra los dos niveles de "ejemplos":
- ESCENARIO (tarea): se itera creando un `LiberoController` por tarea.
- EPISODIO (config inicial): dentro de una tarea, `run_episode(..., episode=i)`
  arranca configuraciones distintas en la MISMA instancia.

El modelo (OpenVLA) se carga UNA vez y se reutiliza en todo (se intercambia el
benchmark, no el modelo).

Requisitos: paquete `libero` importable + GPU + ~29 GB RAM (usar Kaggle).

    python libero_test.py
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from benchmarks.libero import LiberoController
from core import VideoRecorder, View, run_episode
from models.OpenVLA import OpenVLAController

SUITE = "libero_10"
NUM_TASKS = 3               # cuantas tareas (escenarios) probar
EPISODES_PER_TASK = 1       # cuantas configuraciones iniciales por tarea
MAX_STEPS = 220             # libero_10 es de horizonte largo (el eval oficial usa ~520)
OUT_DIR = "output/libero"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    model = OpenVLAController(view=View.AGENT, center_crop=True,
                             load_in_4bit=True, device="cuda")
    print("modelo cargado | unnorm_key =", model.unnorm_key)

    resultados = []
    # --- nivel ESCENARIO: iterar tareas de la suite ---
    for task_id, instruction in LiberoController.tasks(SUITE)[:NUM_TASKS]:
        benchmark = LiberoController(task_id=task_id, suite=SUITE)
        n_ep = min(EPISODES_PER_TASK, benchmark.num_episodes)
        print(f"\n[tarea {task_id}] {instruction}  ({benchmark.num_episodes} episodios)")

        # --- nivel EPISODIO: distintas configuraciones iniciales ---
        for ep in range(n_ep):
            recorder = VideoRecorder(view=View.AGENT)
            result = run_episode(benchmark, model, max_steps=MAX_STEPS,
                                 recorder=recorder, episode=ep)
            recorder.save(os.path.join(OUT_DIR, f"task{task_id}_ep{ep}.mp4"))
            print(f"  episodio {ep}: exito={result.success} | pasos={result.steps}")
            resultados.append((task_id, ep, result.success))

        benchmark.close()

    print("\n===== RESUMEN =====")
    for tid, ep, ok in resultados:
        print(f"  tarea {tid} ep {ep}: {'OK' if ok else 'no completada'}")


if __name__ == "__main__":
    main()
