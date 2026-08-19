"""
Orquestacion de experimentos: corre un modelo sobre varios escenarios y
episodios, y recolecta los resultados de forma estructurada.

No sabe nada de OpenVLA ni de LIBERO: recibe una lista de `BenchMark` (uno por
escenario) y un `Model`, y devuelve una lista de dicts con el resultado de cada
episodio. Asi el notebook queda fino (config + mostrar) y esta logica es
reutilizable y testeable.
"""

import os

from .episode import VideoRecorder, run_episode


def run_experiments(model, benchmarks, max_steps=300, episodes_per_task=1,
                    out_dir=None, record_view=None):
    """
    Corre `model` sobre cada benchmark de `benchmarks` (nivel escenario) y, en
    cada uno, `episodes_per_task` episodios (nivel configuracion inicial).

    Devuelve una lista de dicts:
        {scenario, instruction, episode, success, steps, video}

    Si `out_dir` no es None, graba un video por episodio (usando `record_view`).
    """
    results = []
    for scenario, benchmark in enumerate(benchmarks):
        n_ep = min(episodes_per_task, benchmark.num_episodes)
        for episode in range(n_ep):
            recorder = VideoRecorder(record_view) if out_dir is not None else None
            outcome = run_episode(benchmark, model, max_steps=max_steps,
                                  recorder=recorder, episode=episode)
            video = None
            if out_dir is not None and recorder is not None:
                os.makedirs(out_dir, exist_ok=True)
                video = os.path.join(out_dir, f"scenario{scenario}_ep{episode}.mp4")
                recorder.save(video)
            results.append({
                "scenario": scenario,
                "instruction": benchmark.instruction,
                "episode": episode,
                "success": outcome.success,
                "steps": outcome.steps,
                "video": video,
            })
        benchmark.close()
    return results


def summarize(results):
    """Resumen global: total de episodios, exitos y tasa de exito."""
    total = len(results)
    exitos = sum(1 for r in results if r["success"])
    return {"total": total, "exitos": exitos,
            "tasa_exito": (exitos / total) if total else 0.0}
