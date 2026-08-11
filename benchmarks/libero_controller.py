"""
Controlador de LIBERO: el wrapper que encapsula un entorno de LIBERO (robosuite)
y mantiene la interfaz común `BenchMark`.

- En la `Observation` entrega la vista `agentview` **ya orientada** (robosuite la
  renderiza girada 180°): así el modelo recibe la imagen derecha sin saber de
  ese detalle.
- En `step` toma el `cartesian_delta` + `gripper` de la `Action` y adapta la
  **convención de pinza de LIBERO** ([0,1] → {-1,+1} invertida).

Encuentra el repo de LIBERO clonado bajo `benchmarks/` (sin importar cuantas
carpetas de anidamiento tenga) y lo agrega al `sys.path`, así funciona sin
`pip install`. Las deps pesadas (robosuite, mujoco) siguen siendo necesarias.
"""

import glob
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")   # render headless (antes de mujoco)

import numpy as np

from benchmarks.benchmark import BenchMark
from core import Observation, StepResult, View

_BENCHMARKS_DIR = os.path.dirname(os.path.abspath(__file__))


def _ensure_libero_importable():
    """Agrega la raiz del repo LIBERO al sys.path (el dir que contiene `libero/`)."""
    try:
        import libero  # noqa: F401  ya importable (p. ej. instalado con pip)
        return
    except ImportError:
        pass
    # Busca el subpaquete anidado libero/libero para ubicar la raiz del repo.
    hits = glob.glob(os.path.join(_BENCHMARKS_DIR, "**", "libero", "libero",
                                  "__init__.py"), recursive=True)
    if not hits:
        raise ImportError(
            "No encontre el repo de LIBERO bajo benchmarks/. Clonalo ahi o "
            "instala el paquete `libero`.")
    # .../<repo>/libero/libero/__init__.py -> raiz del repo = 3 dirs arriba.
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(hits[0])))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


class LiberoController(BenchMark):
    """Una tarea de una suite de LIBERO (por defecto `libero_10`)."""

    def __init__(self, task_id=0, suite="libero_10",
                 init_state=0, num_settle=10, camera_size=256):
        _ensure_libero_importable()
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        self._suite = benchmark.get_benchmark_dict()[suite]()
        task = self._suite.get_task(task_id)
        self._instruction = task.language
        bddl = os.path.join(get_libero_path("bddl_files"),
                            task.problem_folder, task.bddl_file)

        self.env = OffScreenRenderEnv(bddl_file_name=bddl,
                                      camera_heights=camera_size,
                                      camera_widths=camera_size)
        self.env.seed(0)
        self._init_states = self._suite.get_task_init_states(task_id)
        self._init_state = init_state
        self._num_settle = num_settle

    @property
    def instruction(self) -> str:
        return self._instruction

    def _to_observation(self, raw) -> Observation:
        # robosuite entrega las imagenes giradas 180° -> las dejamos derechas.
        images = {View.AGENT.value:
                  np.ascontiguousarray(raw["agentview_image"][::-1, ::-1])}
        wrist = raw.get("robot0_eye_in_hand_image")
        if wrist is not None:
            images[View.WRIST.value] = np.ascontiguousarray(wrist[::-1, ::-1])
        state = {
            "eef_pos": raw.get("robot0_eef_pos"),
            "gripper_qpos": raw.get("robot0_gripper_qpos"),
        }
        return Observation(images=images, state=state,
                           instruction=self._instruction)

    def reset(self) -> Observation:
        self.env.reset()
        self.env.set_init_state(self._init_states[self._init_state])
        raw = None
        for _ in range(self._num_settle):     # dejar asentar los objetos
            raw, _, _, _ = self.env.step([0, 0, 0, 0, 0, 0, -1])
        return self._to_observation(raw)

    def step(self, action) -> StepResult:
        # El benchmark decide QUE usa de la Action: delta cartesiano + pinza.
        vec = np.zeros(7)
        if action.cartesian_delta is not None:
            vec[:6] = np.asarray(action.cartesian_delta, dtype=float)[:6]
        vec[6] = 0.0 if action.gripper is None else float(action.gripper)
        vec = self._libero_gripper(vec)
        raw, reward, done, info = self.env.step(vec.tolist())
        return StepResult(observation=self._to_observation(raw),
                          reward=float(reward), done=bool(done), info=info)

    @staticmethod
    def _libero_gripper(vec):
        """Pinza: [0,1] -> {-1,+1} binarizada e invertida (convención LIBERO)."""
        vec = vec.copy()
        vec[6] = 2.0 * vec[6] - 1.0
        vec[6] = np.sign(vec[6])
        vec[6] = -vec[6]
        return vec

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass
