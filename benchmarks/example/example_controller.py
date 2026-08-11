"""
Controlador de benchmark de ejemplo: el wrapper que encapsula nuestra
`Simulation` (Panda) y mantiene la interfaz común `BenchMark`.

Sirve para probar la arquitectura SIN GPU ni LIBERO: ofrece 3 vistas de camara
(`side_1`, `side_2`, `wrist`) en la `Observation`, y aplica el `cartesian_delta`
de la `Action`. Muestra el patron que luego siguen los benchmarks reales.
"""

from benchmarks.benchmark import BenchMark
from core import Observation, StepResult, View
from robots import Frame, PoseDelta, RobotFactory
from simulation import Simulation


class ExampleController(BenchMark):
    """Entorno mínimo que implementa la interfaz `BenchMark`."""

    def __init__(self, robot="panda", instruction="move the arm around",
                 frame_skip=5, episode_len=40, image_size=224):
        self.sim = Simulation(RobotFactory.create(robot))
        self._instruction = instruction
        self.frame_skip = frame_skip
        self.episode_len = episode_len
        self.image_size = image_size
        self._cameras = {
            View.SIDE_1.value: self.sim.default_camera(azimuth=135, elevation=-20),
            View.SIDE_2.value: self.sim.default_camera(azimuth=45, elevation=-20),
            View.WRIST.value: self.sim.default_camera(azimuth=90, elevation=-70,
                                                      distance=0.6),
        }
        self._steps = 0

    @property
    def instruction(self) -> str:
        return self._instruction

    def _observe(self) -> Observation:
        n = self.image_size
        images = {name: self.sim.observation(cam, n, n)
                  for name, cam in self._cameras.items()}
        state = {
            "joint_positions": self.sim.robot.get_joint_positions(),
            "tcp": self.sim.robot.end_effector_position(),
        }
        return Observation(images=images, state=state,
                           instruction=self._instruction)

    def reset(self) -> Observation:
        self.sim.reset()
        self._steps = 0
        return self._observe()

    def step(self, action) -> StepResult:
        # El benchmark decide QUE usa de la Action: el delta cartesiano + pinza.
        if action.cartesian_delta is not None:
            gripper_open = action.gripper is None or action.gripper > 0.5
            self.sim.robot.controller.cartesian_step(
                PoseDelta.from_sequence(action.cartesian_delta),
                gripper_open=gripper_open, frame=Frame.TOOL)
        self.sim.step(self.frame_skip)
        self._steps += 1
        done = self._steps >= self.episode_len
        return StepResult(observation=self._observe(), reward=0.0, done=done)
