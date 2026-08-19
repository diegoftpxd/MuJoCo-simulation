"""
Demostracion de `move_delta`: una secuencia tipo "pick & place" controlando el
TCP con incrementos de pose (`PoseDelta`) y la pinza.

    python demo_pick.py            # graba output/pick.mp4

Cada `Waypoint` describe, de forma legible, que hacer y cuando: en su instante
de inicio se llama una vez a `robot.controller.move_delta(...)`, que resuelve la
cinematica inversa; la simulacion se encarga del movimiento.
"""

from dataclasses import dataclass, field
from typing import Optional

from robots import Frame, PoseDelta, RobotFactory
from simulation import Simulation


@dataclass
class Waypoint:
    """Un paso de la secuencia: cuando, que movimiento y estado de la pinza."""
    start_time: float                       # segundos de simulacion
    delta: PoseDelta = field(default_factory=PoseDelta)
    gripper_open: Optional[bool] = None
    frame: Frame = Frame.WORLD


SEQUENCE = [
    Waypoint(0.5, gripper_open=True),                 # asegurar pinza abierta
    Waypoint(1.0, PoseDelta(dz=-0.12)),               # bajar hacia el objeto
    Waypoint(2.5, gripper_open=False),                # cerrar (agarrar)
    Waypoint(3.5, PoseDelta(dz=0.12)),                # levantar
    Waypoint(4, PoseDelta(dx=0.12)),
    Waypoint(5, PoseDelta(dy=0.15)),
    Waypoint(5.5, PoseDelta(dy=-0.15)),
    Waypoint(6, PoseDelta(dz=-0.10)),
    Waypoint(7, PoseDelta(droll=1)),
]


def make_controller(sequence):
    state = {"next": 0}

    def control(t, robot):
        i = state["next"]
        if i < len(sequence) and t >= sequence[i].start_time:
            wp = sequence[i]
            robot.controller.move_delta(wp.delta, gripper_open=wp.gripper_open,
                                        frame=wp.frame)
            state["next"] += 1

    return control


def main():
    sim = Simulation(RobotFactory.create("tm5-700"))
    #sim = Simulation(RobotFactory.create("panda"))
    sim.record_video(
        path="pick.mp4",
        duration=10.0,
        fps=30,
        control=make_controller(SEQUENCE),
        camera=sim.default_camera(azimuth=120, elevation=-20),
    )


if __name__ == "__main__":
    main()
