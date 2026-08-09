"""
Genera un video del robot moviendose.

Uso:
    python record.py                       # movimiento demo, guarda movimiento.mp4
    python record.py --robot tm5-700 --out demo.mp4 --duration 6 --fps 30

El movimiento se define en `demo_control(t, robot)`: cambialo para grabar la
trayectoria que quieras.
"""

import argparse

import numpy as np

from robots import RobotFactory
from simulation import Simulation


def demo_control(t, robot):
    """
    Movimiento de demostracion: oscilaciones suaves de cada articulacion
    alrededor de la pose 'home'. `t` es el tiempo simulado en segundos.
    """
    home = np.array([0.0, 0.0, 1.5708, 0.0, 1.5708, 0.0])
    # Amplitud (rad) y frecuencia (Hz) por articulacion
    amp = np.array([0.8, 0.5, 0.5, 0.9, 0.6, 1.2])
    freq = np.array([0.2, 0.25, 0.3, 0.35, 0.4, 0.5])
    targets = home + amp * np.sin(2 * np.pi * freq * t)
    robot.set_joint_targets(targets)


def main():
    parser = argparse.ArgumentParser(description="Graba un video del robot moviendose")
    parser.add_argument("--robot", default="tm5-700", help="Robot a cargar.")
    parser.add_argument("--out", default="movimiento.mp4", help="Archivo de salida.")
    parser.add_argument("--duration", type=float, default=6.0, help="Duracion (s).")
    parser.add_argument("--fps", type=int, default=30, help="Cuadros por segundo.")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)
    args = parser.parse_args()

    sim = Simulation(RobotFactory.create(args.robot))
    sim.record_video(
        path=args.out,
        duration=args.duration,
        fps=args.fps,
        control=demo_control,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
