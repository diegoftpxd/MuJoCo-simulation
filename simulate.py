"""
Entorno de simulacion de brazos roboticos en MuJoCo.

Toda la logica especifica de cada robot vive en el paquete `robots/` (clase RobotArm y
sus subclases). Este script solo elige un robot y lo simula.

Uso:
    python simulate.py                 # abre el visor con el TM5-700
    python simulate.py --robot tm5-700 # elige el robot por nombre
    python simulate.py --check         # solo valida que el modelo carga (sin GUI)
    python simulate.py --list          # lista los robots disponibles

Controles del visor:
    - Arrastrar con el mouse: orbitar / hacer zoom.
    - Doble clic en un cuerpo: seleccionarlo.
    - Ctrl + arrastrar sobre un cuerpo: aplicar fuerzas.
    - Backspace: reiniciar la simulacion.
"""

import argparse

from robots import RobotFactory
from simulation import Simulation


def main():
    parser = argparse.ArgumentParser(description="Simulacion MuJoCo de brazos roboticos")

    parser.add_argument("--robot", default="tm5-700",
                        help="Nombre del robot a cargar (ver --list).")

    parser.add_argument("--check", action="store_true",
                        help="Solo valida que el modelo carga, sin abrir la GUI.")

    parser.add_argument("--list", action="store_true",
                        help="Lista los robots disponibles y termina.")
    args = parser.parse_args()

    if args.list:
        print("Robots disponibles:")
        for name in RobotFactory.available():
            print(f"  - {name}")
        return

    robot = RobotFactory.create(args.robot)
    robot.summary()
    sim = Simulation(robot)

    if args.check:
        print("\nOK: el modelo compila y simula correctamente.")
        print("TCP en pose home:", robot.end_effector_position().round(4))
        return

    print("\nAbriendo el visor de MuJoCo... (cierra la ventana para terminar)")
    sim.launch_viewer()


if __name__ == "__main__":
    main()
