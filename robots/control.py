"""
Control cartesiano de un brazo mediante cinematica inversa.

`CartesianController` es una POLITICA DE CONTROL sobre un robot: recibe un
`RobotArm` y le permite mover el TCP por incrementos de pose (Δx, Δy, Δz,
Δroll, Δpitch, Δyaw) resolviendo la IK. Se separa de `RobotArm` a proposito:
el robot es estado + control articular + cinematica; la IK es una estrategia
que podria intercambiarse (IK amortiguada, OSC, control por velocidad, ...)
sin tocar la clase del robot.

Uso tipico (via el acceso `robot.controller`):

    from robots import PoseDelta, Frame, CartesianAction

    robot.controller.move_delta(PoseDelta(dz=-0.05), frame=Frame.WORLD,
                                gripper_open=False)
    robot.controller.cartesian_step(CartesianAction.from_vector(vec))  # RL
"""

import mujoco
import numpy as np

from .structures import CartesianAction, Frame, PoseDelta


class CartesianController:
    """Controla el TCP de un `RobotArm` por incrementos de pose (IK)."""

    def __init__(self, robot):
        self.robot = robot

    # ------------------------------------------------------------------ #
    #  Movimiento a un objetivo (IK iterativa) — para waypoints scripted
    # ------------------------------------------------------------------ #
    def move_delta(self, delta, gripper_open=None, frame=Frame.TOOL, **ik_kwargs):
        """
        Mueve el efector final por un incremento de pose y (opcionalmente)
        abre/cierra la pinza.

        Resuelve la cinematica inversa para llevar el TCP a la pose objetivo
        (pose actual + delta) y fija los objetivos de las articulaciones. El
        movimiento fisico ocurre cuando se avanza la simulacion (`sim.step`);
        con `Simulation.move_delta` se hace en un solo paso.

        Parametros
        ----------
        delta : PoseDelta | CartesianAction | secuencia de 6
            Incremento de pose. Con `CartesianAction` tambien lleva la pinza.
        gripper_open : bool | None
            `True` abre, `False` cierra, `None` la deja igual. Tiene prioridad
            sobre la pinza que traiga un `CartesianAction`.
        frame : Frame
            `Frame.TOOL` (por defecto): deltas en el marco del TCP. `Frame.WORLD`:
            deltas en el marco del mundo. Tambien acepta "tool"/"world".
        **ik_kwargs :
            Parametros extra para el solver de IK (`max_iters`, `tol`, `damping`).

        Devuelve
        --------
        bool : True si la IK convergio a la pose objetivo.
        """
        delta, gripper_open = self._split_action(delta, gripper_open)
        frame = Frame(frame)
        robot = self.robot

        pose = robot.end_effector_pose()
        rotation_delta = self._euler_to_rotation_matrix(*delta.rotation_rpy)

        if frame is Frame.TOOL:
            target_position = pose.position + pose.rotation @ delta.translation
            target_rotation = pose.rotation @ rotation_delta
        else:  # Frame.WORLD
            target_position = pose.position + delta.translation
            target_rotation = rotation_delta @ pose.rotation

        qsol, converged = self._solve_inverse_kinematics(
            target_position, target_rotation, **ik_kwargs)
        robot.set_joint_targets(qsol)
        if gripper_open is not None:
            robot.set_gripper(gripper_open)
        return converged

    # ------------------------------------------------------------------ #
    #  Accion de un paso (IK diferencial) — para RL
    # ------------------------------------------------------------------ #
    def cartesian_step(self, action, gripper_open=None, frame=Frame.TOOL,
                       damping=1e-2, max_dq=0.2):
        """
        Accion de control cartesiano de UN paso, pensada para RL.

        A diferencia de `move_delta`, no itera hasta converger ni avanza la
        fisica: hace una sola linealizacion (un paso de IK diferencial con el
        Jacobiano) y actualiza los objetivos de las articulaciones. Pensada para
        llamarse una vez por `env.step`, seguida de `sim.step(frame_skip)`.

        Es ~20x mas barata que `move_delta` (no usa `mj_forward` ni bucle) y no
        perturba el estado de la simulacion.

        Parametros
        ----------
        action : CartesianAction | PoseDelta | secuencia de 6
            La accion cartesiana. Con `CartesianAction` incluye la pinza, ideal
            para RL: `cartesian_step(CartesianAction.from_vector(vec))`.
        gripper_open : bool | None
            Sobrescribe la pinza del `CartesianAction` si se pasa.
        frame : Frame
            `Frame.TOOL` (por defecto) o `Frame.WORLD` (tambien "tool"/"world").
        damping : float
            Amortiguamiento del minimo cuadrado (regulariza cerca de singularidades).
        max_dq : float
            Limite de |dq| por articulacion y paso (rad), por estabilidad.

        Devuelve
        --------
        np.ndarray : el vector dq aplicado a las articulaciones del brazo.
        """
        delta, gripper_open = self._split_action(action, gripper_open)
        frame = Frame(frame)
        robot = self.robot

        twist_local = delta.as_twist()
        if frame is Frame.TOOL:
            rotation = robot.end_effector_orientation()
            twist = np.concatenate([rotation @ twist_local[:3],
                                    rotation @ twist_local[3:]])
        else:  # Frame.WORLD
            twist = twist_local

        J = robot.arm_jacobian()
        dq = J.T @ np.linalg.solve(J @ J.T + (damping ** 2) * np.eye(6), twist)
        dq = np.clip(dq, -max_dq, max_dq)

        rng = robot.joint_ranges
        target = np.clip(robot.get_joint_positions() + dq, rng[:, 0], rng[:, 1])
        robot.set_joint_targets(target)
        if gripper_open is not None:
            robot.set_gripper(gripper_open)
        return dq

    @staticmethod
    def _split_action(action, gripper_open):
        """
        Normaliza la entrada a `(PoseDelta, gripper_open)`.

        Acepta un `CartesianAction` (usa su pinza salvo que `gripper_open` la
        sobrescriba), un `PoseDelta` o una secuencia de 6 numeros.
        """
        if isinstance(action, CartesianAction):
            if gripper_open is None:
                gripper_open = action.gripper_open
            return action.delta, gripper_open
        return PoseDelta.coerce(action), gripper_open

    # ------------------------------------------------------------------ #
    #  Solver de IK
    # ------------------------------------------------------------------ #
    def _solve_inverse_kinematics(self, target_position, target_rotation,
                  max_iters=200, tol=1e-4, damping=1e-2, step=1.0):
        """
        Cinematica inversa por minimos cuadrados amortiguados (Jacobiano).

        Itera sobre las articulaciones del brazo hasta que el TCP alcanza la
        pose objetivo. No altera el estado dinamico: guarda y restaura qpos/qvel
        pase lo que pase (incluso si el solver lanza una excepcion). Devuelve
        (q_solucion, convergio).
        """
        robot = self.robot
        model, data = robot.model, robot.data
        arm_q = robot.arm_joint_position_indices
        rng = robot.joint_ranges

        qpos_backup = data.qpos.copy()
        qvel_backup = data.qvel.copy()
        converged = False
        try:
            for _ in range(max_iters):
                # Solo cinematica (no dinamica): 10x mas barato que mj_forward.
                mujoco.mj_kinematics(model, data)
                mujoco.mj_comPos(model, data)

                err = np.concatenate([
                    target_position - robot.end_effector_position(),
                    self._orientation_error(robot.end_effector_orientation(), target_rotation),
                ])
                if np.linalg.norm(err) < tol:
                    converged = True
                    break

                J = robot.arm_jacobian()
                # dq = J^T (J J^T + lambda^2 I)^-1 err   (damped least squares)
                dq = J.T @ np.linalg.solve(J @ J.T + (damping ** 2) * np.eye(6), err)
                data.qpos[arm_q] = np.clip(data.qpos[arm_q] + step * dq,
                                           rng[:, 0], rng[:, 1])

            qsol = data.qpos[arm_q].copy()
        finally:
            # La IK solo era un calculo: restaurar SIEMPRE el estado real.
            data.qpos[:] = qpos_backup
            data.qvel[:] = qvel_backup
            mujoco.mj_forward(model, data)
        return qsol, converged

    # ------------------------------------------------------------------ #
    #  Utilidades de rotacion
    # ------------------------------------------------------------------ #
    @staticmethod
    def _euler_to_rotation_matrix(roll, pitch, yaw):
        """Matriz de rotacion a partir de roll/pitch/yaw (XYZ extrinseco)."""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    @staticmethod
    def _orientation_error(R_cur, R_des):
        """Error de orientacion (vector 3D en el marco del mundo)."""
        return 0.5 * (np.cross(R_cur[:, 0], R_des[:, 0]) +
                      np.cross(R_cur[:, 1], R_des[:, 1]) +
                      np.cross(R_cur[:, 2], R_des[:, 2]))
