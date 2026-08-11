"""
Clase base de un brazo robotico en MuJoCo.

`RobotArm` representa AL ROBOT: su modelo, el estado de sus articulaciones, su
interfaz de control y su cinematica. NO se encarga de simular ni de visualizar;
de eso se ocupa la clase `Simulation` (ver `simulation.py`), que recibe un robot
como atributo.

`RobotArm` es GENERICA: descubre por introspeccion las articulaciones, los
actuadores y el sitio del efector final a partir del MJCF, por lo que sirve para
cualquier brazo. Lo especifico de un robot concreto (ruta del XML, nombre del
TCP, keyframe de arranque) se define en una subclase, en su propio archivo
dentro de este paquete (p. ej. `tm5.py`, `panda.py`).
"""

import os

import mujoco
import numpy as np

from .structures import Pose

# Carpeta con los modelos MJCF (en la raiz del proyecto, junto al paquete).
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
MJCF_DIR = os.path.join(os.path.dirname(_PKG_DIR), "MJCF")


class RobotArm:
    """El robot: modelo, estado articular, control y cinematica."""

    def __init__(self, xml_path, end_effector_site=None, end_effector_body=None, end_effector_offset=(0, 0, 0),
                 home_key="home", gripper_markers=("gripper", "finger"),
                 objects=None):
        """
        Parametros
        ----------
        xml_path : str
            Ruta al modelo MJCF (.xml) del robot.
        end_effector_site : str | None
            Nombre del `<site>` que marca el efector final (TCP). Si es None y
            tampoco se da `end_effector_body`, se usa el ultimo site del modelo (si existe).
        end_effector_body : str | None
            Alternativa a `end_effector_site` para modelos sin sitio TCP: nombre del
            cuerpo del efector. El TCP es ese cuerpo desplazado por `end_effector_offset`
            (util p. ej. para el Franka Panda, cuya mano no trae site).
        end_effector_offset : (3,)
            Desplazamiento del TCP respecto al origen de `end_effector_body` (marco local).
        home_key : str
            Nombre del keyframe usado como pose de arranque. Si no existe, el
            robot arranca en la pose cero.
        gripper_markers : tuple[str, ...]
            Subcadenas que identifican las articulaciones de la pinza por su
            nombre (p. ej. "gripper", "finger"). Sirven para separar el brazo
            de la pinza automaticamente.
        objects : list[SceneObject] | None
            Objetos a agregar al entorno (cubos, esferas, ...). Se insertan en
            el modelo antes de compilar. Quedan accesibles en `self.objects`.
        """
        self.xml_path = os.path.abspath(xml_path)
        objects = list(objects) if objects else []
        if objects:
            self.model = self._build_model_with_objects(self.xml_path, objects)
        else:
            self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.home_key = home_key
        self.gripper_markers = tuple(gripper_markers)

        # Buffers del Jacobiano, preasignados (se reutilizan en cada IK).
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))

        # Controlador cartesiano (perezoso), ver `controller`.
        self._controller = None

        # --- Articulaciones movibles (hinge/slide), separadas brazo vs pinza ---
        movable = [
            i for i in range(self.model.njnt)
            if self.model.jnt_type[i] in (mujoco.mjtJoint.mjJNT_HINGE,
                                          mujoco.mjtJoint.mjJNT_SLIDE)
        ]

        def _is_gripper(jid):
            nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
            return any(mk in nm for mk in self.gripper_markers)

        self.joint_ids = [i for i in movable if not _is_gripper(i)]
        self.gripper_joint_ids = [i for i in movable if _is_gripper(i)]
        self.joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in self.joint_ids
        ]
        # Direcciones en qpos y en qvel (dof) de las articulaciones del brazo
        self._qadr = np.array([self.model.jnt_qposadr[i] for i in self.joint_ids],
                              dtype=int)
        self._dofadr = np.array([self.model.jnt_dofadr[i] for i in self.joint_ids],
                                dtype=int)

        # --- Efector final (TCP): por sitio o por cuerpo + offset ---
        self.end_effector_offset = np.asarray(end_effector_offset, dtype=float)
        if end_effector_body is not None:
            self._end_effector_mode = "body"
            self.end_effector_body = end_effector_body
            self.end_effector_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                                end_effector_body)
            self.end_effector_site, self.end_effector_site_id = None, -1
        else:
            site = end_effector_site if end_effector_site is not None else self._default_end_effector_site()
            self._end_effector_mode = "site"
            self.end_effector_site = site
            self.end_effector_site_id = (
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site)
                if site else -1
            )
            self.end_effector_body, self.end_effector_body_id = None, -1

        # --- Actuadores ---
        self.actuator_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]
        # Mapa articulacion -> actuador, y actuadores que mueven el brazo
        self._joint_to_act = {}
        _arm_joint_set = set(self.joint_ids)
        _arm_actuators = set()
        for a in range(self.model.nu):
            if self.model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
                jid = int(self.model.actuator_trnid[a, 0])
                self._joint_to_act[jid] = a
                if jid in _arm_joint_set:
                    _arm_actuators.add(a)
        # Actuadores de la pinza: todo lo que NO mueve el brazo (incluye pinzas
        # accionadas por tendon, como la del Franka Panda).
        self.gripper_actuator_ids = [
            a for a in range(self.model.nu) if a not in _arm_actuators
        ]

        # --- Objetos del entorno ---
        # Enlaza cada objeto al modelo ya compilado; acceso por nombre.
        self.objects = {}
        for obj in objects:
            obj.bind(self.model, self.data)
            self.objects[obj.name] = obj

        self.reset()

    @staticmethod
    def _build_model_with_objects(xml_path, objects):
        """
        Compila el modelo agregando `objects` al mundo, via `mujoco.MjSpec`.

        Los objetos deben existir antes de compilar; ademas, si aportan un free
        joint (son movibles), se extiende el qpos de cada keyframe con su pose
        inicial para que las poses guardadas sigan siendo validas.
        """
        spec = mujoco.MjSpec.from_file(xml_path)
        extra_qpos, extra_qvel = [], []
        for obj in objects:
            obj.add_to_spec(spec)
            extra_qpos += obj.init_qpos
            extra_qvel += [0.0] * obj.n_dof
        if extra_qpos:
            for key in spec.keys:
                if len(key.qpos):
                    key.qpos = list(key.qpos) + extra_qpos
                if len(key.qvel):
                    key.qvel = list(key.qvel) + extra_qvel
        return spec.compile()

    @property
    def has_gripper(self):
        """True si el modelo tiene una pinza controlable."""
        return len(self.gripper_actuator_ids) > 0

    # ------------------------------------------------------------------ #
    #  Propiedades de solo lectura
    # ------------------------------------------------------------------ #
    @property
    def name(self):
        return self.model.names.split(b"\x00")[0].decode() or "robot"

    @property
    def num_joints(self):
        """Numero de articulaciones (grados de libertad del brazo)."""
        return len(self.joint_ids)

    @property
    def num_actuators(self):
        return self.model.nu

    @property
    def joint_ranges(self):
        """Array (num_joints, 2) con los limites [min, max] de cada articulacion."""
        return self.model.jnt_range[self.joint_ids].copy()

    # ------------------------------------------------------------------ #
    #  Estado de las articulaciones
    # ------------------------------------------------------------------ #
    def get_joint_positions(self):
        """Angulos actuales de las articulaciones (rad)."""
        return self.data.qpos[self._qadr].copy()

    def set_joint_positions(self, q):
        """Fija instantaneamente los angulos de las articulaciones (sin dinamica)."""
        self.data.qpos[self._qadr] = q
        mujoco.mj_forward(self.model, self.data)
        return self

    def get_joint_velocities(self):
        """Velocidades articulares actuales (rad/s)."""
        dofadr = np.array([self.model.jnt_dofadr[i] for i in self.joint_ids])
        return self.data.qvel[dofadr].copy()

    # ------------------------------------------------------------------ #
    #  Control
    # ------------------------------------------------------------------ #
    def set_joint_targets(self, q):
        """
        Fija los objetivos de las articulaciones (para actuadores de posicion).
        `q` debe tener longitud `num_joints`, en el orden de `self.joint_names`.
        """
        q = np.asarray(q, dtype=float)
        for k, jid in enumerate(self.joint_ids):
            a = self._joint_to_act.get(jid)
            if a is not None:
                self.data.ctrl[a] = q[k]
        return self

    def set_control(self, u):
        """Escribe directamente el vector de control (longitud `num_actuators`)."""
        self.data.ctrl[:] = u
        return self

    # ------------------------------------------------------------------ #
    #  Pinza
    # ------------------------------------------------------------------ #
    def set_gripper(self, is_open):
        """Abre (`True`) o cierra (`False`) la pinza."""
        if not self.has_gripper:
            raise ValueError("El modelo no tiene una pinza controlable.")
        for a in self.gripper_actuator_ids:
            lo, hi = self.model.actuator_ctrlrange[a]
            self.data.ctrl[a] = hi if is_open else lo
        return self

    def open_gripper(self):
        """Abre la pinza."""
        return self.set_gripper(True)

    def close_gripper(self):
        """Cierra la pinza."""
        return self.set_gripper(False)

    # ------------------------------------------------------------------ #
    #  Estado / pose
    # ------------------------------------------------------------------ #
    def reset(self, keyframe=None):
        """Reinicia el estado a un keyframe (por defecto el de `home_key`)."""
        key = keyframe if keyframe is not None else self.home_key
        kid = (mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, key)
               if key else -1)
        if kid >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, kid)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self

    # ------------------------------------------------------------------ #
    #  Cinematica del efector final (TCP)
    # ------------------------------------------------------------------ #
    def end_effector_position(self):
        """Posicion cartesiana del TCP en el mundo (x, y, z)."""
        self._require_end_effector()
        if self._end_effector_mode == "site":
            return self.data.site_xpos[self.end_effector_site_id].copy()
        R = self.data.xmat[self.end_effector_body_id].reshape(3, 3)
        return self.data.xpos[self.end_effector_body_id] + R @ self.end_effector_offset

    def end_effector_orientation(self):
        """Matriz de rotacion 3x3 del TCP respecto al mundo."""
        self._require_end_effector()
        if self._end_effector_mode == "site":
            return self.data.site_xmat[self.end_effector_site_id].reshape(3, 3).copy()
        return self.data.xmat[self.end_effector_body_id].reshape(3, 3).copy()

    def _end_effector_jacobian(self):
        """Rellena los buffers self._jacp/_jacr con el Jacobiano del TCP."""
        if self._end_effector_mode == "site":
            mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr,
                              self.end_effector_site_id)
        else:
            mujoco.mj_jac(self.model, self.data, self._jacp, self._jacr,
                          self.end_effector_position(), self.end_effector_body_id)

    def end_effector_pose(self):
        """Devuelve la `Pose` del TCP (posicion (3,) + rotacion (3x3))."""
        return Pose(self.end_effector_position(), self.end_effector_orientation())

    def arm_jacobian(self):
        """
        Jacobiano (6, num_joints) del TCP respecto a las articulaciones del brazo.
        Las tres primeras filas son la parte lineal; las tres ultimas, la angular.
        """
        self._end_effector_jacobian()
        return np.vstack([self._jacp[:, self._dofadr],
                          self._jacr[:, self._dofadr]])

    @property
    def arm_joint_position_indices(self):
        """Indices en `data.qpos` de las articulaciones del brazo."""
        return self._qadr

    # ------------------------------------------------------------------ #
    #  Control cartesiano (delegado en un CartesianController)
    # ------------------------------------------------------------------ #
    @property
    def controller(self):
        """
        Controlador cartesiano asociado al robot (creado de forma perezosa).

        La IK y el movimiento por delta de pose viven en `CartesianController`
        (ver `robots/control.py`), no en el robot:

            robot.controller.move_delta(dz=-0.05, frame="world")
            robot.controller.cartesian_step([0, 0, 0.01, 0, 0, 0])
        """
        if self._controller is None:
            from .control import CartesianController
            self._controller = CartesianController(self)
        return self._controller

    # ------------------------------------------------------------------ #
    #  Utilidades
    # ------------------------------------------------------------------ #
    def summary(self):
        """Imprime un resumen de la estructura del robot."""
        ee = (f"site '{self.end_effector_site}'" if self._end_effector_mode == "site"
              else f"cuerpo '{self.end_effector_body}' + offset {self.end_effector_offset}")
        print(f"Robot: '{self.name}'  ({self.xml_path})")
        print(f"  Articulaciones: {self.num_joints} | Actuadores: {self.num_actuators}"
              f" | Pinza: {'si' if self.has_gripper else 'no'}")
        print(f"  TCP: {ee}")
        rngs = self.joint_ranges
        for k, jn in enumerate(self.joint_names):
            lo, hi = rngs[k]
            act = self._joint_to_act.get(self.joint_ids[k])
            act_name = self.actuator_names[act] if act is not None else "-"
            print(f"    - {jn:10s} rango=[{lo:+.3f}, {hi:+.3f}] rad  "
                  f"actuador={act_name}")

    # ------------------------------------------------------------------ #
    #  Internos
    # ------------------------------------------------------------------ #
    def _default_end_effector_site(self):
        if self.model.nsite > 0:
            return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE,
                                     self.model.nsite - 1)
        return None

    def _require_end_effector(self):
        if self._end_effector_mode == "site" and self.end_effector_site_id < 0:
            raise ValueError(
                "El modelo no tiene un site de efector final definido. "
                "Pasa end_effector_site='<nombre>' o end_effector_body='<nombre>' al crear el robot."
            )
        if self._end_effector_mode == "body" and self.end_effector_body_id < 0:
            raise ValueError(f"No existe el cuerpo del efector '{self.end_effector_body}'.")
