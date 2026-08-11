"""
Clase base abstracta de los objetos del entorno.

`SceneObject` define cómo un objeto se agrega al modelo y cómo se consulta/mueve
su pose, pero NO su geometría. Cada tipo concreto (en su propio archivo del
paquete: `cube.py`, `sphere.py`, ...) implementa `_add_geom` con su forma.

En MuJoCo los cuerpos deben existir ANTES de compilar el modelo, así que los
objetos se agregan sobre un `mujoco.MjSpec` (editor de modelos) y luego se
compila. Eso lo orquesta `RobotArm(objects=[...])`; aquí solo vive la definición.
"""

from abc import ABC, abstractmethod

import mujoco


class SceneObject(ABC):
    """Objeto abstracto del entorno. Subclasear e implementar `_add_geom`."""

    def __init__(self, name, position=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0),
                 rgba=(0.8, 0.2, 0.2, 1.0), mass=0.1, movable=True):
        """
        Parametros
        ----------
        name : str
            Nombre unico del cuerpo en el modelo.
        position : (3,)
            Posicion inicial (x, y, z) en metros.
        quat : (4,)
            Orientacion inicial (w, x, y, z).
        rgba : (4,)
            Color.
        mass : float
            Masa en kg (solo relevante si es movable).
        movable : bool
            True: cuerpo libre (con free joint), se puede agarrar/mover.
            False: cuerpo fijo (obstaculo/mesa).
        """
        self.name = name
        self.position = tuple(float(x) for x in position)
        self.quat = tuple(float(x) for x in quat)
        self.rgba = tuple(float(x) for x in rgba)
        self.mass = float(mass)
        self.movable = bool(movable)

        # Se rellenan en `bind`, tras compilar el modelo.
        self._model = None
        self._data = None
        self._body_id = -1
        self._qadr = None

    # ------------------------------------------------------------------ #
    #  A implementar por cada tipo concreto
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _add_geom(self, body):
        """Agrega la(s) geometria(s) del objeto al cuerpo `body` (MjsBody)."""

    # ------------------------------------------------------------------ #
    #  Construccion (sobre un MjSpec, antes de compilar)
    # ------------------------------------------------------------------ #
    def add_to_spec(self, spec):
        """Agrega el cuerpo del objeto al `mujoco.MjSpec` dado."""
        body = spec.worldbody.add_body(name=self.name, pos=self.position,
                                       quat=self.quat)
        if self.movable:
            body.add_freejoint()
        self._add_geom(body)
        return body

    @property
    def init_qpos(self):
        """qpos inicial del objeto (pos+quat si es movable; vacío si es fijo)."""
        return list(self.position) + list(self.quat) if self.movable else []

    @property
    def n_dof(self):
        """Grados de libertad que aporta al estado (6 si movable, 0 si fijo)."""
        return 6 if self.movable else 0

    # ------------------------------------------------------------------ #
    #  Enlace al modelo compilado (lo llama RobotArm)
    # ------------------------------------------------------------------ #
    def bind(self, model, data):
        """Guarda referencias al modelo/datos ya compilados."""
        self._model, self._data = model, data
        self._body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.name)
        if self.movable:
            for j in range(model.njnt):
                if (model.jnt_bodyid[j] == self._body_id
                        and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE):
                    self._qadr = int(model.jnt_qposadr[j])
                    break
        return self

    # ------------------------------------------------------------------ #
    #  Consulta / control de la pose
    # ------------------------------------------------------------------ #
    def get_pose(self):
        """Devuelve (posicion (3,), quat (4,)) actuales del objeto en el mundo."""
        return (self._data.xpos[self._body_id].copy(),
                self._data.xquat[self._body_id].copy())

    def set_pose(self, position=None, quat=None):
        """Reubica el objeto (solo si es movable). Actualiza la cinemática."""
        if not self.movable:
            raise ValueError(f"El objeto '{self.name}' es fijo; no se puede mover.")
        adr = self._qadr
        if position is not None:
            self._data.qpos[adr:adr + 3] = position
        if quat is not None:
            self._data.qpos[adr + 3:adr + 7] = quat
        mujoco.mj_forward(self._model, self._data)
        return self
