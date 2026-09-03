"""
Controlador de Pi-zero (pi0): wrapper de la politica `PI0Policy` de LeRobot que
implementa la interfaz `Model` devolviendo un CHUNK de acciones.

A diferencia de OpenVLA (que es un modelo de `transformers`), pi0 se carga con
lerobot y vive en un ENTORNO SEPARADO (ver `requirements/environment-pizero.yml`): sus
dependencias chocan con el stack de OpenVLA. pi0 predice un HORIZONTE de acciones
(action chunking) mediante flow-matching.

FLUJO DE INFERENCIA (lerobot 0.4.0): la normalizacion de entradas, la
des-normalizacion de salida y la TOKENIZACION del lenguaje ya NO viven en la
politica, sino en un pipeline de processors (`make_pre_post_processors`). Por eso
seguimos el ejemplo oficial: `preprocess(frame) -> select_action -> postprocess`.
`select_action` mantiene su propia COLA de chunk dentro de la politica (solo
re-ejecuta el modelo cada `n_action_steps`), asi que `act` devuelve UNA accion por
llamada y el runner vuelve a pedir en el siguiente paso; no se recomputa el modelo
en cada paso. `reset` limpia esa cola entre episodios.

Checkpoint por defecto: 'lerobot/pi0_libero_finetuned' (~4B, fine-tuneado en
LIBERO). Se carga con `PI0Policy.from_pretrained(...)`.

IMPORTANTE — cosas que dependen del checkpoint y hay que VERIFICAR contra su
config (`policy.config.input_features` / `output_features`):
  - Los nombres de las claves de imagen/estado del batch (parametrizados abajo).
  - El vector de estado que espera: el checkpoint de LIBERO usa un estado de
    ~8 dims. `LiberoController` hoy solo expone `eef_pos` (3) y `gripper_qpos`;
    puede que tengas que ampliarlo para exponer la orientacion del efector y
    armar el estado completo. Ajusta `_state_vector` en consecuencia.
  - pi0-LIBERO produce acciones de 7 dims [dx,dy,dz,droll,dpitch,dyaw, pinza],
    que ya es la convencion que `LiberoController.step` entiende.
"""

import numpy as np

from core import Action, View
from models.Model import Model

DEFAULT_MODEL = "lerobot/pi0_libero_finetuned"


class PiZeroController(Model):
    """Wrapper de pi0 (LeRobot `PI0Policy`) que implementa la interfaz `Model`."""

    def __init__(self, model_id=DEFAULT_MODEL, device="cuda",
                 view=View.AGENT, wrist_view=View.WRIST,
                 image_key="observation.images.image",
                 wrist_key="observation.images.wrist_image",
                 state_key="observation.state"):
        """
        Parametros
        ----------
        model_id : str
            Repo de HuggingFace del checkpoint (por defecto, pi0 LIBERO).
        device : str
            "cuda" o "cpu".
        view : View | str
            Vista principal (tercera persona) que usa el modelo.
        wrist_view : View | str
            Vista de muñeca; se incluye solo si la observacion la ofrece.
        image_key, wrist_key, state_key : str
            Nombres de las claves del batch que espera `PI0Policy`. DEBEN
            coincidir con `policy.config.input_features` del checkpoint.
        """
        self.model_id = model_id
        self.device = device
        self.view = view
        self.wrist_view = wrist_view
        self.image_key = image_key
        self.wrist_key = wrist_key
        self.state_key = state_key
        self._load()

    # ------------------------------------------------------------------ #
    #  Interfaz comun (Model)
    # ------------------------------------------------------------------ #
    def act(self, observation) -> list:
        # Camino oficial (examples/tutorial/pi0): preprocess -> select_action ->
        # postprocess. Devolvemos UNA accion; `select_action` sirve de su cola
        # interna y solo re-ejecuta el modelo cada n_action_steps.
        frame = self._build_frame(observation)
        batch = self._preprocess(frame)               # normaliza + tokeniza + device
        action = self.policy.select_action(batch)     # (1, action_dim) normalizado
        action = self._postprocess(action)            # des-normaliza y pasa a cpu
        a = np.asarray(action.squeeze(0).float().cpu().numpy(), dtype=float)  # (7,)
        return [Action.from_cartesian(a[:6], gripper=float(a[6]), raw=a)]

    def reset(self):
        # Limpia la cola de acciones interna de la politica entre episodios.
        if getattr(self, "policy", None) is not None and hasattr(self.policy, "reset"):
            self.policy.reset()

    # ------------------------------------------------------------------ #
    #  Bajo nivel: carga e inferencia
    # ------------------------------------------------------------------ #
    def _load(self):
        import functools
        import torch
        # Shim de torch.load: lerobot 0.4.0 trae torch >= 2.6, cuyo torch.load
        # usa weights_only=True por defecto y rompe la carga del checkpoint
        # (pickle con numpy). Restauramos el comportamiento anterior. Seguro
        # aqui: el checkpoint es de fuente confiable. Debe ir ANTES de cargar.
        if not getattr(torch.load, "_pizero_shimmed", False):
            torch.load = functools.partial(torch.load, weights_only=False)
            torch.load._pizero_shimmed = True
        from lerobot.policies.pi0 import PI0Policy
        from lerobot.policies.factory import make_pre_post_processors
        print("Empiezo load (pi0)")
        self._torch = torch
        self.policy = (PI0Policy.from_pretrained(self.model_id)
                       .to(self.device).eval())
        # Pipeline de processors (lerobot 0.4.0): normaliza/tokeniza las entradas
        # (preprocess) y des-normaliza la salida (postprocess). Carga las stats
        # de normalizacion desde el propio checkpoint (self.model_id).
        self._preprocess, self._postprocess = make_pre_post_processors(
            self.policy.config, self.model_id,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}})

    # ------------------------------------------------------------------ #
    #  Internos: armado del frame CRUDO que consume el preprocess
    # ------------------------------------------------------------------ #
    def _build_frame(self, observation):
        """
        Frame crudo que consume el `preprocess` de pi0 (mismo formato que
        `build_inference_frame` del ejemplo oficial): imagenes (1, 3, H, W) en
        float [0, 1], estado (1, D) float, y `task` como STRING (no lista). El
        `preprocess` se encarga de normalizar, tokenizar el lenguaje y mover a
        device; `predict_action_chunk` espera todo eso ya hecho.
        """
        torch = self._torch
        frame = {
            self.image_key: self._to_chw(observation.image(self.view)),
            self.state_key: torch.as_tensor(
                self._state_vector(observation), dtype=torch.float32,
                device=self.device).unsqueeze(0),
            "task": observation.instruction,   # instruccion en lenguaje natural
        }
        # La vista de muñeca es opcional: solo si el benchmark la ofrece.
        if observation.has_image(self.wrist_view):
            frame[self.wrist_key] = self._to_chw(observation.image(self.wrist_view))
        return frame

    def _to_chw(self, img):
        """(H, W, 3) uint8 -> tensor (1, 3, H, W) float en [0, 1]."""
        torch = self._torch
        t = torch.as_tensor(np.asarray(img), device=self.device)
        return (t.permute(2, 0, 1).float() / 255.0).unsqueeze(0)

    def _state_vector(self, observation):
        """
        Arma el vector de estado (propriocepcion) que espera el checkpoint.

        OJO: DEBE coincidir en dimension y orden con el estado con que se entreno
        pi0-LIBERO (~8 dims). Lo que `LiberoController` expone hoy (`eef_pos` +
        `gripper_qpos`) puede NO ser suficiente; si el checkpoint espera tambien
        la orientacion del efector, amplia `LiberoController._to_observation`
        para incluirla y ajusta este armado.
        """
        parts = []
        eef_pos = observation.get_state("eef_pos")
        if eef_pos is not None:
            parts.append(np.asarray(eef_pos, dtype=float).reshape(-1))
        gripper = observation.get_state("gripper_qpos")
        if gripper is not None:
            parts.append(np.asarray(gripper, dtype=float).reshape(-1))
        if not parts:
            raise ValueError(
                "La observacion no trae estado propioceptivo para pi0. "
                "Revisa que el benchmark exponga 'eef_pos'/'gripper_qpos' y que "
                "el estado coincida con lo que espera el checkpoint.")
        return np.concatenate(parts)
