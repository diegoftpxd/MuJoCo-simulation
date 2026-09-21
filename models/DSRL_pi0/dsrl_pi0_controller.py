"""
Controlador de DSRL-pi0 (Diffusion Steering via Latent-space RL) para LIBERO.

DSRL (Nakamoto et al., CoRL 2025 -- https://github.com/nakamotoo/dsrl_pi0) NO
reentrena la politica base: aprende, con RL (SAC), una politica "actor" que elige
el RUIDO INICIAL en el espacio latente de una politica de flow-matching base
(pi0 de openpi). En inferencia, el actor produce el ruido y pi0 lo integra
("denoisa") hasta una accion. Es la version APRENDIDA de lo que en
models/Pi_zero/patches.py se hacia a mano con `noise_mean`/`noise_scale`.

STACK PROPIO (aislamiento por proceso): a diferencia de `PiZeroController`
(lerobot / PyTorch), DSRL usa JAX + openpi + jaxrl2. Ademas usa un FORK de openpi
(el submodulo del repo) que anade el argumento `noise=` a `Policy.infer`; el
openpi de PyPI no lo acepta. Por eso vive en su propio entorno conda
(`requirements/environment-dsrl.yml`) y se sirve por HTTP con `models/serving/`.

FLUJO DE INFERENCIA (copiado de examples/train_utils_sim.py del repo):

    obs_dict = {"pixels": <img 64x64>, "state": <8,>}      # entrada del actor SAC
    noise    = agent.sample_actions(obs_dict)              # ruido latente aplanado
    noise    = noise.reshape((1, *agent.action_chunk_shape))   # (1, 50, 32)
    obs_pi0  = {"observation/image", "observation/wrist_image",
                "observation/state", "prompt"}             # convencion openpi
    actions  = agent_dp.infer(obs_pi0, noise=noise)["actions"]  # (chunk, 7)

El actor re-muestrea el ruido cada `query_freq` pasos; por eso `act` devuelve un
chunk de `query_freq` acciones y el runner (`run_episode`) vuelve a pedir cuando
se agota -> ahi se re-muestrea el ruido y se re-infiere.

CARGA (examples/train_sim.py del repo):
  - Base pi0 (openpi):
        config = openpi_config.get_config("pi0_libero")
        ckpt   = download.maybe_download("s3://openpi-assets/checkpoints/pi0_libero")
        agent_dp = policy_config.create_trained_policy(config, ckpt)
  - Actor DSRL (jaxrl2):
        agent = PixelSACLearner(seed, sample_obs, sample_action, **kwargs)
        agent.restore_checkpoint(agent_checkpoint)

ACCION: openpi para LIBERO emite 7 dims [dx,dy,dz,droll,dpitch,dyaw, pinza] en la
convencion NATIVA de LIBERO (igual que pi0-lerobot), asi que `LiberoController.step`
la entiende tal cual.
"""

import numpy as np

from core import Action, Capabilities, View
from models.Model import Model

# Base pi0 de openpi para LIBERO (checkpoint publico de openpi-assets).
DEFAULT_DP_CONFIG = "pi0_libero"
DEFAULT_DP_CHECKPOINT = "s3://openpi-assets/checkpoints/pi0_libero"


class DSRLPi0Controller(Model):
    """Wrapper de DSRL-pi0 (actor SAC en el ruido + base pi0) que implementa `Model`."""

    def __init__(self, agent_checkpoint, dp_config=DEFAULT_DP_CONFIG,
                 dp_checkpoint=DEFAULT_DP_CHECKPOINT, device="cuda",
                 view=View.AGENT, wrist_view=View.WRIST,
                 query_freq=20, image_size=64, seed=0, agent_kwargs=None):
        """
        Parametros
        ----------
        agent_checkpoint : str
            Directorio del checkpoint del actor DSRL (SAC) entrenado para la tarea
            (lo que `PixelSACLearner.restore_checkpoint` espera). OBLIGATORIO: es lo
            unico especifico de DSRL; sin el, esto seria pi0 base.
        dp_config : str
            Nombre de la config de openpi para la base pi0 (por defecto "pi0_libero").
        dp_checkpoint : str
            Ruta/URI del checkpoint de la base pi0 (por defecto el de openpi-assets;
            se descarga con `download.maybe_download`).
        device : str
            "cuda" o "cpu" (JAX toma el device por su cuenta; se expone por simetria).
        view / wrist_view : View | str
            Vista principal (tercera persona) y de muñeca de la `Observation`.
        query_freq : int
            Pasos que se ejecutan del chunk antes de re-muestrear el ruido y
            re-inferir (del repo: --query_freq 20). Es el tamano del chunk que
            devuelve `act`.
        image_size : int
            Lado al que se redimensiona la imagen para el ENCODER del actor SAC
            (del repo: --resize_image 64). No afecta a la base pi0 (openpi
            redimensiona internamente).
        seed : int
            Semilla del actor SAC.
        agent_kwargs : dict | None
            Hiperparametros extra para `PixelSACLearner` (hidden_dims, action_magnitude,
            num_cameras, ...). DEBEN coincidir con los del entrenamiento (ver
            examples/scripts/run_libero.sh) para que las redes calcen con el checkpoint.
        """
        self.agent_checkpoint = agent_checkpoint
        self.dp_config = dp_config
        self.dp_checkpoint = dp_checkpoint
        self.device = device
        self.view = view
        self.wrist_view = wrist_view
        self.query_freq = int(query_freq)
        self.image_size = int(image_size)
        self.seed = int(seed)
        self.agent_kwargs = dict(agent_kwargs or {})
        self._load()

    # ------------------------------------------------------------------ #
    #  Interfaz comun (Model)
    # ------------------------------------------------------------------ #
    def requirements(self) -> Capabilities:
        # Igual que pi0: vista principal + estado de 8 dims (eef_pos, eef_quat,
        # gripper_qpos) + instruccion. La muñeca es opcional (se usa si esta).
        return Capabilities.of(
            views=[self.view],
            state=["eef_pos", "eef_quat", "gripper_qpos"],
            instruction=True)

    def act(self, observation) -> list:
        # 1) El actor DSRL elige el ruido latente a partir de su observacion.
        noise = self._sample_noise(observation)
        # 2) La base pi0 (openpi) integra ese ruido -> chunk de acciones.
        obs_pi0 = self._build_pi0_input(observation)
        actions = self.policy_dp.infer(obs_pi0, noise=noise)["actions"]
        actions = np.asarray(actions, dtype=float)          # (chunk, 7)
        # 3) Se devuelven `query_freq` acciones; al agotarse, el runner vuelve a
        #    pedir -> se re-muestrea el ruido (nuevo `act`).
        return [self._to_action(a) for a in actions[:self.query_freq]]

    def reset(self):
        # El actor SAC no mantiene cola interna (el chunk lo bufferea el runner);
        # no hay estado que limpiar entre episodios.
        pass

    # ------------------------------------------------------------------ #
    #  Bajo nivel: carga
    # ------------------------------------------------------------------ #
    def _load(self):
        # --- Base pi0 (openpi). Fork del repo: su Policy.infer acepta `noise=`. ---
        from openpi.training import config as openpi_config
        from openpi.policies import policy_config
        from openpi.shared import download
        print("Empiezo load (DSRL-pi0): base pi0 (openpi)...", flush=True)
        config = openpi_config.get_config(self.dp_config)
        checkpoint_dir = download.maybe_download(self.dp_checkpoint)
        self.policy_dp = policy_config.create_trained_policy(config, checkpoint_dir)

        # --- Actor DSRL (jaxrl2 SAC) + restauracion del checkpoint entrenado. ---
        from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
        print("Cargando actor DSRL (SAC)...", flush=True)
        sample_obs, sample_action = self._sample_spaces()
        self.agent = PixelSACLearner(self.seed, sample_obs, sample_action,
                                     **self.agent_kwargs)
        self.agent.restore_checkpoint(self.agent_checkpoint)
        # Forma del ruido que espera pi0: (chunk_len, action_dim), p. ej. (50, 32).
        self.action_chunk_shape = self.agent.action_chunk_shape
        print("DSRL-pi0 listo.", flush=True)

    def _sample_spaces(self):
        """
        Observacion y accion de MUESTRA para construir las redes del actor antes de
        restaurar sus pesos (jaxrl2 infiere las shapes de estos ejemplos).

        NOTA: las shapes/keys DEBEN coincidir con las del entrenamiento DSRL (ver
        `obs_dict` y el action space en examples/train_sim.py /
        examples/train_utils_sim.py del repo). Aqui se usan los defaults de LIBERO
        (imagen `image_size`, estado de 8 dims, accion-ruido de 32). Si el checkpoint
        no calza, ajusta esto (o pasa `agent_kwargs`) segun tu config de entrenamiento.
        """
        n = self.image_size
        sample_obs = {
            "pixels": np.zeros((n, n, 3, 1), dtype=np.uint8),   # HWC + 1 cámara
            "state": np.zeros((8, 1), dtype=np.float32),
        }
        # El actor actua en el espacio de ruido de pi0 (Box(-1,1, shape=(1,32))).
        sample_action = np.zeros((1, 32), dtype=np.float32)
        return sample_obs, sample_action

    # ------------------------------------------------------------------ #
    #  Bajo nivel: armado de entradas e inferencia del ruido
    # ------------------------------------------------------------------ #
    def _sample_noise(self, observation):
        """Ruido latente del actor DSRL, con la forma que espera `pi0.infer`."""
        obs_dict = self._build_agent_input(observation)
        noise = np.asarray(self.agent.sample_actions(obs_dict))
        # (1, chunk_len, action_dim), p. ej. (1, 50, 32).
        return noise.reshape((1, *self.action_chunk_shape))

    def _build_agent_input(self, observation):
        """
        Observacion para el ACTOR SAC: imagen redimensionada + estado.

        NOTA: el layout exacto (nombre de las keys, dim de cámaras/frames) depende
        del encoder del actor; sigue `obs_dict` de examples/train_utils_sim.py. Aqui:
          - "pixels": vista principal redimensionada a (image_size, image_size, 3, 1).
          - "state" : el mismo vector de 8 dims de openpi, como (8, 1).
        """
        img = self._resize(observation.image(self.view), self.image_size)
        return {
            "pixels": img[..., None].astype(np.uint8),          # (H, W, 3, 1)
            "state": self._state_vector(observation).reshape(8, 1).astype(np.float32),
        }

    def _build_pi0_input(self, observation):
        """
        Observacion para la BASE pi0 (convencion openpi), como en
        `obs_to_pi_zero_input` del repo: imagenes uint8 HWC, estado de 8 dims y el
        prompt como string.
        """
        obs = {
            "observation/image": np.asarray(observation.image(self.view), dtype=np.uint8),
            "observation/state": self._state_vector(observation).astype(np.float32),
            "prompt": observation.instruction,
        }
        if observation.has_image(self.wrist_view):     # la muñeca es opcional
            obs["observation/wrist_image"] = np.asarray(
                observation.image(self.wrist_view), dtype=np.uint8)
        return obs

    def _to_action(self, vec):
        """Vector de 7 (openpi/LIBERO) -> `Action` (6 deltas + pinza nativa)."""
        vec = np.asarray(vec, dtype=float).reshape(-1)
        return Action.from_cartesian(vec[:6], gripper=float(vec[6]), raw=vec)

    # ------------------------------------------------------------------ #
    #  Utilidades (mismas convenciones que PiZeroController)
    # ------------------------------------------------------------------ #
    def _state_vector(self, observation):
        """
        Estado propioceptivo de 8 dims (convencion openpi/pi0-LIBERO):
            [ eef_pos (3), orientacion en axis-angle (3), gripper_qpos (2) ]
        La orientacion llega como cuaternion (x, y, z, w) y se pasa a axis-angle.
        """
        eef_pos = observation.get_state("eef_pos")
        eef_quat = observation.get_state("eef_quat")
        gripper = observation.get_state("gripper_qpos")
        if eef_pos is None or eef_quat is None or gripper is None:
            raise ValueError(
                "DSRL-pi0/LIBERO espera estado de 8 dims [eef_pos(3), axis_angle(3), "
                "gripper_qpos(2)]. Falta 'eef_pos', 'eef_quat' o 'gripper_qpos' en la "
                "observacion; revisa LiberoController._to_observation.")
        return np.concatenate([
            np.asarray(eef_pos, dtype=float).reshape(-1),
            self._quat2axisangle(np.asarray(eef_quat, dtype=float).reshape(-1)),
            np.asarray(gripper, dtype=float).reshape(-1),
        ])

    @staticmethod
    def _quat2axisangle(quat):
        """Cuaternion (x, y, z, w) -> axis-angle (3). Igual que robosuite/pi0."""
        w = float(np.clip(quat[3], -1.0, 1.0))
        den = np.sqrt(1.0 - w * w)
        if den < 1e-8:
            return np.zeros(3)
        return (quat[:3] * 2.0 * np.arccos(w)) / den

    @staticmethod
    def _resize(img, size):
        """Redimensiona una imagen HWC uint8 a (size, size). Usa PIL (perezoso)."""
        img = np.asarray(img)
        if img.shape[0] == size and img.shape[1] == size:
            return img
        from PIL import Image
        return np.asarray(Image.fromarray(img.astype(np.uint8)).resize((size, size)))
