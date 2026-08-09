"""
Controlador de OpenVLA: encapsula TODA la interaccion con el modelo.

Convierte `(imagen, instruccion)` en una `CartesianAction` (Δpose + pinza), que
luego se aplica al robot con `robot.controller.cartesian_step(...)`. Asi el
modelo queda desacoplado de la simulacion: este archivo es lo unico que sabe de
`torch`/`transformers`.

Las dependencias pesadas (torch, transformers, bitsandbytes) se importan de
forma PEREZOSA, dentro de los metodos, para que importar este modulo no falle en
un entorno que solo tenga la simulacion (p. ej. tu entorno `tm5` local).

Uso tipico (en Colab, con GPU):

    from openvla_controller import OpenVLAController
    vla = OpenVLAController(load_in_4bit=True)          # T4-friendly
    action = vla.predict(image_rgb, "pick up the object")   # -> CartesianAction
    robot.controller.cartesian_step(action, frame=vla.frame)
"""

import numpy as np

from robots import CartesianAction, Frame, PoseDelta

DEFAULT_MODEL = "openvla/openvla-7b-finetuned-libero-10"


class OpenVLAController:
    """Envuelve un modelo OpenVLA y produce acciones cartesianas."""

    def __init__(self, model_id=DEFAULT_MODEL, unnorm_key=None,
                 device="cuda", load_in_4bit=True,
                 frame=Frame.WORLD, gripper_threshold=0.5,
                 gripper_open_is_high=True):
        """
        Parametros
        ----------
        model_id : str
            Repo de HuggingFace del checkpoint (por defecto, LIBERO-10).
        unnorm_key : str | None
            Clave de estadisticas para des-normalizar la accion. Si es None y el
            modelo trae una sola, se usa esa automaticamente.
        device : str
            "cuda" (Colab) o "cpu".
        load_in_4bit : bool
            Carga cuantizada a 4-bit (NF4). Necesario para caber en una T4 16 GB.
        frame : Frame
            Marco en el que OpenVLA entrega los deltas (a calibrar; ver notas).
        gripper_threshold : float
            Umbral del 7º valor para decidir abrir/cerrar la pinza.
        gripper_open_is_high : bool
            Si el valor alto significa "abrir" (True) o "cerrar" (False).
            Es una convencion del dataset; hay que calibrarla.
        """
        self.model_id = model_id
        self.device = device
        self.frame = Frame(frame)
        self.gripper_threshold = gripper_threshold
        self.gripper_open_is_high = gripper_open_is_high
        self._load(load_in_4bit)
        self.unnorm_key = self._resolve_unnorm_key(unnorm_key)

    # ------------------------------------------------------------------ #
    #  Carga del modelo (imports perezosos)
    # ------------------------------------------------------------------ #
    def _load(self, load_in_4bit):
        import torch
        from transformers import (AutoModelForVision2Seq, AutoProcessor,
                                   BitsAndBytesConfig)

        self._torch = torch
        # La T4 (Turing) no tiene bf16 nativo -> computar en fp16.
        self.compute_dtype = torch.float16

        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True)

        kwargs = dict(
            attn_implementation="sdpa",   # no hay flash-attn en la T4
            low_cpu_mem_usage=True,       # la RAM de Colab es justa al cargar 8B
            trust_remote_code=True,
        )
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["torch_dtype"] = self.compute_dtype

        self.model = AutoModelForVision2Seq.from_pretrained(self.model_id, **kwargs)
        if not load_in_4bit:
            self.model = self.model.to(self.device)
        self.model.eval()

    def _resolve_unnorm_key(self, key):
        stats = getattr(self.model, "norm_stats", None)
        if key is not None:
            return key
        if stats and len(stats) == 1:
            return next(iter(stats))
        raise ValueError(
            "No pude deducir `unnorm_key`. Pasalo explicitamente. "
            f"Claves disponibles: {list(stats) if stats else 'desconocidas'}"
        )

    # ------------------------------------------------------------------ #
    #  Prediccion
    # ------------------------------------------------------------------ #
    def predict_action(self, image, instruction):
        """
        Devuelve el vector de accion crudo (7,) YA des-normalizado:
        [dx, dy, dz, droll, dpitch, dyaw, pinza].
        """
        torch = self._torch
        image = self._to_pil(image)
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        inputs = self.processor(prompt, image).to(self.device, dtype=self.compute_dtype)
        with torch.no_grad():
            action = self.model.predict_action(
                **inputs, unnorm_key=self.unnorm_key, do_sample=False)
        return np.asarray(action, dtype=float).reshape(-1)

    def predict(self, image, instruction):
        """
        Devuelve una `CartesianAction` lista para
        `robot.controller.cartesian_step(action, frame=vla.frame)`.
        """
        return self._to_cartesian_action(self.predict_action(image, instruction))

    # ------------------------------------------------------------------ #
    #  Internos
    # ------------------------------------------------------------------ #
    def _to_cartesian_action(self, vector):
        delta = PoseDelta.from_sequence(vector[:6])
        gripper_open = None
        if vector.size >= 7:
            high = vector[6] > self.gripper_threshold
            gripper_open = bool(high) if self.gripper_open_is_high else bool(not high)
        return CartesianAction(delta, gripper_open)

    @staticmethod
    def _to_pil(image):
        from PIL import Image
        if hasattr(image, "save"):          # ya es PIL.Image
            return image
        return Image.fromarray(np.asarray(image).astype(np.uint8))
