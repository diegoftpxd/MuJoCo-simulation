"""
Controlador de OpenVLA: el wrapper que encapsula el modelo OpenVLA y mantiene la
interfaz común `Model` (`act(Observation) -> Action`).

Es lo único que sabe de `torch`/`transformers` (imports perezosos, para que el
modulo se pueda importar sin ellos). Selecciona la vista que necesita de la
`Observation`, predice el vector de 7 y lo devuelve como `core.Action`.
"""

import numpy as np

from core import Action, View
from models.Model import Model

DEFAULT_MODEL = "openvla/openvla-7b-finetuned-libero-10"


class OpenVLAController(Model):
    """Wrapper de OpenVLA que implementa la interfaz `Model`."""

    def __init__(self, model_id=DEFAULT_MODEL, unnorm_key=None, device="cuda",
                 view=View.AGENT, center_crop=True):
        """
        Parametros
        ----------
        model_id : str
            Repo de HuggingFace del checkpoint (por defecto, LIBERO-10).
        unnorm_key : str | None
            Clave para des-normalizar la accion. Si es None y el modelo trae una
            sola, se usa esa automaticamente.
        device : str
            "cuda" (Colab/Kaggle) o "cpu".
        view : View | str
            Que vista de la `Observation` usa este modelo.
        center_crop : bool
            Recorte central (los checkpoints fine-tuned se entrenaron con crop).
        """
        self.model_id = model_id
        self.device = device
        self.view = view
        self.center_crop = center_crop
        self._load()
        self.unnorm_key = self._resolve_unnorm_key(unnorm_key)

    # ------------------------------------------------------------------ #
    #  Interfaz comun (Model)
    # ------------------------------------------------------------------ #
    def act(self, observation) -> list:
        image = observation.image(self.view)          # el modelo ELIGE su vista
        if self.center_crop:
            image = self._center_crop(image)
        vec = self.predict_action(image, observation.instruction)
        # OpenVLA predice una sola accion -> chunk de 1.
        return [Action.from_cartesian(vec[:6], gripper=float(vec[6]), raw=vec)]

    # ------------------------------------------------------------------ #
    #  Bajo nivel: carga y prediccion cruda
    # ------------------------------------------------------------------ #
    def _load(self):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor
        print("Empiezo load")
        self._torch = torch
        # La T4 (Turing) no tiene bf16 nativo -> computar en fp16.
        self.compute_dtype = torch.float16

        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True)

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            torch_dtype=self.compute_dtype,
            device_map="auto",          # <-- reparte capas entre GPUs
        )

        self.model.eval()

    def predict_action(self, image, instruction):
        """Vector de accion crudo (7,) YA des-normalizado: [dx,dy,dz,dr,dp,dy,pinza]."""
        torch = self._torch
        image = self._to_pil(image)
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        inputs = self.processor(prompt, image).to(self.device, dtype=self.compute_dtype)
        with torch.no_grad():
            action = self.model.predict_action(
                **inputs, unnorm_key=self.unnorm_key, do_sample=False)
        return np.asarray(action, dtype=float).reshape(-1)

    # ------------------------------------------------------------------ #
    #  Internos
    # ------------------------------------------------------------------ #
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

    @staticmethod
    def _center_crop(img, frac=0.95):
        h, w = img.shape[:2]
        ch, cw = int(h * frac), int(w * frac)
        top, left = (h - ch) // 2, (w - cw) // 2
        return np.ascontiguousarray(img[top:top + ch, left:left + cw])

    @staticmethod
    def _to_pil(image):
        from PIL import Image
        if hasattr(image, "save"):          # ya es PIL.Image
            return image
        return Image.fromarray(np.asarray(image).astype(np.uint8))
