"""
Cliente GENERICO: implementa la interfaz `Model` hablando con un servidor de
inferencia (models/serving/server.py) por HTTP.

Es AGNOSTICO al modelo: como por el cable solo cruzan `Observation` y
`List[Action]`, el mismo `RemoteModel` sirve para pi0, OpenVLA o cualquier otro
que se sirva con el servidor generico. Solo cambia la URL (el puerto del modelo).

Vive en el entorno del BENCHMARK, que NO necesita torch/lerobot/transformers:
`RemoteModel` depende solo de numpy + stdlib. Asi el notebook del benchmark no
arrastra las dependencias de ningun modelo.

Uso en el notebook del benchmark:
    from models.serving import RemoteModel
    model = RemoteModel(url="http://localhost:9000")   # 9000=pi0, 9001=openvla
    # ...luego identico: run_experiments(model, benchmarks, ...)
"""

import urllib.request

from models.Model import Model
from models.serving import wire


class RemoteModel(Model):
    """Proxy HTTP de un modelo servido. Misma interfaz que cualquier `Model`."""

    def __init__(self, url="http://localhost:9000", timeout=120.0):
        """
        Parametros
        ----------
        url : str
            Base del servidor de inferencia (host:puerto donde corre server.py).
        timeout : float
            Segundos maximos por request. La primera inferencia puede tardar;
            subelo si ves timeouts.
        """
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _post(self, path, body=b"") -> bytes:
        req = urllib.request.Request(self.url + path, data=body, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return resp.read()

    # ---------------------------- interfaz Model ---------------------------- #
    def reset(self):
        # Reinicia el estado interno del modelo remoto entre episodios.
        try:
            self._post("/reset")
        except Exception:                           # noqa: BLE001
            pass      # si el servidor no lo soporta no es critico

    def act(self, observation) -> list:
        blob = wire.dump_observation(observation)
        return wire.load_actions(self._post("/act", blob))
