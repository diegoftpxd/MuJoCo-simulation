"""
Cliente de pi0 (Opcion A): implementa la interfaz `Model` hablando con el
servidor de inferencia (`models/Pi_zero/server.py`) por HTTP.

Vive en el entorno del BENCHMARK (`openvla`: numpy<2, robosuite 1.4.1, el mismo
que ya corre LIBERO). NO importa lerobot ni torch: solo numpy + stdlib. Asi el
benchmark y el modelo no comparten NINGUNA dependencia; la unica frontera es la
red. Es un reemplazo directo (drop-in) de `PiZeroController` en el notebook:

    from models.Pi_zero import PiZeroClient
    model = PiZeroClient(url="http://localhost:9000")
    # ...luego identico: run_experiments(model, benchmarks, ...)

El post-proceso (chunk -> lista de Action) es IDENTICO al de PiZeroController,
para que el runner no note la diferencia.
"""

import urllib.request

from core import Action
from models.Model import Model
from models.Pi_zero import wire


class PiZeroClient(Model):
    """Proxy de pi0 sobre HTTP. Misma interfaz que `PiZeroController`."""

    def __init__(self, url="http://localhost:9000", timeout=120.0):
        """
        Parametros
        ----------
        url : str
            Base del servidor de inferencia (host:puerto donde corre server.py).
        timeout : float
            Segundos maximos por request. La primera inferencia puede tardar
            (compilacion/carga perezosa); subelo si ves timeouts.
        """
        self.url = url.rstrip("/")
        self.timeout = timeout

    # ------------------------------- interno -------------------------------- #
    def _post(self, path, body=b"") -> bytes:
        req = urllib.request.Request(self.url + path, data=body, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return resp.read()

    # ---------------------------- interfaz Model ---------------------------- #
    def reset(self):
        # Limpia la cola de acciones interna de la politica entre episodios.
        try:
            self._post("/reset")
        except Exception:                           # noqa: BLE001
            pass      # si el servidor no lo soporta no es critico

    def act(self, observation) -> list:
        blob = wire.dump_observation(observation)
        chunk = wire.load_chunk(self._post("/act", blob))    # (H, 7)
        # Mismo desempaquetado que PiZeroController.act: chunk entero -> Actions.
        return [Action.from_cartesian(a[:6], gripper=float(a[6]), raw=a)
                for a in chunk]
