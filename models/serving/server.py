"""
Servidor de inferencia GENERICO: expone CUALQUIER `Model` por HTTP.

Un modelo (pi0, OpenVLA, ...) corre en su propio proceso/entorno y se sirve con
este mismo servidor. El benchmark lo consume desde otro entorno con `RemoteModel`
(ver client.py), sin compartir dependencias: por el cable solo cruzan
`Observation` y `List[Action]` serializados con numpy (ver wire.py).

El servidor NO conoce la forma de la accion: solo llama a `model.act(obs)` (la
interfaz comun) y serializa la lista de acciones que devuelva. Toda la logica
especifica del modelo (torch/lerobot/transformers, armado del batch, etc.) vive
dentro de su controller.

Uso (desde la raiz del repo, en el entorno del modelo):
    python -m models.serving.server --model pi0     --port 9000 --device cuda
    python -m models.serving.server --model openvla --port 9001 --device cuda

Endpoints:
    GET  /health -> 200 "ok" cuando el modelo ya esta cargado.
    POST /reset  -> reinicia el estado interno del modelo (entre episodios).
    POST /act    -> body: Observation (npz); responde: List[Action] (npz).
"""

import argparse
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from models.serving import wire


# --------------------------------------------------------------------------- #
#  Registro de modelos: nombre -> funcion que construye el controller.
#  Los imports son perezosos (dentro de cada builder) para que este modulo se
#  pueda importar sin torch/lerobot/transformers.
# --------------------------------------------------------------------------- #
def _build_pi0(args):
    from models.Pi_zero import PiZeroController
    kw = dict(device=args.device, view=args.view)
    if args.model_id:
        kw["model_id"] = args.model_id
    return PiZeroController(**kw)


def _build_openvla(args):
    from models.OpenVLA import OpenVLAController
    kw = dict(device=args.device, view=args.view, center_crop=args.center_crop)
    if args.model_id:
        kw["model_id"] = args.model_id
    if args.unnorm_key:
        kw["unnorm_key"] = args.unnorm_key
    return OpenVLAController(**kw)


def _build_random(args):
    from models.Random import RandomController
    return RandomController()


BUILDERS = {
    "pi0": _build_pi0,
    "openvla": _build_openvla,
    "random": _build_random,
}


# --------------------------------------------------------------------------- #
#  Servidor HTTP
# --------------------------------------------------------------------------- #
class _Handler(BaseHTTPRequestHandler):
    model = None                     # se setea en serve()
    lock = threading.Lock()          # una GPU -> una inferencia a la vez

    def _read_body(self) -> bytes:
        n = int(self.headers.get("Content-Length", 0) or 0)
        return self.rfile.read(n) if n else b""

    def _send(self, code, body=b"", ctype="application/octet-stream"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def log_message(self, *args):
        pass      # silencio: no ensuciar el log de SLURM con cada request

    def do_GET(self):
        if self.path == "/health":
            self._send(200, "ok", "text/plain")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        try:
            if self.path == "/reset":
                with self.lock:
                    self.model.reset()
                self._send(200, "ok", "text/plain")
                return
            if self.path == "/act":
                obs = wire.load_observation(self._read_body())
                with self.lock:
                    actions = self.model.act(obs)       # List[Action]
                self._send(200, wire.dump_actions(actions))
                return
            self._send(404, "not found", "text/plain")
        except Exception as exc:                        # noqa: BLE001
            traceback.print_exc()
            self._send(500, f"{type(exc).__name__}: {exc}", "text/plain")


def serve(model, host="127.0.0.1", port=9000):
    """Levanta el servidor HTTP para un `Model` ya cargado."""
    _Handler.model = model
    server = ThreadingHTTPServer((host, port), _Handler)
    print(f"Modelo listo. Escuchando en http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("Servidor detenido.", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Servidor de inferencia generico.")
    ap.add_argument("--model", required=True, choices=sorted(BUILDERS),
                    help="Que modelo servir.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--view", default="agentview",
                    help="Vista principal que usa el modelo.")
    ap.add_argument("--model-id", default=None,
                    help="Repo HF del checkpoint (default: el del controller).")
    # Opciones especificas de OpenVLA (pi0 las ignora):
    ap.add_argument("--unnorm-key", default=None)
    ap.add_argument("--center-crop", dest="center_crop",
                    action="store_true", default=True)
    ap.add_argument("--no-center-crop", dest="center_crop", action="store_false")
    args = ap.parse_args()

    print(f"Cargando modelo '{args.model}' (esto puede tardar)...", flush=True)
    model = BUILDERS[args.model](args)
    serve(model, args.host, args.port)


if __name__ == "__main__":
    main()
