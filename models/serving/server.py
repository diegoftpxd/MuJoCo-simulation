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
import inspect
import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from models import ModelFactory
from models.serving import wire


# --------------------------------------------------------------------------- #
#  Construccion del modelo desde el registro (models/factory.py).
#  El servidor es GENERICO: no conoce los args de cada modelo. Reune todos los
#  kwargs posibles del CLI y le pasa a cada controlador SOLO los que declara en
#  su __init__ (via inspeccion de la firma). Asi agregar un modelo con opciones
#  propias no obliga a tocar este archivo: basta registrarlo en ModelFactory y,
#  si su opcion no esta ya en el CLI, exponerla como un flag mas abajo.
# --------------------------------------------------------------------------- #
def _build_model(args):
    target = ModelFactory.get(args.model)      # resuelve la clase (import perezoso)
    candidates = {
        "device": args.device,
        "view": args.view,
        "model_id": args.model_id,
        "unnorm_key": args.unnorm_key,          # especifico de OpenVLA
        "center_crop": args.center_crop,        # especifico de OpenVLA
    }
    params = inspect.signature(target).parameters
    accepts_var_kw = any(p.kind is p.VAR_KEYWORD for p in params.values())
    # Pasa un kwarg solo si el controlador lo acepta (o acepta **kwargs) y no es
    # None (para no pisar los defaults del propio controlador, p. ej. model_id).
    kwargs = {k: v for k, v in candidates.items()
              if v is not None and (accepts_var_kw or k in params)}
    return target(**kwargs)


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
        elif self.path == "/capabilities":
            # Contrato de capacidades: el benchmark (en otro proceso) valida lo
            # que este modelo REQUIERE contra lo que el ofrece. Ver core/capabilities.py.
            body = json.dumps(self.model.requirements().to_dict())
            self._send(200, body, "application/json")
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
            if self.path == "/context":
                # Metadatos opcionales para el modelo (p. ej. step del entorno para
                # el logging). Generico: solo aplica si el modelo tiene set_context.
                body = self._read_body()
                ctx = json.loads(body.decode("utf-8")) if body else {}
                setter = getattr(self.model, "set_context", None)
                if callable(setter):
                    with self.lock:
                        setter(**ctx)
                self._send(200, "ok", "text/plain")
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
    ap.add_argument("--model", required=True, choices=ModelFactory.available(),
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
    model = _build_model(args)
    serve(model, args.host, args.port)


if __name__ == "__main__":
    main()
