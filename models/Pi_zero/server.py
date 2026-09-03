"""
Servidor de inferencia de pi0 (Opcion A: frontera de proceso).

Carga la politica pi0 (lerobot) UNA sola vez y responde peticiones de accion por
HTTP. Corre en el entorno `pizero` (lerobot[pi]==0.4.0, torch 2.7, numpy 2.x).
El benchmark LIBERO lo consume desde OTRO entorno (`openvla`: numpy<2,
robosuite 1.4.1) via `PiZeroClient`, SIN compartir dependencias: por el cable
solo cruzan arrays serializados con numpy (ver `wire.py`).

Reutiliza `PiZeroController` tal cual (carga, armado del batch e inferencia
viven ahi); este modulo solo le pone una capa HTTP delante.

Uso (desde la raiz del repo, en el entorno pizero):
    python -m models.Pi_zero.server --host 0.0.0.0 --port 9000 --device cuda

Endpoints:
    GET  /health -> 200 "ok" cuando el modelo ya esta cargado.
    POST /reset  -> reinicia la cola interna de la politica (entre episodios).
    POST /act    -> body: Observation (npz); responde: chunk (H,7) (npz).

El acceso a la GPU se serializa con un lock: un servidor, una GPU, una
inferencia a la vez (ThreadingHTTPServer permite conexiones concurrentes, pero
la prediccion es exclusiva).
"""

import argparse
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from models.Pi_zero import wire
from models.Pi_zero.Pi_zero_controller import PiZeroController


class _Handler(BaseHTTPRequestHandler):
    controller: PiZeroController = None      # se setea en main()
    lock = threading.Lock()

    # ------------------------------ utilidades ------------------------------ #
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

    # -------------------------------- rutas --------------------------------- #
    def do_GET(self):
        if self.path == "/health":
            self._send(200, "ok", "text/plain")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        try:
            if self.path == "/reset":
                with self.lock:
                    self.controller.reset()
                self._send(200, "ok", "text/plain")
                return
            if self.path == "/act":
                obs = wire.load_observation(self._read_body())
                with self.lock:
                    chunk = self.controller.predict_action(obs)   # (H, 7) numpy
                self._send(200, wire.dump_chunk(chunk))
                return
            self._send(404, "not found", "text/plain")
        except Exception as exc:                    # noqa: BLE001
            traceback.print_exc()
            self._send(500, f"{type(exc).__name__}: {exc}", "text/plain")


def main():
    ap = argparse.ArgumentParser(description="Servidor de inferencia pi0 (lerobot).")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--view", default="agentview",
                    help="Vista principal (tercera persona) que usa pi0.")
    ap.add_argument("--model-id", default=None,
                    help="Repo HF del checkpoint (default: el de PiZeroController).")
    args = ap.parse_args()

    kwargs = dict(device=args.device, view=args.view)
    if args.model_id:
        kwargs["model_id"] = args.model_id

    print("Cargando pi0 (esto puede tardar)...", flush=True)
    _Handler.controller = PiZeroController(**kwargs)
    print(f"pi0 listo. Escuchando en http://{args.host}:{args.port}", flush=True)

    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("Servidor pi0 detenido.", flush=True)


if __name__ == "__main__":
    main()
