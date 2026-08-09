"""
Simulacion y visualizacion de un robot en MuJoCo.

`Simulation` recibe un `RobotArm` como atributo y se encarga de TODO lo que no
es el robot en si: avanzar la fisica, abrir el visor interactivo, renderizar
imagenes y grabar videos. Asi el robot y la simulacion quedan desacoplados: el
mismo robot puede simularse, grabarse o analizarse sin que la clase del robot
sepa nada de camaras ni de video.

Ejemplo
-------
    from robots import RobotFactory
    from simulation import Simulation

    sim = Simulation(RobotFactory.create("tm5-700"))
    sim.render_image("foto.png")
    sim.launch_viewer()
"""

import os

import mujoco

from robots import Frame, RobotFactory

# Carpeta donde se guardan imagenes y videos por defecto.
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(_HERE, "output")


class Simulation:
    """Orquesta la simulacion y visualizacion de un robot."""

    def __init__(self, robot, output_dir=None):
        """
        Parametros
        ----------
        robot : RobotArm
            El robot a simular. Queda accesible como `sim.robot`.
        output_dir : str | None
            Carpeta donde guardar imagenes y videos cuando se pasa solo un
            nombre de archivo (sin ruta). Por defecto, `./output`.
        """
        self.robot = robot
        self.output_dir = output_dir or DEFAULT_OUTPUT_DIR
        # Renderer persistente para observaciones en lazo (ver `observation`).
        self._renderer = None
        self._render_size = None

    @classmethod
    def from_name(cls, name="tm5-700", output_dir=None):
        """Crea la simulacion directamente a partir del nombre de un robot."""
        return cls(RobotFactory.create(name), output_dir=output_dir)

    # Accesos directos al modelo/estado del robot, por comodidad.
    @property
    def model(self):
        return self.robot.model

    @property
    def data(self):
        return self.robot.data

    # ------------------------------------------------------------------ #
    #  Avance de la fisica
    # ------------------------------------------------------------------ #
    def step(self, n=1):
        """Avanza la simulacion `n` pasos."""
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
        return self

    def reset(self, keyframe=None):
        """Reinicia el robot a un keyframe (delega en el robot)."""
        self.robot.reset(keyframe)
        return self

    def move_delta(self, delta, gripper_open=None, frame=Frame.TOOL,
                   settle_steps=800, **ik_kwargs):
        """
        Mueve el robot por un incremento de pose del TCP y avanza la fisica
        hasta que alcanza el objetivo.

        Es el envoltorio de `robot.controller.move_delta(...)` que ademas ejecuta
        la simulacion `settle_steps` pasos para que el movimiento ocurra de verdad.
        `delta` es un `PoseDelta` (o secuencia de 6). Devuelve True si la IK
        convergio a la pose objetivo.
        """
        converged = self.robot.controller.move_delta(
            delta, gripper_open=gripper_open, frame=frame, **ik_kwargs)
        self.step(settle_steps)
        return converged

    # ------------------------------------------------------------------ #
    #  Camara
    # ------------------------------------------------------------------ #
    def default_camera(self, azimuth=135, elevation=-20, distance=None):
        """Crea una camara libre que encuadra al robot automaticamente."""
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, cam)  # encuadra usando model.stat
        cam.azimuth = azimuth
        cam.elevation = elevation
        if distance is not None:
            cam.distance = distance
        return cam

    # ------------------------------------------------------------------ #
    #  Visualizacion
    # ------------------------------------------------------------------ #
    def launch_viewer(self):
        """Abre el visor interactivo y simula hasta que se cierra la ventana."""
        from mujoco import viewer
        with viewer.launch_passive(self.model, self.data) as v:
            while v.is_running():
                mujoco.mj_step(self.model, self.data)
                v.sync()

    def observation(self, camera=None, width=224, height=224):
        """
        Renderiza la observacion RGB desde una camara (por defecto 224x224),
        reutilizando un renderer persistente.

        Pensada para lazos de inferencia (p. ej. OpenVLA), donde se renderiza
        una imagen por paso: crear/cerrar un `Renderer` cada vez seria lento y
        podria fugar contextos GL. Devuelve un array uint8 (H, W, 3).

        `camera` puede ser un `mujoco.MjvCamera`, el nombre/id de una camara del
        modelo, o None (camara libre por defecto).
        """
        if self._renderer is None or self._render_size != (width, height):
            if self._renderer is not None:
                self._renderer.close()
            self._ensure_offscreen(width, height)
            self._renderer = mujoco.Renderer(self.model, height, width)
            self._render_size = (width, height)
        self._renderer.update_scene(self.data, camera=self._resolve_camera(camera))
        return self._renderer.render()

    def render_image(self, path=None, width=1024, height=768, camera=None):
        """
        Renderiza un cuadro del estado actual.

        Devuelve el array RGB (H, W, 3). Si se da `path`, ademas lo guarda.
        """
        self._ensure_offscreen(width, height)
        renderer = mujoco.Renderer(self.model, height, width)
        renderer.update_scene(self.data, camera=self._resolve_camera(camera))
        img = renderer.render()
        renderer.close()
        if path:
            import imageio
            full = self._output_path(path)
            imageio.imwrite(full, img)
            print(f"Imagen guardada en: {full}")
        return img

    def record_video(self, path="movimiento.mp4", duration=5.0, fps=30,
                     control=None, width=1024, height=768, camera=None):
        """
        Graba un video de la simulacion en `path`.

        Parametros
        ----------
        path : str
            Archivo de salida (.mp4, .gif, ...).
        duration : float
            Duracion del video en segundos (de tiempo simulado).
        fps : int
            Cuadros por segundo del video.
        control : callable | None
            Funcion `control(t, robot)` que se llama antes de cada cuadro (con
            `t` = tiempo simulado en segundos) para fijar el movimiento. Si es
            None, el robot solo se mantiene en su pose contra la gravedad.
        width, height : int
            Resolucion del video.
        camera : mujoco.MjvCamera | int | str | None
            Camara a usar. Si es None se usa `default_camera()`.
        """
        import imageio

        full = self._output_path(path)
        cam = self._resolve_camera(camera)
        self._ensure_offscreen(width, height)
        renderer = mujoco.Renderer(self.model, height, width)
        n_frames = int(round(duration * fps))
        steps_per_frame = max(1, round(1.0 / (fps * self.model.opt.timestep)))

        self.robot.reset()
        frames = []
        for f in range(n_frames):
            t = f / fps
            if control is not None:
                control(t, self.robot)
            self.step(steps_per_frame)
            renderer.update_scene(self.data, camera=cam)
            frames.append(renderer.render())
        renderer.close()

        imageio.mimsave(full, frames, fps=fps)
        print(f"Video guardado en: {full}  "
              f"({n_frames} cuadros, {duration:.1f}s @ {fps}fps)")
        return full

    # ------------------------------------------------------------------ #
    #  Internos
    # ------------------------------------------------------------------ #
    def _resolve_camera(self, camera):
        """None -> camara por defecto; int/str/MjvCamera se pasan tal cual."""
        return self.default_camera() if camera is None else camera

    def _ensure_offscreen(self, width, height):
        """Agranda el framebuffer offscreen del modelo si hace falta.

        Muchos modelos (p. ej. los de Menagerie) traen un buffer pequeno (640);
        esto permite renderizar a mayor resolucion sin tocar el XML.
        """
        g = self.model.vis.global_
        g.offwidth = max(int(g.offwidth), int(width))
        g.offheight = max(int(g.offheight), int(height))

    def _output_path(self, path):
        """
        Resuelve la ruta de un archivo de salida. Si `path` no incluye carpeta,
        se coloca dentro de `self.output_dir`. Crea la carpeta si no existe.
        """
        if not os.path.dirname(path):
            path = os.path.join(self.output_dir, path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path
