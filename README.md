# MuJoCo-simulation

Entorno de simulación de brazos robóticos en **MuJoCo** para evaluar modelos de
control (Vision-Language-Action como OpenVLA, políticas de RL, etc.) sobre
distintos benchmarks. La arquitectura desacopla cuatro piezas — **robot**,
**simulación**, **benchmark** y **modelo** — que se comunican solo a través de
dos estructuras de datos comunes, `Observation` y `Action`, de modo que
cualquier pieza se puede intercambiar sin tocar las demás.

---

## Estructura de archivos

```
MuJoCo-simulation/
├── core/                     # Capa común: estructuras de datos + runner
│   ├── observation.py        #   Observation, View (lo que produce un benchmark)
│   ├── action.py             #   Action (lo que produce un modelo)
│   ├── episode.py            #   StepResult, EpisodeResult, VideoRecorder, run_episode
│   └── experiment.py         #   run_experiments, summarize
│
├── robots/                   # El robot: modelo, estado, control y cinemática
│   ├── base.py               #   RobotArm (clase base genérica) + MJCF_DIR
│   ├── control.py            #   CartesianController (IK, movimiento por Δpose)
│   ├── structures.py         #   Pose, PoseDelta, Frame, CartesianAction
│   ├── factory.py            #   RobotFactory (registro/creación por nombre)
│   ├── tm5.py                #   TM5, TM5_700 (Techman)
│   └── panda.py              #   Panda (Franka Emika)
│
├── objects/                  # Objetos del entorno (se insertan en el MJCF)
│   ├── base.py               #   SceneObject (clase base abstracta)
│   ├── cube.py               #   Cube
│   └── sphere.py             #   Sphere
│
├── benchmarks/               # Entornos de prueba (interfaz BenchMark)
│   ├── benchmark.py          #   BenchMark (clase base abstracta)
│   ├── example/              #   ExampleController (Panda, sin GPU ni LIBERO)
│   └── libero/               #   LiberoController (envuelve LIBERO/robosuite)
│
├── models/                   # Políticas (interfaz Model)
│   ├── Model.py              #   Model (clase base abstracta)
│   ├── OpenVLA/              #   OpenVLAController (VLA 7B, torch/transformers)
│   ├── Pi_zero/              #   PiZeroController (esqueleto a completar)
│   └── Random/               #   RandomController (política de prueba)
│
├── simulation.py             # Simulation: avance de física, viewer, render, video
│
├── MJCF/                     # Modelos MuJoCo (XML + mallas)
│   ├── panda/                #   Franka Panda (MuJoCo Menagerie)
│   └── tm5-700/              #   Techman TM5-700
│
├── output/                   # Imágenes/videos generados
│
├── simulate.py               # Script: abre el viewer con un robot
├── record.py                 # Script: graba un video de un movimiento demo
├── demo_pick.py              # Script: secuencia pick & place con move_delta
├── run.py                    # Script: episodio ExampleController + RandomController
├── libero_test.py            # Script: OpenVLA sobre tareas de LIBERO-10
├── openvla_experiments.ipynb # Notebook de experimentos con OpenVLA
├── run_openvla.slurm         # Job de SLURM para correr en clúster
├── requirements.txt          # Dependencias base (mujoco, numpy, imageio, ...)
├── requirements-vla.txt      # Dependencias extra para VLA (torch, transformers)
└── environment.yml           # Entorno conda
```

---

## Flujo general

```
Benchmark  --Observation-->  Model  --Action-->  Benchmark
```

`run_episode` (en `core/episode.py`) implementa el bucle agente-entorno
estándar: pide una `Observation` al benchmark, se la pasa al modelo, aplica la
`Action` resultante y repite hasta `done` o `max_steps`. Ni el benchmark conoce
al modelo ni el modelo conoce al benchmark: ambos hablan solo `Observation` /
`Action`.

- Un **benchmark** rellena en la `Observation` todas las vistas y estados que
  tenga; cada **modelo** toma por nombre solo lo que necesita.
- Un **modelo** rellena en la `Action` lo que sabe generar (p. ej. un delta
  cartesiano + pinza); cada **benchmark** toma solo los campos que su robot
  soporta.

---

## Clases y métodos principales

### `core` — capa común

**`Observation`** (`core/observation.py`) — dataclass que produce un benchmark.
Superconjunto de vistas y estados; el modelo elige lo que usa.
- `image(name)` — devuelve la vista `name` (error claro si no existe).
- `get_image(name, default)` / `has_image(name)` — acceso tolerante.
- `get_state(name, default)` — lee propriocepción.
- `view_names` (property) — vistas disponibles.
- **`View`** — Enum con nombres canónicos de cámara (`FRONT`, `SIDE_1`,
  `SIDE_2`, `WRIST`, `AGENT`, `TOP`).

**`Action`** (`core/action.py`) — dataclass que produce un modelo. Campos:
`cartesian_delta` (6 deltas), `gripper`, `joint_targets`, `raw`, `extra`.
- `from_cartesian(delta6, gripper, raw)` (classmethod) — atajo de acción
  cartesiana + pinza.

**`core/episode.py`**
- **`StepResult`** — dataclass: `observation`, `reward`, `done`, `info`.
- **`EpisodeResult`** — dataclass: `success`, `steps`, `total_reward`.
- **`VideoRecorder`** — acumula una vista cuadro a cuadro:
  `record(observation)`, `save(path, fps)`.
- `run_episode(benchmark, model, max_steps, recorder, episode)` — corre un
  episodio y devuelve un `EpisodeResult`.

**`core/experiment.py`**
- `run_experiments(model, benchmarks, max_steps, episodes_per_task, out_dir,
  record_view)` — corre un modelo sobre varios escenarios y episodios; devuelve
  una lista de dicts con `{scenario, instruction, episode, success, steps,
  video}`.
- `summarize(results)` — resumen global: total, éxitos y tasa de éxito.

### `robots` — el robot

**`RobotArm`** (`robots/base.py`) — clase base genérica: descubre por
introspección del MJCF las articulaciones, actuadores y el sitio/cuerpo del
efector final, separando brazo y pinza. Representa el robot (modelo + estado +
control articular + cinemática), **no** la simulación.
- Estado articular: `get_joint_positions()`, `set_joint_positions(q)`,
  `get_joint_velocities()`.
- Control: `set_joint_targets(q)`, `set_control(u)`.
- Pinza: `set_gripper(is_open)`, `open_gripper()`, `close_gripper()`,
  `has_gripper` (property).
- Reset/pose: `reset(keyframe)`.
- Cinemática del TCP: `end_effector_position()`, `end_effector_orientation()`,
  `end_effector_pose()`, `arm_jacobian()`.
- `controller` (property) — crea de forma perezosa un `CartesianController`.
- `summary()` — imprime la estructura del robot.
- Properties: `name`, `num_joints`, `num_actuators`, `joint_ranges`.

**`CartesianController`** (`robots/control.py`) — política de control cartesiano
por cinemática inversa sobre un `RobotArm`.
- `move_delta(delta, gripper_open, frame, **ik_kwargs)` — IK iterativa hacia una
  pose objetivo (pose actual + Δ). Para waypoints "scripted". Devuelve si
  convergió.
- `cartesian_step(action, gripper_open, frame, damping, max_dq)` — un paso de IK
  diferencial (Jacobiano). Barato, pensado para RL / lazos de inferencia.
- Internos: `_solve_inverse_kinematics(...)` (mínimos cuadrados amortiguados),
  `_euler_to_rotation_matrix(...)`, `_orientation_error(...)`.

**`robots/structures.py`** — objetos de valor:
- **`Frame`** — Enum `TOOL` / `WORLD`.
- **`Pose`** — NamedTuple `(position, rotation)`.
- **`PoseDelta`** — incremento de pose (dx…dyaw); `translation`, `rotation_rpy`,
  `as_twist()`, `from_sequence(...)`, `coerce(...)`.
- **`CartesianAction`** — `PoseDelta` + estado de pinza; `from_vector(values,
  gripper_threshold)`.

**`RobotFactory`** (`robots/factory.py`) — registro y creación por nombre.
- `register(name)` — decorador que registra una clase de robot.
- `create(name, **kwargs)` — instancia el robot registrado.
- `available()` — lista los robots registrados.
- `make_robot(name, **kwargs)` — atajo de `create`.

**Robots concretos**
- **`TM5`** / **`TM5_700`** (`robots/tm5.py`) — familia Techman (6 GDL). Las
  variantes solo definen `MODEL`.
- **`Panda`** (`robots/panda.py`) — Franka Emika (7 GDL). Usa el cuerpo `hand`
  con offset como TCP; pinza por tendón.

### `objects` — objetos del entorno

**`SceneObject`** (`objects/base.py`) — clase base abstracta. Se agrega a un
`mujoco.MjSpec` antes de compilar.
- `add_to_spec(spec)` — inserta el cuerpo (con free joint si es movable).
- `bind(model, data)` — enlaza al modelo compilado.
- `get_pose()` / `set_pose(position, quat)` — consulta/mueve la pose.
- Properties: `init_qpos`, `n_dof`.
- Abstracto: `_add_geom(body)` (lo implementa cada tipo).

- **`Cube`** (`objects/cube.py`) — caja de lado `2*size`.
- **`Sphere`** (`objects/sphere.py`) — esfera de radio `radius`.

### `benchmarks` — entornos de prueba

**`BenchMark`** (`benchmarks/benchmark.py`) — clase base abstracta. Dos niveles:
escenario/tarea (una instancia) y episodio (configuración inicial dentro de la
tarea).
- Abstractos: `reset(episode)`, `step(action)`, `instruction` (property).
- `num_episodes` (property), `close()`.

- **`ExampleController`** (`benchmarks/example/`) — entorno mínimo que envuelve
  la `Simulation` (Panda) para probar la arquitectura sin GPU ni LIBERO. Ofrece
  3 vistas (`side_1`, `side_2`, `wrist`) y aplica el `cartesian_delta`.
- **`LiberoController`** (`benchmarks/libero/`) — envuelve un entorno de LIBERO
  (robosuite). Entrega la vista `agentview` ya orientada y adapta la convención
  de pinza de LIBERO. `tasks(suite)` (classmethod) lista `[(task_id,
  instrucción)]` de una suite.

### `models` — políticas

**`Model`** (`models/Model.py`) — clase base abstracta.
- Abstracto: `act(observation) -> Action`.
- `reset()` — reinicia estado interno entre episodios (opcional).

- **`OpenVLAController`** (`models/OpenVLA/`) — envuelve OpenVLA-7B
  (torch/transformers, imports perezosos; carga en fp16). `act(...)` elige
  su vista, predice el vector de 7 y lo devuelve como `Action`;
  `predict_action(image, instruction)` da el vector crudo des-normalizado.
- **`PiZeroController`** (`models/Pi_zero/`) — esqueleto que sigue el mismo
  patrón; `_load` y `predict_action` están por implementar.
- **`RandomController`** (`models/Random/`) — política de prueba: deltas
  cartesianos pequeños al azar. Útil para validar el pipeline sin pesos ni GPU.

### `simulation.py` — simulación y visualización

**`Simulation`** — recibe un `RobotArm` y se encarga de todo lo que no es el
robot: avanzar la física, abrir el viewer, renderizar imágenes y grabar videos.
- Construcción: `Simulation(robot, output_dir)`, `from_name(name, output_dir)`
  (classmethod). Properties `model` / `data`.
- Física: `step(n)`, `reset(keyframe)`, `move_delta(delta, gripper_open, frame,
  settle_steps, **ik_kwargs)`.
- Cámara/render: `default_camera(azimuth, elevation, distance)`,
  `observation(camera, width, height)` (renderer persistente para lazos),
  `render_image(path, width, height, camera)`.
- Video/viewer: `launch_viewer()`, `record_video(path, duration, fps, control,
  width, height, camera)`.

---

## Scripts

| Script | Qué hace |
|--------|----------|
| `simulate.py` | Abre el viewer interactivo. `--robot`, `--check`, `--list`. |
| `record.py` | Graba un video de un movimiento demo. `--robot --out --duration --fps`. |
| `demo_pick.py` | Secuencia pick & place con `move_delta` → `output/pick.mp4`. |
| `run.py` | Episodio `ExampleController` (Panda) + `RandomController`. |
| `libero_test.py` | OpenVLA sobre tareas de LIBERO-10 (requiere GPU + LIBERO). |

Ejemplos:

```bash
python simulate.py --list
```

```bash
python simulate.py --robot tm5-700
```

```bash
python demo_pick.py
```

```bash
python run.py
```

---

## Instalación

```bash
conda env create -f environment.yml
```

```bash
pip install -r requirements.txt
```

Para los modelos VLA (OpenVLA), además:

```bash
pip install -r requirements-vla.txt
```

LIBERO requiere clonar su repo en `benchmarks/libero/repo/` (o instalar el
paquete `libero`), además de GPU y ~29 GB de RAM.
