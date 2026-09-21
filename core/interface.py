"""
Interfaz interactiva entre un `Model` y un `BenchMark`.

`Interface` empareja un benchmark (entorno) con un modelo (politica) y expone dos
operaciones de alto nivel, pensadas para uso manual/exploratorio (p. ej. en un
notebook):

  - `setup_scene()` : reinicia la escena, MUESTRA su estado inicial y pide por
                      teclado el prompt (texto) que recibira el modelo.
  - `step()`        : OBSERVA, pide al modelo su chunk de acciones y lo EJECUTA
                      en el benchmark.

A diferencia de `run_episode` (bucle no interactivo, ver core/episode.py), aqui
el usuario ve la escena y elige la instruccion. Cada operacion se descompone en
metodos privados pequeños y de una sola responsabilidad, para que se lea que hace
cada paso.
"""


class Interface:
    """Empareja un modelo y un benchmark para interactuar paso a paso."""

    def __init__(self, benchmark, model):
        self.benchmark = benchmark
        self.model = model
        self._observation = None      # ultima observacion del entorno
        self._prompt = None           # instruccion (texto) que recibe el modelo

    # ------------------------------------------------------------------ #
    #  Preparar la escena: ver el estado inicial y elegir el prompt
    # ------------------------------------------------------------------ #
    def setup_scene(self, episode=None, view=None):
        """Reinicia la escena, muestra su estado inicial y pide el prompt."""
        self._observation = self._reset(episode)
        self._show(self._observation, view)
        self._prompt = self._ask_prompt()
        return self._observation

    def _reset(self, episode):
        """Reinicia modelo y benchmark; devuelve la observacion inicial."""
        self.model.reset()
        return self.benchmark.reset(episode)

    def _show(self, observation, view=None):
        """Muestra una vista (imagen) del estado actual de la escena."""
        import matplotlib.pyplot as plt
        name = view if view is not None else observation.view_names[0]
        plt.imshow(observation.image(name))
        plt.title(f"vista: {name}")
        plt.axis("off")
        plt.show()

    def _ask_prompt(self):
        """Pide por teclado la instruccion en texto para el modelo."""
        return input("Prompt para el modelo: ").strip()

    # ------------------------------------------------------------------ #
    #  Un paso: observar -> obtener las acciones -> ejecutarlas
    # ------------------------------------------------------------------ #
    def step(self):
        """Observa, pide el chunk de acciones al modelo y lo ejecuta."""
        observation = self._observe()
        actions = self._predict(observation)
        return self._execute(actions)

    def _observe(self):
        """Devuelve la observacion actual con el prompt del usuario inyectado."""
        if self._observation is None:
            raise RuntimeError("Llama primero a setup_scene().")
        if self._prompt:
            self._observation.instruction = self._prompt
        return self._observation

    def _predict(self, observation):
        """Pide al modelo su lista de acciones (chunk) para la observacion."""
        return list(self.model.act(observation))

    def _execute(self, actions):
        """Ejecuta las acciones en el benchmark; guarda la ultima observacion."""
        result = None
        for action in actions:
            result = self.benchmark.step(action)
            self._observation = result.observation
            if result.done:
                break
        return result
