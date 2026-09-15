"""
Fabrica y registro de benchmarks (entornos de prueba) por nombre.

Analogo a `RobotFactory`/`ModelFactory`, y tambien PEREZOSO (ver
`core.registry.LazyRegistry`): cada benchmark se registra como una cadena
"modulo:Clase" que solo se importa al crearlo. Asi `import benchmarks` no
arrastra LIBERO/robosuite (que ademas fijan variables de entorno de render al
importarse); esas deps pesadas solo se cargan al pedir ese benchmark.

Para agregar un benchmark:
  1. Crea su controlador en `benchmarks/TuBench/` (subclase de `BenchMark`).
  2. Registra su ruta AQUI con una linea:
         BenchmarkFactory.register("tu-bench",
                                   "benchmarks.TuBench.tu_controller:TuController")
  3. Listo: `BenchmarkFactory.available()` lo lista y se crea por nombre.

Uso:
    from benchmarks import BenchmarkFactory
    BenchmarkFactory.available()                    # ["example", "libero"]
    bench = BenchmarkFactory.create("libero", task_id=0)
"""

from core.registry import LazyRegistry

BenchmarkFactory = LazyRegistry("benchmark")

# Registro perezoso: no importa LIBERO/robosuite ni el Panda hasta `create`.
BenchmarkFactory.register("example", "benchmarks.example.example_controller:ExampleController")
BenchmarkFactory.register("libero", "benchmarks.libero.libero_controller:LiberoController")
