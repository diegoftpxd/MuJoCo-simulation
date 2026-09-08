"""
Calcula un vector de ESCALA por dimensión a partir del CSV de denoising ya
generado, para usarlo como amplitud del ruido inicial del flow-matching de pi0
(variable de entorno PI0_NOISE_SCALE, leída por models/Pi_zero/patches.py).

La idea: en vez de arrancar el flow-matching desde ruido N(0,1) (magnitud ~1 por
dimensión), arrancar desde un ruido cuyo tamaño por dimensión sea comparable a la
magnitud típica de los datos del CSV. Así el origen cubre un área mucho mayor y se
puede estudiar si el modelo sigue colapsando a los mismos clusters.

Uso (desde la raíz del repo, con un env que tenga numpy y pandas, p. ej. `tipico`):
    python scripts/make_noise_scale.py \
        --csv output/pi0_denoise/denoise_log.csv \
        --out output/pi0_denoise/noise_scale.npy \
        [--rows finals|all] [--stat absmean|std|mean]

Luego genera el CSV nuevo lanzando el servidor con la escala y un CSV de salida
distinto (para no pisar el original):
    PI0_NOISE_SCALE=output/pi0_denoise/noise_scale.npy \
    PI0_DENOISE_CSV=output/pi0_denoise/denoise_big_noise.csv \
    sbatch scripts/scriptExperiment.sh
(o define esas dos variables dentro de scripts/scriptExperiment.sh).
"""

import argparse

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="output/pi0_denoise/denoise_log.csv",
                    help="CSV de denoising ya generado.")
    ap.add_argument("--out", default="output/pi0_denoise/noise_scale.npy",
                    help="Ruta del .npy de salida con la escala por dimension.")
    ap.add_argument("--rows", choices=["finals", "all"], default="finals",
                    help="'finals' (accion final de cada call, recomendado) o 'all'.")
    ap.add_argument("--stat", choices=["absmean", "std", "mean"], default="absmean",
                    help="Estadistico por dimension: magnitud media (default), std, "
                         "o |media|.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    vcols = [c for c in df.columns if c.startswith("v") and c[1:].isdigit()]
    vcols.sort(key=lambda c: int(c[1:]))          # orden v0,v1,... = orden aplanado
    if not vcols:
        raise SystemExit(f"No hay columnas v* en {args.csv!r}.")

    if args.rows == "finals":
        fi = df.groupby("call")["iteration"].transform("max")
        data = df[df["iteration"] == fi]
    else:
        data = df
    V = data[vcols].to_numpy(dtype=float)         # (n, chunk_size*max_action_dim)

    if args.stat == "absmean":
        scale = np.abs(V).mean(axis=0)
    elif args.stat == "std":
        scale = V.std(axis=0)
    else:                                          # "mean" -> |media|
        scale = np.abs(V.mean(axis=0))

    scale = scale.astype("float32")
    np.save(args.out, scale)

    print(f"CSV entrada : {args.csv}  (filas usadas: {len(V)}, modo '{args.rows}')")
    print(f"dimensiones : {scale.size}   estadistico: {args.stat}")
    print(f"escala      : min={scale.min():.4g}  media={scale.mean():.4g}  "
          f"max={scale.max():.4g}")
    print(f"guardado en : {args.out}")
    print("\\nAhora lanza el servidor con:")
    print(f"  PI0_NOISE_SCALE={args.out} \\")
    print("  PI0_DENOISE_CSV=output/pi0_denoise/denoise_big_noise.csv \\")
    print("  sbatch scripts/scriptExperiment.sh")


if __name__ == "__main__":
    main()
