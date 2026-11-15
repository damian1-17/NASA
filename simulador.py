"""
simulador.py — Generador de datos sintéticos para misiones espaciales de 180 días.
Plan 2: datos con más variabilidad para garantizar las tres clases de fatiga.

ADVERTENCIA IMPORTANTE:
    Datos SINTÉTICOS para validar el flujo de la aplicación y entrenar el prototipo.
    NO representan datos clínicos reales. NO usar para diagnóstico médico.

Uso:
    python simulador.py              # genera datos en datos_mision.csv
    python simulador.py --semilla 42
"""

import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path

SEMILLA = 42
N_ASTRONAUTAS = 50
N_DIAS = 180
CONFIG_PATH = Path(__file__).parent / "config_rangos.json"


def cargar_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def generar_linea_base(rng: np.random.Generator, astronauta_id: int) -> dict:
    """Línea base fisiológica individual."""
    return {
        "astronauta_id": astronauta_id,
        "base_sistolica":        rng.normal(115, 8),
        "base_diastolica":       rng.normal(72, 5),
        "base_fc":               rng.normal(62, 8),
        "base_hrv":              rng.normal(65, 15),
        "base_spo2":             rng.normal(98.0, 0.5),
        "base_temperatura":      rng.normal(36.6, 0.2),
        "base_glucosa":          rng.normal(85, 8),
        "base_sueno_horas":      rng.normal(7.2, 0.5),
        "base_sueno_calidad":    rng.normal(3.8, 0.5),
        "base_animo":            rng.normal(4.0, 0.4),
        "base_carga_trabajo":    rng.normal(2.5, 0.5),
        "base_ejercicio":        rng.normal(120, 20),
        "base_onsd":             rng.normal(4.3, 0.2),
        "base_radiacion_diaria": rng.normal(0.35, 0.05),
        "cumplimiento_ejercicio": rng.uniform(0.4, 1.0),   # más variabilidad
        "resiliencia":            rng.uniform(0.3, 1.0),   # más variabilidad
    }


def generar_datos_astronauta(astronauta_id: int, linea_base: dict,
                              rng: np.random.Generator,
                              n_dias: int = N_DIAS) -> pd.DataFrame:
    """Serie temporal de n_dias para un astronauta con efectos de microgravedad."""
    dias = np.arange(n_dias)
    lb = linea_base
    cum_ejercicio = lb["cumplimiento_ejercicio"]
    resiliencia   = lb["resiliencia"]

    # ── Ruido diario ─────────────────────────────────────────────────────────
    ruido = {
        "sistolica":    rng.normal(0, 4, n_dias),
        "diastolica":   rng.normal(0, 3, n_dias),
        "fc":           rng.normal(0, 5, n_dias),
        "hrv":          rng.normal(0, 8, n_dias),
        "spo2":         rng.normal(0, 0.4, n_dias),
        "temperatura":  rng.normal(0, 0.18, n_dias),
        "glucosa":      rng.normal(0, 6, n_dias),
        "sueno_horas":  rng.normal(0, 0.6, n_dias),
        "sueno_calidad":rng.normal(0, 0.4, n_dias),
        "animo":        rng.normal(0, 0.5, n_dias),
        "carga":        rng.normal(0, 0.4, n_dias),
        "ejercicio":    rng.normal(0, 20, n_dias),
    }

    # ── Degradación por microgravedad (más pronunciada para astronautas con
    #    bajo cumplimiento de ejercicio o baja resiliencia) ───────────────────
    factor_proteccion  = cum_ejercicio * resiliencia
    # La degradación máxima varía ampliamente entre astronautas
    # Nota: uniform(low, high) requiere low < high
    degradacion_hrv_max    = -rng.uniform(15, 40) * (1 - factor_proteccion)
    degradacion_sueno_max  = -rng.uniform(0.5, 2.0) * (1 - 0.5 * resiliencia)
    degradacion_animo_max  = -rng.uniform(0.3, 1.5) * (1 - 0.5 * resiliencia)
    aumento_carga_max      =  rng.uniform(0.2, 1.5) * (1 - 0.4 * resiliencia)

    t = dias / n_dias
    degradacion_hrv    = degradacion_hrv_max    * t
    degradacion_sueno  = degradacion_sueno_max  * t
    degradacion_animo  = degradacion_animo_max  * t
    aumento_carga      = aumento_carga_max      * t
    aumento_onsd       = rng.uniform(0.2, 0.8)  * t * (1 - 0.3 * factor_proteccion)
    degradacion_fc     = -rng.uniform(3, 12)    * t * (1 - factor_proteccion)
    efecto_fluidos_sis = 5 * np.exp(-dias / 30)
    capacidad_fisica   = 1.0 - rng.uniform(0.1, 0.35) * t * (1 - factor_proteccion)

    # ── Eventos atípicos ─────────────────────────────────────────────────────
    n_fatiga   = rng.integers(2, 8)
    n_rad      = rng.integers(0, 3)
    n_estres   = rng.integers(1, 6)
    dias_fatiga  = rng.choice(dias, size=n_fatiga,  replace=False)
    dias_rad     = rng.choice(dias, size=n_rad,     replace=False) if n_rad > 0 else []
    dias_estres  = rng.choice(dias, size=n_estres,  replace=False)
    mag_pico     = rng.uniform(2, 10, size=n_rad) if n_rad > 0 else []

    # ── Construcción de arrays ────────────────────────────────────────────────
    ejercicio_real = lb["base_ejercicio"] * capacidad_fisica * cum_ejercicio

    sistolica    = lb["base_sistolica"]    + ruido["sistolica"]    + degradacion_fc * 0.3 + efecto_fluidos_sis
    diastolica   = lb["base_diastolica"]   + ruido["diastolica"]   + degradacion_fc * 0.15
    fc           = lb["base_fc"]           + ruido["fc"]           + degradacion_fc
    hrv          = lb["base_hrv"]          + ruido["hrv"]          + degradacion_hrv
    spo2         = lb["base_spo2"]         + ruido["spo2"]
    temperatura  = lb["base_temperatura"]  + ruido["temperatura"]
    glucosa      = lb["base_glucosa"]      + ruido["glucosa"]
    sueno_horas  = lb["base_sueno_horas"]  + ruido["sueno_horas"]  + degradacion_sueno
    sueno_calidad= lb["base_sueno_calidad"]+ ruido["sueno_calidad"]+ degradacion_sueno * 0.5
    animo        = lb["base_animo"]        + ruido["animo"]        + degradacion_animo
    carga        = lb["base_carga_trabajo"]+ ruido["carga"]        + aumento_carga
    ejercicio    = ejercicio_real          + ruido["ejercicio"]

    onsd = np.full(n_dias, np.nan)
    for d in np.arange(0, n_dias, 7):
        onsd[d] = lb["base_onsd"] + aumento_onsd[d] + rng.normal(0, 0.12)

    radiacion_diaria  = np.full(n_dias, lb["base_radiacion_diaria"]) + rng.normal(0, 0.04, n_dias)
    for i, d in enumerate(dias_rad):
        dur = rng.integers(1, 4)
        for dd in range(d, min(d + dur, n_dias)):
            radiacion_diaria[dd] += mag_pico[i]
    radiacion_acumulada = np.cumsum(np.maximum(radiacion_diaria, 0))

    # ── Eventos: fatiga ───────────────────────────────────────────────────────
    for d in dias_fatiga:
        dur = rng.integers(1, 5)
        intensidad = rng.uniform(0.5, 2.0)   # variable para generar clase roja
        for dd in range(d, min(d + dur, n_dias)):
            sueno_horas[dd]  -= rng.uniform(1.0, 3.0) * intensidad
            hrv[dd]          -= rng.uniform(10, 30) * intensidad
            animo[dd]        -= rng.uniform(0.5, 2.0) * intensidad
            carga[dd]        += rng.uniform(0.5, 2.0) * intensidad
            fc[dd]           += rng.uniform(5, 20) * intensidad

    # ── Eventos: estrés ───────────────────────────────────────────────────────
    for d in dias_estres:
        dur = rng.integers(1, 3)
        for dd in range(d, min(d + dur, n_dias)):
            sistolica[dd]    += rng.uniform(10, 30)
            diastolica[dd]   += rng.uniform(5, 18)
            fc[dd]           += rng.uniform(10, 25)
            hrv[dd]          -= rng.uniform(5, 20)
            animo[dd]        -= rng.uniform(0.3, 1.2)
            carga[dd]        += rng.uniform(0.5, 1.5)

    # ── Clipping fisiológico ──────────────────────────────────────────────────
    sistolica    = np.clip(sistolica,    80,  200)
    diastolica   = np.clip(diastolica,   40,  120)
    fc           = np.clip(fc,           35,  150)
    hrv          = np.clip(hrv,          10,  200)
    spo2         = np.clip(spo2,         88,  100)
    temperatura  = np.clip(temperatura,  35.0, 40.0)
    glucosa      = np.clip(glucosa,      50,   300)
    sueno_horas  = np.clip(sueno_horas,  2.0,  12.0)
    sueno_calidad= np.clip(sueno_calidad,1.0,   5.0)
    animo        = np.clip(animo,        1.0,   5.0)
    carga        = np.clip(carga,        1.0,   5.0)
    ejercicio    = np.clip(ejercicio,    0,     240)

    # ── Etiquetas objetivo a 7 días ───────────────────────────────────────────
    fatiga_7d    = np.full(n_dias, np.nan)
    onsd_7d      = np.full(n_dias, np.nan)
    radiacion_7d = np.full(n_dias, np.nan)

    # Calcular percentiles del score para umbrales dinámicos por astronauta
    scores = []
    for d in range(n_dias - 7):
        s_fut   = np.nanmean(sueno_horas[d+1:d+8])
        hrv_fut = np.nanmean(hrv[d+1:d+8])
        a_fut   = np.nanmean(animo[d+1:d+8])
        c_fut   = np.nanmean(carga[d+1:d+8])
        # Normalizar usando rangos fisiológicos extremos observables
        sc = (
            np.clip((7.5 - s_fut)   / 5.5, 0, 1) * 0.30 +
            np.clip((90  - hrv_fut) / 80,  0, 1) * 0.30 +
            np.clip((5.0 - a_fut)   / 4.0, 0, 1) * 0.20 +
            np.clip((c_fut - 1.0)   / 4.0, 0, 1) * 0.20
        )
        scores.append(sc)

    scores_arr = np.array(scores)
    # Umbrales percentiles para distribución ~50/35/15
    p50 = np.percentile(scores_arr, 50)
    p85 = np.percentile(scores_arr, 85)

    for d in range(n_dias - 7):
        sc = scores_arr[d]
        if sc < p50:
            fatiga_7d[d] = 0
        elif sc < p85:
            fatiga_7d[d] = 1
        else:
            fatiga_7d[d] = 2

        onsd_futuros = [onsd[dd] for dd in range(d+1, min(d+15, n_dias)) if not np.isnan(onsd[dd])]
        if onsd_futuros:
            onsd_7d[d] = onsd_futuros[0]
        radiacion_7d[d] = radiacion_acumulada[min(d+7, n_dias-1)] - radiacion_acumulada[d]

    # ── Datos faltantes ───────────────────────────────────────────────────────
    campos_con_faltantes = [
        "sistolica","diastolica","fc","hrv","spo2","temperatura",
        "glucosa","sueno_horas","sueno_calidad","animo","carga_trabajo","ejercicio_min"
    ]

    df = pd.DataFrame({
        "astronauta_id":       astronauta_id,
        "dia_mision":          dias + 1,
        "sistolica":           np.round(sistolica, 1),
        "diastolica":          np.round(diastolica, 1),
        "fc":                  np.round(fc, 1),
        "hrv":                 np.round(hrv, 1),
        "spo2":                np.round(spo2, 2),
        "temperatura":         np.round(temperatura, 2),
        "glucosa":             np.round(glucosa, 1),
        "sueno_horas":         np.round(sueno_horas, 2),
        "sueno_calidad":       np.round(np.clip(sueno_calidad, 1, 5), 1),
        "animo":               np.round(np.clip(animo, 1, 5), 1),
        "carga_trabajo":       np.round(np.clip(carga, 1, 5), 1),
        "ejercicio_min":       np.round(ejercicio, 1),
        "onsd_mm":             np.round(onsd, 2),
        "radiacion_diaria":    np.round(radiacion_diaria, 4),
        "radiacion_acumulada": np.round(radiacion_acumulada, 2),
        "fatiga_7d":           fatiga_7d,
        "onsd_7d":             np.round(onsd_7d, 3),
        "radiacion_7d":        np.round(radiacion_7d, 4),
    })

    for col in campos_con_faltantes:
        tasa = rng.uniform(0.05, 0.10)
        mask = rng.random(n_dias) < tasa
        df.loc[mask, col] = np.nan

    return df


def guardar_lineas_base(lineas_base: list, path: Path):
    pd.DataFrame(lineas_base).to_csv(path, index=False)
    print(f"  Líneas base guardadas: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--semilla",        type=int, default=SEMILLA)
    parser.add_argument("--n_astronautas",  type=int, default=N_ASTRONAUTAS)
    parser.add_argument("--n_dias",         type=int, default=N_DIAS)
    parser.add_argument("--salida",         type=str, default="datos_mision.csv")
    args = parser.parse_args()

    n_ast  = args.n_astronautas
    n_dias = args.n_dias

    print("=" * 60)
    print("SIMULADOR DE DATOS SINTÉTICOS — NASA Space Apps Challenge")
    print("ADVERTENCIA: Datos ficticios para prototipo. No uso clínico.")
    print("=" * 60)
    print(f"Semilla: {args.semilla} | Astronautas: {n_ast} | Días: {n_dias}")

    rng = np.random.default_rng(args.semilla)

    todos = []
    lineas_base = []
    for i in range(n_ast):
        lb = generar_linea_base(rng, i)
        lineas_base.append(lb)
        df_ast = generar_datos_astronauta(i, lb, rng, n_dias)
        todos.append(df_ast)
        if (i + 1) % 10 == 0:
            print(f"  Generados {i+1}/{n_ast} astronautas...")

    df_total = pd.concat(todos, ignore_index=True)
    salida = Path(__file__).parent / args.salida
    df_total.to_csv(salida, index=False)
    print(f"\n  Datos guardados: {salida} ({len(df_total):,} filas)")

    pct_miss = df_total.isnull().mean() * 100
    print("\n  % datos faltantes:")
    for col, pct in pct_miss[pct_miss > 0].items():
        print(f"    {col}: {pct:.1f}%")

    dist = df_total["fatiga_7d"].value_counts(normalize=True).sort_index()
    print("\n  Distribución fatiga:")
    for k, v in dist.items():
        print(f"    {['Verde','Amarillo','Rojo'][int(k)]}: {v*100:.1f}%")

    guardar_lineas_base(lineas_base, Path(__file__).parent / "lineas_base.csv")
    print("\nSimulación completada.")
    return df_total, lineas_base


if __name__ == "__main__":
    main()
