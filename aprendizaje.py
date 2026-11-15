"""
aprendizaje.py — Aprendizaje continuo: actualiza modelos con nuevos datos.
Plan 4: PRINCIPIO DE SEGURIDAD CLÍNICA:
El modelo NUNCA aprende de datos que un humano no haya validado.
Solo registros en estado 'confirmado' o 'corregido' de la cola de validación
pueden ser utilizados para el reentrenamiento incremental.

ADVERTENCIA: Modelos sintéticos para prototipo. Sin validez clínica.
"""

import numpy as np
import pandas as pd
import joblib
import json
import shutil
import sqlite3
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

import os
warnings.filterwarnings("ignore")

RUTA_BASE = Path(__file__).parent
if os.environ.get("VERCEL"):
    RUTA_DB = Path("/tmp/astrosalud.db")
    RUTA_VERSION = Path("/tmp/versiones")
    db_orig = RUTA_BASE / "datos" / "astrosalud.db"
    if db_orig.exists() and not RUTA_DB.exists():
        try:
            shutil.copy(db_orig, RUTA_DB)
        except Exception:
            pass
else:
    RUTA_DB = RUTA_BASE / "datos" / "astrosalud.db"
    RUTA_VERSION = RUTA_BASE / "modelos" / "versiones"

RUTA_MODELOS = RUTA_BASE / "modelos"

TOLERANCIA_EMPEORAMIENTO = 0.05
MIN_DATOS_ACTUALIZACION  = 3      # mínimo de registros humanos validados
VENTANA_EVALUACION       = 30


# ─── Base de datos ────────────────────────────────────────────────────────────

def inicializar_db():
    """Crea las tablas SQLite si no existen, incluyendo validación."""
    RUTA_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(RUTA_DB)
    cur = con.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS lecturas (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            astronauta_id   TEXT    NOT NULL DEFAULT 'AST-001',
            timestamp       TEXT    NOT NULL,
            dia_mision      INTEGER,
            sistolica       REAL, diastolica  REAL,
            fc              REAL, hrv         REAL,
            spo2            REAL, temperatura REAL,
            glucosa         REAL, sueno_horas REAL,
            sueno_calidad   REAL, animo       REAL,
            carga_trabajo   REAL, ejercicio_min REAL,
            onsd_mm         REAL,
            radiacion_diaria REAL,
            radiacion_acumulada REAL,
            marcado_erroneo INTEGER DEFAULT 0,
            fuente          TEXT DEFAULT 'manual'
        );

        CREATE TABLE IF NOT EXISTS predicciones (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            lectura_id      INTEGER REFERENCES lecturas(id),
            timestamp       TEXT NOT NULL,
            modelo_version  TEXT,
            fatiga_pred     INTEGER,
            fatiga_prob_v   REAL, fatiga_prob_a REAL, fatiga_prob_r REAL,
            onsd_pred       REAL, onsd_std      REAL,
            radiacion_pred  REAL
        );

        CREATE TABLE IF NOT EXISTS resultados_reales (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            prediccion_id   INTEGER REFERENCES predicciones(id),
            dia_objetivo    INTEGER,
            fatiga_real     INTEGER,
            onsd_real       REAL,
            radiacion_real  REAL,
            usado_entrenamiento INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS versiones_modelo (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            nombre          TEXT NOT NULL,
            ruta_archivo    TEXT NOT NULL,
            f1_macro        REAL,
            f1_rojo         REAL,
            mae_onsd        REAL,
            n_datos_entrenamiento INTEGER,
            activa          INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS configuracion (
            clave TEXT PRIMARY KEY,
            valor TEXT
        );
    """)

    cur.execute("INSERT OR IGNORE INTO configuracion VALUES ('dia_mision_actual', '1')")
    cur.execute("INSERT OR IGNORE INTO configuracion VALUES ('modelo_estado', 'inicial')")
    cur.execute("INSERT OR IGNORE INTO configuracion VALUES ('dias_datos_reales', '0')")
    con.commit()
    con.close()

    # Inicializar también esquema de validación humana
    try:
        from validacion import inicializar_tablas_validacion
        inicializar_tablas_validacion()
    except Exception as e:
        print("Aviso al inicializar tablas de validación:", e)


def get_config(clave: str, default=None):
    con = sqlite3.connect(RUTA_DB)
    r = con.execute("SELECT valor FROM configuracion WHERE clave=?", (clave,)).fetchone()
    con.close()
    return r[0] if r else default


def set_config(clave: str, valor):
    con = sqlite3.connect(RUTA_DB)
    con.execute("INSERT OR REPLACE INTO configuracion VALUES (?,?)", (clave, str(valor)))
    con.commit()
    con.close()


# ─── Validación fisiológica ───────────────────────────────────────────────────

RANGOS_VALIDOS = {
    "sistolica":    (60, 220),
    "diastolica":   (30, 140),
    "fc":           (25, 200),
    "hrv":          (1, 300),
    "spo2":         (70, 100),
    "temperatura":  (34, 42),
    "glucosa":      (30, 600),
    "sueno_horas":  (0, 24),
    "sueno_calidad":(1, 5),
    "animo":        (1, 5),
    "carga_trabajo":(1, 5),
    "ejercicio_min":(0, 600),
    "onsd_mm":      (1, 10),
    "radiacion_diaria": (0, 100),
}


def validar_lectura(lectura: dict) -> list:
    """Devuelve lista de advertencias. Lista vacía = válida."""
    advertencias = []
    for campo, (lo, hi) in RANGOS_VALIDOS.items():
        v = lectura.get(campo)
        if v is not None and (v < lo or v > hi):
            advertencias.append(f"{campo}={v} fuera de rango plausible [{lo},{hi}]")
    return advertencias


# ─── Guardar lectura ──────────────────────────────────────────────────────────

def guardar_lectura(lectura: dict) -> int:
    """Inserta una lectura en SQLite. Retorna el id insertado."""
    con = sqlite3.connect(RUTA_DB)
    cur = con.cursor()
    campos = ["astronauta_id","timestamp","dia_mision",
              "sistolica","diastolica","fc","hrv","spo2","temperatura",
              "glucosa","sueno_horas","sueno_calidad","animo","carga_trabajo",
              "ejercicio_min","onsd_mm","radiacion_diaria","radiacion_acumulada",
              "marcado_erroneo","fuente"]
    valores = [lectura.get(c) for c in campos]
    cur.execute(f"INSERT INTO lecturas ({','.join(campos)}) VALUES ({','.join(['?']*len(campos))})", valores)
    lid = cur.lastrowid
    con.commit()
    con.close()

    # Actualizar contador de días
    dias = int(get_config("dias_datos_reales", "0")) + 1
    set_config("dias_datos_reales", dias)

    # Revisar si se pueden sincronizar pares pendientes de validación
    try:
        from validacion import sincronizar_cola_pendientes
        sincronizar_cola_pendientes(lectura.get("astronauta_id", "AST-001"))
    except Exception:
        pass

    return lid


# ─── Cargar modelo activo ─────────────────────────────────────────────────────

def cargar_modelo(nombre: str) -> dict:
    ruta = RUTA_MODELOS / f"{nombre}.joblib"
    if not ruta.exists():
        raise FileNotFoundError(f"Modelo {nombre} no encontrado en {ruta}")
    return joblib.load(ruta)


# ─── Preparar features desde SQLite ──────────────────────────────────────────

def preparar_features_desde_db(astronauta_id: str = "AST-001", ultimos_n: int = 60) -> pd.DataFrame:
    con = sqlite3.connect(RUTA_DB)
    df = pd.read_sql_query(
        """SELECT * FROM lecturas
           WHERE astronauta_id=? AND marcado_erroneo=0
           ORDER BY dia_mision DESC LIMIT ?""",
        con, params=(astronauta_id, ultimos_n)
    )
    con.close()

    if len(df) == 0:
        return pd.DataFrame()

    df = df.sort_values("dia_mision").reset_index(drop=True)

    metricas = ["sistolica","diastolica","fc","hrv","spo2","temperatura",
                "glucosa","sueno_horas","sueno_calidad","animo","carga_trabajo",
                "ejercicio_min","radiacion_diaria"]

    for m in metricas:
        if m not in df.columns:
            continue
        df[f"{m}_ma3"]    = df[m].rolling(3, min_periods=1).mean()
        df[f"{m}_ma7"]    = df[m].rolling(7, min_periods=1).mean()
        df[f"{m}_slope7"] = df[m].rolling(7, min_periods=3).apply(
            lambda v: np.polyfit(np.arange(len(v)), v, 1)[0] if len(v) >= 3 else 0, raw=True)

    for m in metricas:
        if m not in df.columns:
            continue
        media_p = df[m].expanding(min_periods=5).mean()
        std_p   = df[m].expanding(min_periods=5).std().replace(0, 1).fillna(1)
        df[f"{m}_zscore"] = (df[m] - media_p) / std_p

    df["dia_mision_norm"] = df["dia_mision"] / 180.0
    return df


# ─── Obtener pares validados para entrenamiento ──────────────────────────────

def obtener_pares_entrenamiento(astronauta_id: str = "AST-001") -> pd.DataFrame:
    """
    REGLA PLAN 4: El modelo SOLO aprende de registros que un HUMANO
    ha revisado y marcado como 'confirmado' o 'corregido' en la cola_validacion.
    """
    con = sqlite3.connect(RUTA_DB)
    tabla_existe = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='cola_validacion'"
    ).fetchone()

    if not tabla_existe:
        con.close()
        return pd.DataFrame()

    pares = pd.read_sql_query("""
        SELECT id as res_id, fatiga_validada as fatiga_real, onsd_validado as onsd_real,
               lectura_id_origen as lectura_id, estado, validado_por
        FROM cola_validacion
        WHERE astronauta_id=?
          AND estado IN ('confirmado', 'corregido')
          AND usado_entrenamiento=0
          AND fatiga_validada IS NOT NULL
    """, con, params=(astronauta_id,))
    con.close()
    return pares


# ─── Actualización incremental gobernada ─────────────────────────────────────

def actualizar_modelo(astronauta_id: str = "AST-001", forzar_sin_candidato: bool = False) -> dict:
    """
    En Plan 4: Genera un modelo CANDIDATO para que un humano lo apruebe antes de activarse.
    Si forzar_sin_candidato es True, aplica directamente (p.ej. en pruebas controladas).
    """
    pares = obtener_pares_entrenamiento(astronauta_id)
    if len(pares) < MIN_DATOS_ACTUALIZACION:
        return {
            "estado": "sin_datos_suficientes",
            "n_pares_validados": len(pares),
            "minimo": MIN_DATOS_ACTUALIZACION,
            "mensaje": f"Se requieren al menos {MIN_DATOS_ACTUALIZACION} observaciones validadas por humanos. Hay {len(pares)}."
        }

    from validacion import generar_modelo_candidato
    resultado_cand = generar_modelo_candidato(astronauta_id, usuario="Sistema (Auto)")
    if "error" in resultado_cand:
        return resultado_cand

    if forzar_sin_candidato:
        from validacion import decidir_modelo_candidato
        decidir_modelo_candidato(resultado_cand["candidato_id"], aprobar=True, usuario="TestRunner", motivo="Auto-aprobado en prueba")
        return {"estado": "actualizado", "mensaje": "Modelo actualizado y aprobado directamente."}

    return {
        "estado": "candidato_generado",
        "candidato_id": resultado_cand.get("candidato_id"),
        "mensaje": "Se generó una versión candidata. Requiere aprobación humana con PIN antes de activarse."
    }


def restaurar_modelo_anterior(nombre: str = "fatiga") -> bool:
    """Restaura la versión previa de los backups."""
    versiones = sorted(RUTA_VERSION.glob(f"{nombre}_*.joblib"), reverse=True)
    if not versiones:
        return False
    shutil.copy(versiones[0], RUTA_MODELOS / f"{nombre}.joblib")
    set_config("modelo_estado", "restaurado")
    return True


# ─── Predicción ──────────────────────────────────────────────────────────────

def predecir(lectura: dict, astronauta_id: str = "AST-001") -> dict:
    df_hist = preparar_features_desde_db(astronauta_id, ultimos_n=14)
    if df_hist.empty:
        return {"error": "Sin historial suficiente para predecir"}

    try:
        modelo_fat = cargar_modelo("fatiga")
    except Exception as e:
        return {"error": f"No se pudo cargar el modelo: {e}"}

    scaler     = modelo_fat["scaler"]
    feat_cols  = modelo_fat["feature_cols"]
    sgd_cls    = modelo_fat["modelo"]

    ultima_fila = df_hist.tail(1)
    cols_disp   = [c for c in feat_cols if c in ultima_fila.columns]
    X = np.zeros((1, len(feat_cols)), dtype=np.float32)
    idx_map = {c: i for i, c in enumerate(feat_cols)}
    for col in cols_disp:
        if col in idx_map:
            X[0, idx_map[col]] = float(ultima_fila[col].fillna(0).values[0])

    X_s = scaler.transform(X)
    fatiga_pred = int(sgd_cls.predict(X_s)[0])

    try:
        fatiga_prob = sgd_cls.predict_proba(X_s)[0].tolist()
    except Exception:
        fatiga_prob = [0.7 if fatiga_pred==0 else 0.15,
                       0.7 if fatiga_pred==1 else 0.15,
                       0.7 if fatiga_pred==2 else 0.15]

    resultado = {
        "fatiga_7d": fatiga_pred,
        "fatiga_prob": {
            "verde": float(fatiga_prob[0]) if len(fatiga_prob) > 0 else 0.0,
            "amarillo": float(fatiga_prob[1]) if len(fatiga_prob) > 1 else 0.0,
            "rojo": float(fatiga_prob[2]) if len(fatiga_prob) > 2 else 0.0
        },
        "modelo_version": get_config("modelo_estado", "inicial"),
        "dias_datos": get_config("dias_datos_reales", "0"),
    }

    try:
        modelo_onsd = cargar_modelo("onsd")
        X_onsd = np.zeros((1, len(modelo_onsd["feature_cols"])), dtype=np.float32)
        for j, col in enumerate(modelo_onsd["feature_cols"]):
            if col in ultima_fila.columns:
                X_onsd[0, j] = float(ultima_fila[col].fillna(0).values[0])
        X_onsd_s = modelo_onsd["scaler"].transform(X_onsd)
        resultado["onsd_7d"] = float(modelo_onsd["modelo"].predict(X_onsd_s)[0])
        resultado["onsd_std"] = modelo_onsd.get("std_residuos", 0.1)
    except Exception:
        pass

    return resultado


if __name__ == "__main__":
    inicializar_db()
    print("BD inicializada en:", RUTA_DB)
    print("Estado del modelo:", get_config("modelo_estado", "inicial"))
