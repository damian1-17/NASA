"""
entrenar_base.py — Entrenamiento inicial con datos sintéticos (arranque en frío).
Plan 2: SGDClassifier/SGDRegressor + joblib versionados.

ADVERTENCIA: Modelos entrenados con datos SINTÉTICOS. Sin validez clínica.

Uso:
    python entrenar_base.py
"""

import numpy as np
import pandas as pd
import json
import joblib
import shutil
import warnings
from pathlib import Path
from datetime import datetime
from typing import Tuple

from sklearn.linear_model import SGDClassifier, SGDRegressor, Ridge
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error
)
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")

RUTA_BASE    = Path(__file__).parent
RUTA_DATOS   = RUTA_BASE / "datos_mision.csv"
RUTA_LB      = RUTA_BASE / "lineas_base.csv"
RUTA_MODELOS = RUTA_BASE / "modelos"
RUTA_VERSION = RUTA_MODELOS / "versiones"

METRICAS_BASE = [
    "sistolica","diastolica","fc","hrv","spo2","temperatura",
    "glucosa","sueno_horas","sueno_calidad","animo","carga_trabajo",
    "ejercicio_min","radiacion_diaria"
]


# ─── Ingeniería de características ───────────────────────────────────────────

def crear_caracteristicas(df: pd.DataFrame, lineas_base: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.merge(
        lineas_base[["astronauta_id"] + [c for c in lineas_base.columns if c.startswith("base_")]],
        on="astronauta_id", how="left"
    )

    # Interpolación temporal por astronauta
    for col in METRICAS_BASE:
        if col in df.columns:
            df[col] = (df.groupby("astronauta_id")[col]
                         .transform(lambda x: x.interpolate(method="linear", limit=3)
                                               .ffill().bfill()))

    grupos = df.groupby("astronauta_id")

    for metrica in METRICAS_BASE:
        if metrica not in df.columns:
            continue
        df[f"{metrica}_ma3"]    = grupos[metrica].transform(lambda x: x.rolling(3, min_periods=1).mean())
        df[f"{metrica}_ma7"]    = grupos[metrica].transform(lambda x: x.rolling(7, min_periods=1).mean())
        df[f"{metrica}_slope7"] = grupos[metrica].transform(
            lambda x: x.rolling(7, min_periods=3).apply(
                lambda v: np.polyfit(np.arange(len(v)), v, 1)[0] if len(v) >= 3 else 0, raw=True))
        col_base = f"base_{metrica}"
        if col_base in df.columns:
            std_p = grupos[metrica].transform(lambda x: x.iloc[:14].std() if len(x) >= 14 else x.std()).replace(0, 1)
            df[f"{metrica}_zscore"] = (df[metrica] - df[col_base]) / std_p

    df["dia_mision_norm"] = df["dia_mision"] / 180.0
    return df


def seleccionar_features(df: pd.DataFrame, objetivo: str):
    excluir = {
        "astronauta_id","dia_mision","fatiga_7d","onsd_7d",
        "radiacion_7d","onsd_mm","radiacion_acumulada"
    } | set(c for c in df.columns if c.startswith("base_"))
    feature_cols = [c for c in df.columns if c not in excluir]
    mask  = df[objetivo].notna()
    X     = df.loc[mask, feature_cols].fillna(0).values.astype(np.float32)
    y     = df.loc[mask, objetivo].values
    grupos= df.loc[mask, "astronauta_id"].values
    return X, y, grupos, feature_cols


def split_por_astronauta(X, y, grupos, test_size=0.2, semilla=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=semilla)
    idx_tr, idx_te = next(gss.split(X, y, grupos))
    return X[idx_tr], X[idx_te], y[idx_tr], y[idx_te]


# ─── Modelo de fatiga (SGD + GB comparación) ─────────────────────────────────

def entrenar_fatiga(df_feat: pd.DataFrame) -> dict:
    print("\n── Modelo Fatiga a 7 días ─────────────────────────────────")
    X, y, grupos, feature_cols = seleccionar_features(df_feat, "fatiga_7d")
    X_tr, X_te, y_tr, y_te = split_por_astronauta(X, y, grupos)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # ── SGDClassifier (online learning, partial_fit compatible) ──────────────
    clases = np.array([0.0, 1.0, 2.0])
    pesos  = {0: 1.0, 1: 2.0, 2: 4.0}  # mayor peso a clase roja
    sgd = SGDClassifier(
        loss="modified_huber", max_iter=1, warm_start=True,
        class_weight=pesos, random_state=42, tol=None
    )
    # Entrenamiento incremental en mini-lotes
    idx = np.random.default_rng(42).permutation(len(X_tr_s))
    tam_lote = max(64, len(X_tr_s) // 20)
    for epoch in range(10):
        for start in range(0, len(idx), tam_lote):
            lote = idx[start:start + tam_lote]
            sgd.partial_fit(X_tr_s[lote], y_tr[lote], classes=clases)

    pred_sgd  = sgd.predict(X_te_s)
    f1_sgd    = f1_score(y_te, pred_sgd, average="macro", zero_division=0)
    f1r_sgd   = f1_score(y_te, pred_sgd, labels=[2], average="macro", zero_division=0)
    clases_pres = sorted(np.unique(np.concatenate([y_tr, y_te])).astype(int))
    nombres_cl  = [["Verde","Amarillo","Rojo"][c] for c in clases_pres]
    print(f"  SGDClassifier — F1 macro: {f1_sgd:.3f} | F1 rojo: {f1r_sgd:.3f}")
    print(classification_report(y_te, pred_sgd, labels=clases_pres,
                                 target_names=nombres_cl, zero_division=0))

    # ── Gradient Boosting (referencia de precisión) ──────────────────────────
    gb = GradientBoostingClassifier(n_estimators=120, max_depth=4,
                                     learning_rate=0.1, subsample=0.8, random_state=42)
    gb.fit(X_tr_s, y_tr)
    pred_gb = gb.predict(X_te_s)
    f1_gb   = f1_score(y_te, pred_gb, average="macro", zero_division=0)
    f1r_gb  = f1_score(y_te, pred_gb, labels=[2], average="macro", zero_division=0)
    print(f"\n  GradientBoosting — F1 macro: {f1_gb:.3f} | F1 rojo: {f1r_gb:.3f}")
    print(classification_report(y_te, pred_gb, labels=clases_pres,
                                 target_names=nombres_cl, zero_division=0))

    mejor = "gb" if f1r_gb >= f1r_sgd else "sgd"
    print(f"\n  Modelo operativo: SGDClassifier (online). Referencia GB mejor: {mejor=='gb'}")

    return {
        "sgd": sgd, "gb": gb, "scaler": scaler,
        "feature_cols": feature_cols,
        "resultados": {"sgd": {"f1_macro": f1_sgd, "f1_rojo": f1r_sgd},
                       "gb":  {"f1_macro": f1_gb,  "f1_rojo": f1r_gb}},
        "clases": clases
    }


# ─── Modelo de ONSD ───────────────────────────────────────────────────────────

def entrenar_onsd(df_feat: pd.DataFrame) -> dict:
    print("\n── Modelo ONSD a 7 días ───────────────────────────────────")
    X, y, grupos, feature_cols = seleccionar_features(df_feat, "onsd_7d")
    X_tr, X_te, y_tr, y_te = split_por_astronauta(X, y, grupos)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # SGDRegressor (online)
    sgd_r = SGDRegressor(max_iter=1, warm_start=True, random_state=42, tol=None)
    idx = np.random.default_rng(42).permutation(len(X_tr_s))
    tam_lote = max(64, len(X_tr_s) // 20)
    for epoch in range(10):
        for start in range(0, len(idx), tam_lote):
            lote = idx[start:start + tam_lote]
            sgd_r.partial_fit(X_tr_s[lote], y_tr[lote])

    pred = sgd_r.predict(X_te_s)
    mae  = mean_absolute_error(y_te, pred)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    std_res = float(np.std(y_tr - sgd_r.predict(X_tr_s)))
    print(f"  SGDRegressor — MAE: {mae:.4f} mm | RMSE: {rmse:.4f} mm | ±1σ: {std_res:.4f}")

    return {"modelo": sgd_r, "scaler": scaler, "feature_cols": feature_cols,
            "mae": mae, "rmse": rmse, "std_residuos": std_res}


# ─── Modelo de Radiación ──────────────────────────────────────────────────────

def entrenar_radiacion(df_feat: pd.DataFrame) -> dict:
    print("\n── Modelo Radiación a 7 días ──────────────────────────────")
    X, y, grupos, feature_cols = seleccionar_features(df_feat, "radiacion_7d")
    X_tr, X_te, y_tr, y_te = split_por_astronauta(X, y, grupos)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    ridge = Ridge(alpha=0.5)
    ridge.fit(X_tr_s, y_tr)
    pred = ridge.predict(X_te_s)
    mae  = mean_absolute_error(y_te, pred)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    print(f"  Ridge — MAE: {mae:.4f} mSv | RMSE: {rmse:.4f} mSv")

    return {"modelo": ridge, "scaler": scaler, "feature_cols": feature_cols,
            "mae": mae, "rmse": rmse}


# ─── Guardado versionado ──────────────────────────────────────────────────────

def guardar_modelo_versionado(nombre: str, payload: dict) -> Path:
    """Guarda modelo con joblib + versión numerada en la carpeta versiones/."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_version = RUTA_VERSION / f"{nombre}_{ts}.joblib"
    ruta_actual  = RUTA_MODELOS / f"{nombre}.joblib"
    ruta_meta    = RUTA_MODELOS / f"{nombre}_meta.json"

    joblib.dump(payload, ruta_version)
    shutil.copy(ruta_version, ruta_actual)   # copia como "actual"

    # Metadatos legibles
    meta = {
        "nombre": nombre,
        "version": ts,
        "feature_cols": payload.get("feature_cols", []),
        "resultados": payload.get("resultados", {}),
        "mae": payload.get("mae"),
        "rmse": payload.get("rmse"),
        "std_residuos": payload.get("std_residuos"),
        "advertencia": "Modelo sintético para prototipo. Sin validez clínica.",
    }
    with open(ruta_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"  Guardado: {ruta_actual}  (versión: {ts})")
    return ruta_actual


# ─── Punto de entrada ─────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ENTRENAMIENTO BASE — NASA Space Apps Challenge")
    print("ADVERTENCIA: Modelos sintéticos. Sin uso clínico.")
    print("=" * 60)

    if not RUTA_DATOS.exists():
        print("ERROR: datos_mision.csv no encontrado. Ejecuta simulador.py primero.")
        return

    df = pd.read_csv(RUTA_DATOS)
    lb = pd.read_csv(RUTA_LB)
    print(f"Datos: {len(df):,} filas, {df['astronauta_id'].nunique()} astronautas")

    print("\nGenerando características...")
    df_feat = crear_caracteristicas(df, lb)

    RUTA_MODELOS.mkdir(exist_ok=True)
    RUTA_VERSION.mkdir(exist_ok=True)

    # Entrenar
    res_fat = entrenar_fatiga(df_feat)
    res_on  = entrenar_onsd(df_feat)
    res_rad = entrenar_radiacion(df_feat)

    # Guardar
    guardar_modelo_versionado("fatiga", {
        "modelo": res_fat["sgd"],
        "scaler": res_fat["scaler"],
        "feature_cols": res_fat["feature_cols"],
        "clases": res_fat["clases"],
        "resultados": res_fat["resultados"]
    })
    guardar_modelo_versionado("onsd", {
        "modelo": res_on["modelo"],
        "scaler": res_on["scaler"],
        "feature_cols": res_on["feature_cols"],
        "mae": res_on["mae"],
        "rmse": res_on["rmse"],
        "std_residuos": res_on["std_residuos"]
    })
    guardar_modelo_versionado("radiacion", {
        "modelo": res_rad["modelo"],
        "scaler": res_rad["scaler"],
        "feature_cols": res_rad["feature_cols"],
        "mae": res_rad["mae"],
        "rmse": res_rad["rmse"]
    })

    # También guardar el scaler + feature_cols del modelo de fatiga como referencia global
    joblib.dump({
        "scaler": res_fat["scaler"],
        "feature_cols": res_fat["feature_cols"],
        "clases": res_fat["clases"]
    }, RUTA_MODELOS / "referencia_global.joblib")

    print("\n" + "=" * 60)
    print("RESUMEN")
    print(f"  Fatiga SGD — F1 macro: {res_fat['resultados']['sgd']['f1_macro']:.3f} | F1 rojo: {res_fat['resultados']['sgd']['f1_rojo']:.3f}")
    print(f"  Fatiga GB  — F1 macro: {res_fat['resultados']['gb']['f1_macro']:.3f}  | F1 rojo: {res_fat['resultados']['gb']['f1_rojo']:.3f}")
    print(f"  ONSD       — MAE: {res_on['mae']:.4f} mm")
    print(f"  Radiación  — MAE: {res_rad['mae']:.4f} mSv")
    print("\nModelos en:", RUTA_MODELOS)


if __name__ == "__main__":
    main()

