"""
servidor.py — Backend local FastAPI + SQLite para AstroSalud.
Plan 4: Servidor local 100% offline para Raspberry Pi.
Soporta:
- Servicio de la UI táctil (index.html)
- Telemetría y predicciones continuas
- Validación médica con PIN
- Gestión de candidatos de modelos y auditoría
- Motor de consejos de bienestar personalizados

Uso:
    python servidor.py
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException, Request, Body
    from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    from pydantic import BaseModel
    from typing import Optional, List, Dict, Any
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

RUTA_BASE  = Path(__file__).parent
RUTA_APP   = RUTA_BASE / "app"
RUTA_DB    = RUTA_BASE / "datos" / "astrosalud.db"
RUTA_MOD   = RUTA_BASE / "modelos"
PUERTO     = 8000

# ─── Inicialización de módulos ────────────────────────────────────────────────
try:
    from aprendizaje import (
        inicializar_db, guardar_lectura, validar_lectura,
        predecir, actualizar_modelo, restaurar_modelo_anterior,
        get_config, set_config, preparar_features_desde_db
    )
    from validacion import (
        inicializar_tablas_validacion, verificar_pin,
        sincronizar_cola_pendientes, obtener_pendientes,
        resolver_validacion, generar_modelo_candidato,
        decidir_modelo_candidato, obtener_candidatos,
        registrar_feedback_consejo, obtener_consejos_para_metricas
    )
    MODULO_APRENDIZAJE_OK = True
except ImportError as e:
    MODULO_APRENDIZAJE_OK = False
    print(f"AVISO: módulo de aprendizaje o validación no disponible: {e}")


def estado_modelo() -> dict:
    """Resumen del estado del modelo para la UI y la barra de estado."""
    dias  = get_config("dias_datos_reales", "0") if MODULO_APRENDIZAJE_OK else "0"
    est   = get_config("modelo_estado", "inicial") if MODULO_APRENDIZAJE_OK else "inicial"
    n_ver = len(list((RUTA_MOD / "versiones").glob("fatiga_*.joblib"))) if (RUTA_MOD / "versiones").exists() else 0
    con = sqlite3.connect(RUTA_DB)
    con.row_factory = sqlite3.Row
    pendientes_cnt = 0
    candidatos_cnt = 0
    try:
        pendientes_cnt = con.execute("SELECT COUNT(*) as c FROM cola_validacion WHERE estado='pendiente'").fetchone()["c"]
        candidatos_cnt = con.execute("SELECT COUNT(*) as c FROM modelos_candidatos WHERE estado='pendiente_aprobacion'").fetchone()["c"]
    except Exception:
        pass
    con.close()

    return {
        "estado": est,
        "dias_datos_reales": int(dias),
        "n_versiones": n_ver,
        "pendientes_validacion": pendientes_cnt,
        "candidatos_pendientes": candidatos_cnt,
        "modelos_disponibles": {
            "fatiga":    (RUTA_MOD / "fatiga.joblib").exists(),
            "onsd":      (RUTA_MOD / "onsd.joblib").exists(),
            "radiacion": (RUTA_MOD / "radiacion.joblib").exists(),
        }
    }


# ─── FastAPI App ──────────────────────────────────────────────────────────────

if HAS_FASTAPI:
    app = FastAPI(title="AstroSalud API v4", version="4.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    class LecturaIn(BaseModel):
        astronauta_id:       str   = "AST-001"
        dia_mision:          Optional[int]   = None
        sistolica:           Optional[float] = None
        diastolica:          Optional[float] = None
        fc:                  Optional[float] = None
        hrv:                 Optional[float] = None
        spo2:                Optional[float] = None
        temperatura:         Optional[float] = None
        glucosa:             Optional[float] = None
        sueno_horas:         Optional[float] = None
        sueno_calidad:       Optional[float] = None
        animo:               Optional[float] = None
        carga_trabajo:       Optional[float] = None
        ejercicio_min:       Optional[float] = None
        onsd_mm:             Optional[float] = None
        radiacion_diaria:    Optional[float] = None
        radiacion_acumulada: Optional[float] = None
        marcado_erroneo:     int  = 0
        fuente:              str  = "manual"

    class PinIn(BaseModel):
        pin: str

    class ResolucionIn(BaseModel):
        item_id: int
        accion: str # 'confirmar', 'corregir', 'descartar'
        usuario: str = "Oficial Médico (CMO)"
        motivo: str = ""
        fatiga_corregida: Optional[int] = None
        onsd_corregido: Optional[float] = None

    class CandidatoDecisionIn(BaseModel):
        candidato_id: int
        aprobar: bool
        usuario: str = "Oficial Médico (CMO)"
        motivo: str = ""

    class FeedbackConsejoIn(BaseModel):
        consejo_id: str
        astronauta_id: str = "AST-001"
        util: bool
        dia_mision: Optional[int] = None

    # ── Servir app HTML ───────────────────────────────────────────────────────
    @app.get("/", response_class=HTMLResponse)
    async def raiz():
        idx = RUTA_APP / "index.html"
        if idx.exists():
            return HTMLResponse(idx.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>AstroSalud — index.html no encontrado</h1>", status_code=404)

    # ── Estado del sistema ────────────────────────────────────────────────────
    @app.get("/api/estado")
    async def api_estado():
        return JSONResponse(estado_modelo())

    # ── Lecturas ──────────────────────────────────────────────────────────────
    @app.post("/api/lecturas")
    async def api_guardar_lectura(lectura: LecturaIn):
        datos = lectura.dict()
        datos["timestamp"] = datetime.now().isoformat()

        advertencias = validar_lectura(datos) if MODULO_APRENDIZAJE_OK else []
        if advertencias and not datos.get("marcado_erroneo"):
            return JSONResponse({
                "advertencias": advertencias,
                "accion": "confirmar_o_marcar_erroneo"
            }, status_code=422)

        if not MODULO_APRENDIZAJE_OK:
            return JSONResponse({"error": "Módulo de aprendizaje no disponible"}, status_code=503)

        lid = guardar_lectura(datos)
        pred = predecir(datos, datos["astronauta_id"])

        # Guardar la predicción en la tabla de predicciones para seguimiento a 7 días
        con = sqlite3.connect(RUTA_DB)
        con.execute(
            """INSERT INTO predicciones (lectura_id, timestamp, modelo_version, fatiga_pred,
                                         fatiga_prob_v, fatiga_prob_a, fatiga_prob_r, onsd_pred, onsd_std)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                lid, datos["timestamp"], pred.get("modelo_version", "v4"),
                pred.get("fatiga_7d"),
                pred.get("fatiga_prob", {}).get("verde", 0),
                pred.get("fatiga_prob", {}).get("amarillo", 0),
                pred.get("fatiga_prob", {}).get("rojo", 0),
                pred.get("onsd_7d"),
                pred.get("onsd_std")
            )
        )
        con.commit()
        con.close()

        # Revisar si se generan pares pendientes
        sincronizar_cola_pendientes(datos["astronauta_id"])

        return JSONResponse({
            "lectura_id": lid,
            "prediccion": pred,
            "estado_modelo": estado_modelo()
        })

    @app.get("/api/lecturas")
    async def api_leer_lecturas(astronauta_id: str = "AST-001", n: int = 30):
        con = sqlite3.connect(RUTA_DB)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT * FROM lecturas WHERE astronauta_id=? AND marcado_erroneo=0 ORDER BY dia_mision DESC LIMIT ?",
            (astronauta_id, n)
        ).fetchall()
        con.close()
        return JSONResponse([dict(r) for r in rows])

    # ── Predicción ────────────────────────────────────────────────────────────
    @app.post("/api/predecir")
    async def api_predecir(lectura: LecturaIn):
        if not MODULO_APRENDIZAJE_OK:
            return JSONResponse({"error": "Módulo no disponible"}, status_code=503)
        pred = predecir(lectura.dict(), lectura.astronauta_id)
        return JSONResponse(pred)

    # ── Validación Médica y Cola Humana (Plan 4) ──────────────────────────────
    @app.post("/api/validacion/verificar-pin")
    async def api_verificar_pin(pin_in: PinIn):
        es_valido = verificar_pin(pin_in.pin) if MODULO_APRENDIZAJE_OK else (pin_in.pin == "1234")
        return JSONResponse({"valido": es_valido})

    @app.get("/api/validacion/pendientes")
    async def api_pendientes(astronauta_id: Optional[str] = "AST-001"):
        if not MODULO_APRENDIZAJE_OK:
            return JSONResponse([])
        sincronizar_cola_pendientes(astronauta_id)
        pendientes = obtener_pendientes(astronauta_id)
        return JSONResponse(pendientes)

    @app.post("/api/validacion/resolver")
    async def api_resolver(res: ResolucionIn):
        if not MODULO_APRENDIZAJE_OK:
            return JSONResponse({"error": "Módulo no disponible"}, status_code=503)
        resp = resolver_validacion(
            item_id=res.item_id,
            accion=res.accion,
            usuario=res.usuario,
            motivo=res.motivo,
            fatiga_corregida=res.fatiga_corregida,
            onsd_corregido=res.onsd_corregido
        )
        return JSONResponse(resp)

    @app.get("/api/validacion/candidatos")
    async def api_candidatos():
        if not MODULO_APRENDIZAJE_OK:
            return JSONResponse([])
        return JSONResponse(obtener_candidatos())

    @app.post("/api/validacion/candidato/generar")
    async def api_candidato_generar(astronauta_id: str = "AST-001", usuario: str = "CMO"):
        if not MODULO_APRENDIZAJE_OK:
            return JSONResponse({"error": "Módulo no disponible"}, status_code=503)
        resp = generar_modelo_candidato(astronauta_id, usuario)
        return JSONResponse(resp)

    @app.post("/api/validacion/candidato/decidir")
    async def api_candidato_decidir(dec: CandidatoDecisionIn):
        if not MODULO_APRENDIZAJE_OK:
            return JSONResponse({"error": "Módulo no disponible"}, status_code=503)
        resp = decidir_modelo_candidato(
            candidato_id=dec.candidato_id,
            aprobar=dec.aprobar,
            usuario=dec.usuario,
            motivo=dec.motivo
        )
        return JSONResponse(resp)

    @app.get("/api/validacion/auditoria")
    async def api_auditoria(limite: int = 50):
        con = sqlite3.connect(RUTA_DB)
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute("SELECT * FROM auditoria_validacion ORDER BY id DESC LIMIT ?", (limite,)).fetchall()
            res = [dict(r) for r in rows]
        except Exception:
            res = []
        con.close()
        return JSONResponse(res)

    # ── Motor de Consejos (Plan 4) ────────────────────────────────────────────
    @app.post("/api/consejos/obtener")
    async def api_obtener_consejos(payload: Dict[str, Any] = Body(...)):
        ult_lectura = payload.get("lectura", {})
        lineas_base = payload.get("lineas_base", {})
        estado_global = payload.get("estado_global", "verde")

        if not MODULO_APRENDIZAJE_OK:
            return JSONResponse([])

        consejos = obtener_consejos_para_metricas(ult_lectura, lineas_base, estado_global)
        return JSONResponse(consejos)

    @app.post("/api/consejos/feedback")
    async def api_feedback_consejo(fb: FeedbackConsejoIn):
        if MODULO_APRENDIZAJE_OK:
            registrar_feedback_consejo(fb.consejo_id, fb.astronauta_id, fb.util, fb.dia_mision)
        return JSONResponse({"ok": True})

    # ── Restaurar modelo ──────────────────────────────────────────────────────
    @app.post("/api/modelo/restaurar")
    async def api_restaurar(nombre: str = "fatiga"):
        if not MODULO_APRENDIZAJE_OK:
            return JSONResponse({"error": "Módulo no disponible"}, status_code=503)
        ok = restaurar_modelo_anterior(nombre)
        return JSONResponse({"exito": ok, "estado": estado_modelo()})

    # ── Exportar paquete Tierra ───────────────────────────────────────────────
    @app.get("/api/exportar/{astronauta_id}")
    async def api_exportar(astronauta_id: str):
        con = sqlite3.connect(RUTA_DB)
        con.row_factory = sqlite3.Row
        lecturas = con.execute(
            "SELECT * FROM lecturas WHERE astronauta_id=? ORDER BY dia_mision DESC LIMIT 30",
            (astronauta_id,)
        ).fetchall()
        preds = con.execute(
            """SELECT p.* FROM predicciones p
               JOIN lecturas l ON p.lectura_id=l.id
               WHERE l.astronauta_id=? ORDER BY p.timestamp DESC LIMIT 10""",
            (astronauta_id,)
        ).fetchall()
        validaciones = []
        try:
            validaciones = [dict(r) for r in con.execute(
                "SELECT * FROM cola_validacion WHERE astronauta_id=? ORDER BY id DESC LIMIT 20",
                (astronauta_id,)
            ).fetchall()]
        except Exception:
            pass
        con.close()

        paquete = {
            "_advertencia": "Datos sintéticos de prototipo. Sin validez clínica.",
            "_sistema": "AstroSalud v4.0 — NASA Space Apps Challenge 2026",
            "timestamp_exportacion": datetime.now().isoformat(),
            "astronauta_id": astronauta_id,
            "estado_modelo": estado_modelo(),
            "lecturas_recientes": [dict(r) for r in lecturas],
            "predicciones_recientes": [dict(p) for p in preds],
            "cola_validaciones_reciente": validaciones
        }
        return JSONResponse(paquete)

    # ── Simulación telemetría ─────────────────────────────────────────────────
    @app.post("/api/simular")
    async def api_simular(n_dias: int = 7, astronauta_id: str = "AST-001"):
        import numpy as np
        rng = np.random.default_rng()

        base = {
            "sistolica": 115, "diastolica": 72, "fc": 62, "hrv": 65,
            "spo2": 98.0, "temperatura": 36.6, "glucosa": 85,
            "sueno_horas": 7.0, "sueno_calidad": 3.8, "animo": 4.0,
            "carga_trabajo": 2.5, "ejercicio_min": 120, "onsd_mm": 4.3,
            "radiacion_diaria": 0.35, "radiacion_acumulada": 20.0
        }

        dias_actual = int(get_config("dia_mision_actual", "1")) if MODULO_APRENDIZAJE_OK else 1
        resultados = []

        for d in range(n_dias):
            dia = dias_actual + d
            lectura = {
                "astronauta_id": astronauta_id,
                "timestamp": datetime.now().isoformat(),
                "dia_mision": dia,
                "fuente": "simulador",
                **{k: float(round(base[k] + rng.normal(0, abs(base[k]) * 0.05), 2))
                   for k in base}
            }
            if MODULO_APRENDIZAJE_OK:
                lid = guardar_lectura(lectura)
                pred = predecir(lectura, astronauta_id)
                con = sqlite3.connect(RUTA_DB)
                con.execute(
                    """INSERT INTO predicciones (lectura_id, timestamp, modelo_version, fatiga_pred,
                                                 fatiga_prob_v, fatiga_prob_a, fatiga_prob_r, onsd_pred, onsd_std)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        lid, lectura["timestamp"], "v4",
                        pred.get("fatiga_7d", 0),
                        pred.get("fatiga_prob", {}).get("verde", 0.8),
                        pred.get("fatiga_prob", {}).get("amarillo", 0.15),
                        pred.get("fatiga_prob", {}).get("rojo", 0.05),
                        pred.get("onsd_7d", 4.3),
                        pred.get("onsd_std", 0.1)
                    )
                )
                con.commit()
                con.close()
                resultados.append({"dia": dia, "lectura_id": lid})

        if MODULO_APRENDIZAJE_OK:
            set_config("dia_mision_actual", dias_actual + n_dias)
            sincronizar_cola_pendientes(astronauta_id)

        return JSONResponse({"simulados": n_dias, "dias": resultados, "estado_modelo": estado_modelo()})


# ─── Punto de entrada ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("AstroSalud v4 — Servidor local 100% offline (Raspberry Pi)")
    print("ADVERTENCIA: Prototipo para validación. Sin uso clínico.")
    print("=" * 60)

    if MODULO_APRENDIZAJE_OK:
        inicializar_db()
        print(f"  BD SQLite: {RUTA_DB}")

    print(f"  App táctil: {RUTA_APP / 'index.html'}")
    print(f"  Modelos:    {RUTA_MOD}")
    print(f"  Estado:     {estado_modelo()['estado']}")

    if HAS_FASTAPI:
        print(f"\n  Iniciando en: http://localhost:{PUERTO}")
        print("  Documentación API: http://localhost:8000/docs")
        uvicorn.run(app, host="0.0.0.0", port=PUERTO, log_level="warning")
    else:
        print("ERROR: FastAPI no disponible. Instalar con pip install fastapi uvicorn")
