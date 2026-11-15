"""
validacion.py — Sistema de Validación Humana de Retroalimentación y Modelos Candidatos.
NASA Space Apps Challenge — Plan 4.

Principio: El modelo NUNCA aprende de datos que un humano no haya validado.
Todo cambio pasa por cola de validación protegida por PIN y registro de auditoría.
"""

import json
import sqlite3
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

RUTA_BASE = Path(__file__).parent
RUTA_DB = RUTA_BASE / "datos" / "astrosalud.db"
RUTA_MODELOS = RUTA_BASE / "modelos"
RUTA_VERSIONES = RUTA_MODELOS / "versiones"
RUTA_CONSEJOS = RUTA_BASE / "consejos.json"

PIN_POR_DEFECTO = "1234"


def get_db():
    con = sqlite3.connect(RUTA_DB)
    con.row_factory = sqlite3.Row
    return con


def inicializar_tablas_validacion():
    """Inicializa las tablas de cola de validación, auditoría y modelos candidatos."""
    RUTA_DB.parent.mkdir(parents=True, exist_ok=True)
    con = get_db()
    cur = con.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS cola_validacion (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            astronauta_id           TEXT NOT NULL,
            prediccion_id           INTEGER,
            lectura_id_origen       INTEGER,
            dia_prediccion          INTEGER,
            dia_objetivo            INTEGER,
            fatiga_pred             INTEGER,
            fatiga_real_observada   INTEGER,
            fatiga_validada         INTEGER,
            onsd_pred               REAL,
            onsd_real_observado     REAL,
            onsd_validado           REAL,
            estado                  TEXT DEFAULT 'pendiente', -- 'pendiente', 'confirmado', 'corregido', 'descartado'
            validado_por            TEXT,
            fecha_validacion        TEXT,
            motivo                  TEXT,
            usado_entrenamiento     INTEGER DEFAULT 0,
            fecha_creacion          TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS auditoria_validacion (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            usuario     TEXT NOT NULL,
            accion      TEXT NOT NULL,
            entidad     TEXT NOT NULL,
            entidad_id  INTEGER,
            detalles    TEXT
        );

        CREATE TABLE IF NOT EXISTS modelos_candidatos (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp               TEXT NOT NULL,
            nombre_modelo           TEXT NOT NULL,
            ruta_archivo            TEXT NOT NULL,
            f1_macro                REAL,
            f1_rojo                 REAL,
            mae_onsd                REAL,
            f1_vigente              REAL,
            f1_rojo_vigente         REAL,
            n_muestras_validadas    INTEGER,
            estado                  TEXT DEFAULT 'pendiente_aprobacion', -- 'pendiente_aprobacion', 'aprobado', 'rechazado'
            aprobado_por            TEXT,
            fecha_decision          TEXT,
            motivo_decision         TEXT
        );

        CREATE TABLE IF NOT EXISTS feedback_consejos (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            consejo_id      TEXT NOT NULL,
            astronauta_id   TEXT NOT NULL,
            dia_mision      INTEGER,
            util            INTEGER, -- 1: útil, 0: no útil
            timestamp       TEXT NOT NULL,
            validado        INTEGER DEFAULT 0,
            validado_por    TEXT
        );
    """)

    # Configuración de PIN
    cur.execute("CREATE TABLE IF NOT EXISTS configuracion (clave TEXT PRIMARY KEY, valor TEXT);")
    cur.execute("INSERT OR IGNORE INTO configuracion VALUES ('pin_validacion', ?)", (PIN_POR_DEFECTO,))
    cur.execute("INSERT OR IGNORE INTO configuracion VALUES ('cmo_nombre', 'Dr. Oficial Médico (CMO)');")
    con.commit()
    con.close()


def verificar_pin(pin: str) -> bool:
    """Verifica si el PIN suministrado coincide con el configurado."""
    con = get_db()
    r = con.execute("SELECT valor FROM configuracion WHERE clave='pin_validacion'").fetchone()
    con.close()
    pin_valido = r["valor"] if r else PIN_POR_DEFECTO
    return str(pin).strip() == str(pin_valido).strip()


def registrar_auditoria(usuario: str, accion: str, entidad: str, entidad_id: Optional[int], detalles: dict):
    """Guarda registro inmutable de auditoría para decisiones clínicas y de aprendizaje."""
    con = get_db()
    con.execute(
        """INSERT INTO auditoria_validacion (timestamp, usuario, accion, entidad, entidad_id, detalles)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (datetime.now().isoformat(), usuario, accion, entidad, entidad_id, json.dumps(detalles, ensure_ascii=False))
    )
    con.commit()
    con.close()


def sincronizar_cola_pendientes(astronauta_id: str = "AST-001") -> int:
    """
    Busca lecturas que hicieron una predicción hace 7 días (o más) y que ya cuentan con
    datos observados en el día objetivo, para proponerlas a la cola de validación humana.
    """
    inicializar_tablas_validacion()
    con = get_db()

    # Obtener lecturas que tengan día de misión y ver si existe lectura 7 días después
    query = """
        SELECT l1.id as l1_id, l1.dia_mision as dia_orig,
               l2.id as l2_id, l2.dia_mision as dia_obj,
               l2.sueno_horas, l2.hrv, l2.animo, l2.carga_trabajo, l2.onsd_mm
        FROM lecturas l1
        JOIN lecturas l2 ON l1.astronauta_id = l2.astronauta_id
                        AND l2.dia_mision >= l1.dia_mision + 7
                        AND l2.dia_mision <= l1.dia_mision + 8
        WHERE l1.astronauta_id = ?
          AND l1.marcado_erroneo = 0
          AND l2.marcado_erroneo = 0
          AND NOT EXISTS (
              SELECT 1 FROM cola_validacion cv
              WHERE cv.astronauta_id = l1.astronauta_id
                AND cv.dia_prediccion = l1.dia_mision
          )
        ORDER BY l1.dia_mision ASC
    """
    filas = con.execute(query, (astronauta_id,)).fetchall()
    insertados = 0

    for f in filas:
        dia_orig = f["dia_orig"]
        dia_obj = f["dia_obj"]

        # Calcular fatiga real aproximada observada
        s = f["sueno_horas"] or 7.0
        h = f["hrv"] or 60.0
        a = f["animo"] or 4.0
        c = f["carga_trabajo"] or 2.5
        sc = ((7.5 - s) / 5.5) * 0.3 + ((90.0 - h) / 80.0) * 0.3 + ((5.0 - a) / 4.0) * 0.2 + ((c - 1.0) / 4.0) * 0.2
        fatiga_obs = 0 if sc < 0.35 else (1 if sc < 0.60 else 2)

        # Buscar si hubo predicción previa registrada
        pred = con.execute(
            "SELECT id, fatiga_pred, onsd_pred FROM predicciones WHERE lectura_id = ?",
            (f["l1_id"],)
        ).fetchone()

        fatiga_p = pred["fatiga_pred"] if pred else fatiga_obs
        onsd_p = pred["onsd_pred"] if pred else (f["onsd_mm"] or 4.3)
        pred_id = pred["id"] if pred else None

        con.execute(
            """INSERT INTO cola_validacion (
                astronauta_id, prediccion_id, lectura_id_origen,
                dia_prediccion, dia_objetivo, fatiga_pred,
                fatiga_real_observada, fatiga_validada,
                onsd_pred, onsd_real_observado, onsd_validado,
                estado, fecha_creacion
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pendiente', ?)""",
            (
                astronauta_id, pred_id, f["l1_id"],
                dia_orig, dia_obj, fatiga_p,
                fatiga_obs, fatiga_obs,
                onsd_p, f["onsd_mm"], f["onsd_mm"],
                datetime.now().isoformat()
            )
        )
        insertados += 1

    con.commit()
    con.close()
    return insertados


def obtener_pendientes(astronauta_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Devuelve los casos de la cola pendientes de revisión humana."""
    inicializar_tablas_validacion()
    con = get_db()
    if astronauta_id:
        rows = con.execute(
            "SELECT * FROM cola_validacion WHERE estado='pendiente' AND astronauta_id=? ORDER BY dia_prediccion ASC",
            (astronauta_id,)
        ).fetchall()
    else:
        rows = con.execute(
            "SELECT * FROM cola_validacion WHERE estado='pendiente' ORDER BY dia_prediccion ASC"
        ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def resolver_validacion(
    item_id: int,
    accion: str,
    usuario: str,
    motivo: str,
    fatiga_corregida: Optional[int] = None,
    onsd_corregido: Optional[float] = None
) -> Dict[str, Any]:
    """
    Ejecuta la decisión médica de validación:
    - 'confirmar': El humano aprueba la etiqueta observada.
    - 'corregir': El humano ajusta la etiqueta real.
    - 'descartar': El humano descarta el registro por artefacto o error.
    """
    inicializar_tablas_validacion()
    con = get_db()
    item = con.execute("SELECT * FROM cola_validacion WHERE id=?", (item_id,)).fetchone()
    if not item:
        con.close()
        return {"error": "Registro no encontrado"}

    accion = accion.lower().strip()
    if accion not in ["confirmar", "corregir", "descartar"]:
        con.close()
        return {"error": "Acción inválida. Use confirmar, corregir o descartar."}

    fatiga_val = item["fatiga_real_observada"]
    onsd_val = item["onsd_real_observado"]

    if accion == "confirmar":
        estado = "confirmado"
    elif accion == "corregir":
        estado = "corregido"
        if fatiga_corregida is not None:
            fatiga_val = int(fatiga_corregida)
        if onsd_corregido is not None:
            onsd_val = float(onsd_corregido)
    else: # descartar
        estado = "descartado"

    ahora = datetime.now().isoformat()
    con.execute(
        """UPDATE cola_validacion
           SET estado=?, fatiga_validada=?, onsd_validado=?,
               validado_por=?, fecha_validacion=?, motivo=?
           WHERE id=?""",
        (estado, fatiga_val, onsd_val, usuario, ahora, motivo, item_id)
    )
    con.commit()
    con.close()

    registrar_auditoria(
        usuario=usuario,
        accion=f"validacion_{estado}",
        entidad="cola_validacion",
        entidad_id=item_id,
        detalles={
            "accion": accion,
            "estado_resultante": estado,
            "fatiga_validada": fatiga_val,
            "onsd_validado": onsd_val,
            "motivo": motivo
        }
    )

    return {
        "id": item_id,
        "estado": estado,
        "fatiga_validada": fatiga_val,
        "onsd_validado": onsd_val,
        "mensaje": f"Registro {item_id} resuelto como {estado} por {usuario}."
    }


def generar_modelo_candidato(astronauta_id: str, usuario: str) -> Dict[str, Any]:
    """
    Entrena un modelo CANDIDATO usando EXCLUSIVAMENTE datos validados por humanos
    (confirmados o corregidos). Compara con el modelo vigente y lo somete a aprobación humana.
    """
    inicializar_tablas_validacion()
    import copy
    import joblib
    import numpy as np
    from sklearn.metrics import f1_score

    con = get_db()
    # Buscar datos validados que no hayan sido incorporados a producción
    query = """
        SELECT cv.*, l.sistolica, l.diastolica, l.fc, l.hrv, l.spo2,
               l.temperatura, l.glucosa, l.sueno_horas, l.sueno_calidad,
               l.animo, l.carga_trabajo, l.ejercicio_min, l.radiacion_diaria
        FROM cola_validacion cv
        JOIN lecturas l ON cv.lectura_id_origen = l.id
        WHERE cv.estado IN ('confirmado', 'corregido')
          AND cv.usado_entrenamiento = 0
          AND cv.astronauta_id = ?
    """
    validados = con.execute(query, (astronauta_id,)).fetchall()

    if len(validados) < 3:
        con.close()
        return {
            "error": "sin_datos_suficientes",
            "mensaje": f"Se requieren al menos 3 registros validados para proponer un modelo candidato. Hay {len(validados)}.",
            "n_validados": len(validados),
            "minimo": 3
        }

    # Cargar modelo vigente
    ruta_vigente = RUTA_MODELOS / "fatiga.joblib"
    if not ruta_vigente.exists():
        con.close()
        return {"error": "modelo_vigente_no_encontrado"}

    modelo_payload = joblib.load(ruta_vigente)
    sgd_vigente = modelo_payload["modelo"]
    scaler = modelo_payload["scaler"]
    feature_cols = modelo_payload["feature_cols"]

    # Preparar matriz de features para las muestras validadas
    # Construir aproximación de features para las muestras validadas
    X_rows = []
    y_rows = []
    for row in validados:
        y_rows.append(int(row["fatiga_validada"]))
        # Vector base
        feat_dict = dict(row)
        vec = []
        for col in feature_cols:
            if col in feat_dict and feat_dict[col] is not None:
                vec.append(float(feat_dict[col]))
            else:
                vec.append(0.0)
        X_rows.append(vec)

    X_arr = np.array(X_rows, dtype=np.float32)
    y_arr = np.array(y_rows, dtype=np.float32)

    X_s = scaler.transform(X_arr)

    # Evaluar modelo vigente sobre estos datos
    pred_vig = sgd_vigente.predict(X_s)
    f1_vig = float(f1_score(y_arr, pred_vig, average="macro", zero_division=0))
    f1r_vig = float(f1_score(y_arr, pred_vig, labels=[2], average="macro", zero_division=0))

    # Entrenar copia candidata
    sgd_candidato = copy.deepcopy(sgd_vigente)
    clases = modelo_payload.get("clases", np.array([0.0, 1.0, 2.0]))
    sgd_candidato.partial_fit(X_s, y_arr, classes=clases)

    # Evaluar modelo candidato
    pred_cand = sgd_candidato.predict(X_s)
    f1_cand = float(f1_score(y_arr, pred_cand, average="macro", zero_division=0))
    f1r_cand = float(f1_score(y_arr, pred_cand, labels=[2], average="macro", zero_division=0))

    # Guardar archivo del modelo candidato
    RUTA_VERSIONES.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_cand = RUTA_VERSIONES / f"candidato_fatiga_{ts}.joblib"

    payload_candidato = copy.deepcopy(modelo_payload)
    payload_candidato["modelo"] = sgd_candidato
    payload_candidato["version_candidata"] = ts
    payload_candidato["n_muestras_validadas"] = len(validados)
    joblib.dump(payload_candidato, ruta_cand)

    # Registrar en tabla modelos_candidatos
    cur = con.cursor()
    cur.execute(
        """INSERT INTO modelos_candidatos (
            timestamp, nombre_modelo, ruta_archivo, f1_macro, f1_rojo,
            f1_vigente, f1_rojo_vigente, n_muestras_validadas, estado
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pendiente_aprobacion')""",
        (
            datetime.now().isoformat(), "fatiga", str(ruta_cand),
            f1_cand, f1r_cand, f1_vig, f1r_vig, len(validados)
        )
    )
    cid = cur.lastrowid
    con.commit()
    con.close()

    registrar_auditoria(
        usuario=usuario,
        accion="generar_modelo_candidato",
        entidad="modelos_candidatos",
        entidad_id=cid,
        detalles={
            "ruta_archivo": str(ruta_cand),
            "f1_candidato": f1_cand,
            "f1_vigente": f1_vig,
            "f1_rojo_candidato": f1r_cand,
            "f1_rojo_vigente": f1r_vig,
            "n_muestras": len(validados)
        }
    )

    return {
        "candidato_id": cid,
        "estado": "pendiente_aprobacion",
        "f1_candidato": f1_cand,
        "f1_vigente": f1_vig,
        "f1_rojo_candidato": f1r_cand,
        "f1_rojo_vigente": f1r_vig,
        "n_muestras": len(validados),
        "mensaje": "Modelo candidato generado y en cola para aprobación humana."
    }


def decidir_modelo_candidato(candidato_id: int, aprobar: bool, usuario: str, motivo: str) -> Dict[str, Any]:
    """El humano aprueba o rechaza formalmente la activación de un modelo candidato."""
    inicializar_tablas_validacion()
    con = get_db()
    c = con.execute("SELECT * FROM modelos_candidatos WHERE id=?", (candidato_id,)).fetchone()
    if not c:
        con.close()
        return {"error": "Candidato no encontrado"}

    ahora = datetime.now().isoformat()
    nuevo_estado = "aprobado" if aprobar else "rechazado"

    if aprobar:
        # Activar el modelo candidato copiándolo a modelos/fatiga.joblib
        ruta_origen = Path(c["ruta_archivo"])
        ruta_destino = RUTA_MODELOS / "fatiga.joblib"
        if ruta_origen.exists():
            shutil.copy(ruta_origen, ruta_destino)
            # Marcar registros validados como incorporados
            con.execute("UPDATE cola_validacion SET usado_entrenamiento=1 WHERE estado IN ('confirmado', 'corregido')")

    con.execute(
        """UPDATE modelos_candidatos
           SET estado=?, aprobado_por=?, fecha_decision=?, motivo_decision=?
           WHERE id=?""",
        (nuevo_estado, usuario, ahora, motivo, candidato_id)
    )
    con.commit()
    con.close()

    registrar_auditoria(
        usuario=usuario,
        accion=f"decision_candidato_{nuevo_estado}",
        entidad="modelos_candidatos",
        entidad_id=candidato_id,
        detalles={"aprobado": aprobar, "motivo": motivo}
    )

    return {
        "candidato_id": candidato_id,
        "estado": nuevo_estado,
        "mensaje": f"Modelo candidato {candidato_id} marcado como {nuevo_estado} por {usuario}."
    }


def obtener_candidatos() -> List[Dict[str, Any]]:
    """Lista todos los modelos candidatos para revisión."""
    inicializar_tablas_validacion()
    con = get_db()
    rows = con.execute("SELECT * FROM modelos_candidatos ORDER BY id DESC LIMIT 10").fetchall()
    con.close()
    return [dict(r) for r in rows]


def registrar_feedback_consejo(consejo_id: str, astronauta_id: str, util: bool, dia_mision: Optional[int] = None):
    """Guarda si un consejo fue útil ('me sirvió' / 'no me sirvió')."""
    inicializar_tablas_validacion()
    con = get_db()
    con.execute(
        """INSERT INTO feedback_consejos (consejo_id, astronauta_id, dia_mision, util, timestamp)
           VALUES (?, ?, ?, ?, ?)""",
        (consejo_id, astronauta_id, dia_mision or 1, 1 if util else 0, datetime.now().isoformat())
    )
    con.commit()
    con.close()


def obtener_consejos_para_metricas(ult_lectura: dict, lineas_base: dict, estado_global: str) -> List[Dict[str, Any]]:
    """
    Selecciona hasta 3 consejos personalizados según las métricas actuales y sus desviaciones.
    REGLA DE SEGURIDAD OBLIGATORIA: En estado crítico (rojo), el primer consejo es SIEMPRE
    contactar inmediatamente al oficial médico o cirujano de vuelo.
    """
    if not RUTA_CONSEJOS.exists():
        return []

    try:
        with open(RUTA_CONSEJOS, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return []

    catalogo = cfg.get("catalogo", [])
    consejos_seleccionados = []

    # 1. Regla crítica de emergencia si hay algún estado rojo
    if estado_global == "rojo":
        regla_urgente = cfg.get("regla_emergencia_roja", {
            "texto": "ALERTA CRÍTICA: Notificar de inmediato al Oficial Médico de la Tripulación (CMO) o al Cirujano de Vuelo en Tierra.",
            "evidencia": "alta"
        })
        consejos_seleccionados.append({
            "id": "emergencia_medica_inmediata",
            "titulo": "🚨 ATENCIÓN MÉDICA INMEDIATA",
            "texto": regla_urgente.get("texto"),
            "evidencia": regla_urgente.get("evidencia", "alta"),
            "es_critico": True
        })

    # 2. Evaluar condiciones del catálogo
    for c in catalogo:
        metrica = c.get("metrica")
        val = ult_lectura.get(metrica)
        base = lineas_base.get(metrica, lineas_base.get(f"base_{metrica}"))

        if val is None or base is None:
            continue

        std = max(base * 0.1, 1.0)
        cond = c.get("condicion", "")
        # Evaluación segura de la condición
        contexto = {
            "valor": float(val),
            "base": float(base),
            "std": float(std),
            "abs": abs,
            "carga_trabajo": float(ult_lectura.get("carga_trabajo", 2.5))
        }

        try:
            aplica = eval(cond, {"__builtins__": {}}, contexto)
        except Exception:
            aplica = False

        if aplica:
            plantilla = c.get("plantilla", "")
            texto = plantilla.format(valor=val, base=base, std=std)
            consejos_seleccionados.append({
                "id": c.get("id"),
                "titulo": c.get("titulo"),
                "texto": texto,
                "evidencia": c.get("evidencia", "media"),
                "fuente": c.get("fuente"),
                "es_critico": False
            })

        if len(consejos_seleccionados) >= 3:
            break

    return consejos_seleccionados[:3]

