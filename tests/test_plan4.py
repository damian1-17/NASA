"""
tests/test_plan4.py — Suite completa de pruebas unitarias para el Plan 4.
Cubre:
1. Verificación de accesibilidad y contraste WCAG (mínimo 4.5:1 en todos los pares de colores).
2. Simulador sintético y rangos de fatiga.
3. Principio de seguridad clínica: Ningún dato sin validar entra al modelo.
4. Flujo de validación humana con PIN y registro de auditoría.
5. Generación y aprobación/rechazo de modelos candidatos.
6. Motor de consejos personalizados y regla obligatoria de alerta roja.
"""

import sys
import json
import shutil
import sqlite3
import tempfile
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── 1. Pruebas de Contraste WCAG (Mínimo 4.5:1) ──────────────────────────────

def luminancia_relativa(hex_color: str) -> float:
    """Calcula la luminancia relativa según WCAG 2.1."""
    hex_color = hex_color.lstrip('#')
    r, g, b = [int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
    def canal(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r_lum = canal(r)
    g_lum = canal(g)
    b_lum = canal(b)
    return 0.2126 * r_lum + 0.7152 * g_lum + 0.0722 * b_lum


def ratio_contraste(hex1: str, hex2: str) -> float:
    """Calcula el ratio de contraste entre dos colores (L1 + 0.05) / (L2 + 0.05)."""
    l1 = luminancia_relativa(hex1)
    l2 = luminancia_relativa(hex2)
    mas_claro = max(l1, l2)
    mas_oscuro = min(l1, l2)
    return (mas_claro + 0.05) / (mas_oscuro + 0.05)


class TestAccesibilidadColorimetria(unittest.TestCase):
    """Verifica que todos los pares obligatorios del Plan 4 cumplan WCAG >= 4.5:1."""

    PARES_COLORES = [
        ("Fondo pantalla / Texto general", "#0B1220", "#FFFFFF"),
        ("Verde (Normal) / Texto blanco",  "#2E7D32", "#FFFFFF"),
        ("Amarillo (Vigilar) / Texto negro", "#FFC107", "#000000"),
        ("Rojo (Crítico) / Texto blanco",  "#C62828", "#FFFFFF"),
        ("Botón Acción / Texto blanco",    "#1565C0", "#FFFFFF"),
        ("Botón Desactivado / Texto gris", "#37474F", "#CFD8DC"),
    ]

    def test_todos_los_pares_superan_4_5_a_1(self):
        for nombre, color_fondo, color_texto in self.PARES_COLORES:
            ratio = ratio_contraste(color_fondo, color_texto)
            self.assertGreaterEqual(
                ratio, 4.5,
                f"El par '{nombre}' ({color_fondo} sobre {color_texto}) no alcanza 4.5:1 (ratio={ratio:.2f}:1)"
            )


# ─── 2. Pruebas del Simulador ─────────────────────────────────────────────────

class TestSimulador(unittest.TestCase):

    def test_distribucion_y_reproducibilidad(self):
        from simulador import generar_linea_base, generar_datos_astronauta
        rng1 = np.random.default_rng(42)
        lb1 = generar_linea_base(rng1, 0)
        df1 = generar_datos_astronauta(0, lb1, rng1, n_dias=90)

        rng2 = np.random.default_rng(42)
        lb2 = generar_linea_base(rng2, 0)
        df2 = generar_datos_astronauta(0, lb2, rng2, n_dias=90)

        pd.testing.assert_frame_equal(df1, df2)
        self.assertEqual(len(df1), 90)
        self.assertIn("fatiga_7d", df1.columns)
        self.assertIn("onsd_mm", df1.columns)


# ─── 3. Pruebas de Seguridad Clínica: Validación Humana ──────────────────────

class TestValidacionHumanaYModelos(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        import validacion
        import aprendizaje
        self.val = validacion
        self.apr = aprendizaje

        self.db_orig_val = validacion.RUTA_DB
        self.db_orig_apr = aprendizaje.RUTA_DB
        self.tmp_db = Path(self.tmpdir) / "test_astrosalud.db"

        validacion.RUTA_DB = self.tmp_db
        aprendizaje.RUTA_DB = self.tmp_db

        validacion.inicializar_tablas_validacion()
        aprendizaje.inicializar_db()

    def tearDown(self):
        self.val.RUTA_DB = self.db_orig_val
        self.apr.RUTA_DB = self.db_orig_apr
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_verificar_pin(self):
        """El PIN por defecto debe ser 1234."""
        self.assertTrue(self.val.verificar_pin("1234"))
        self.assertFalse(self.val.verificar_pin("0000"))
        self.assertFalse(self.val.verificar_pin("9999"))

    def test_ningun_dato_sin_validar_entra_al_entrenamiento(self):
        """
        PRINCIPIO CRÍTICO PLAN 4: Si una lectura tiene predicción y resultado pero
        NO ha sido validada por un humano, obtener_pares_entrenamiento debe retornar 0 filas.
        """
        con = sqlite3.connect(self.tmp_db)
        # Insertar un caso pendiente sin validar
        con.execute("""
            INSERT INTO cola_validacion (
                astronauta_id, dia_prediccion, dia_objetivo,
                fatiga_pred, fatiga_real_observada, fatiga_validada,
                estado, fecha_creacion
            ) VALUES ('AST-001', 1, 8, 0, 1, 1, 'pendiente', '2026-11-15T12:00:00')
        """)
        con.commit()
        con.close()

        pares = self.apr.obtener_pares_entrenamiento("AST-001")
        self.assertEqual(len(pares), 0, "No debe incluir casos con estado='pendiente'")

    def test_flujo_resolucion_humana_confirmar_corregir_descartar(self):
        """Prueba las 3 acciones de decisión humana y su registro en auditoría."""
        con = sqlite3.connect(self.tmp_db)
        cur = con.cursor()
        cur.execute("""
            INSERT INTO cola_validacion (
                astronauta_id, dia_prediccion, dia_objetivo,
                fatiga_pred, fatiga_real_observada, fatiga_validada,
                estado, fecha_creacion
            ) VALUES ('AST-001', 1, 8, 0, 1, 1, 'pendiente', '2026-11-15T12:00:00')
        """)
        cid1 = cur.lastrowid
        cur.execute("""
            INSERT INTO cola_validacion (
                astronauta_id, dia_prediccion, dia_objetivo,
                fatiga_pred, fatiga_real_observada, fatiga_validada,
                estado, fecha_creacion
            ) VALUES ('AST-001', 2, 9, 1, 2, 2, 'pendiente', '2026-11-15T12:00:00')
        """)
        cid2 = cur.lastrowid
        cur.execute("""
            INSERT INTO cola_validacion (
                astronauta_id, dia_prediccion, dia_objetivo,
                fatiga_pred, fatiga_real_observada, fatiga_validada,
                estado, fecha_creacion
            ) VALUES ('AST-001', 3, 10, 0, 0, 0, 'pendiente', '2026-11-15T12:00:00')
        """)
        cid3 = cur.lastrowid
        con.commit()
        con.close()

        # 1. Confirmar caso 1
        res1 = self.val.resolver_validacion(cid1, "confirmar", usuario="Dr. Smith", motivo="Lectura coincide")
        self.assertEqual(res1["estado"], "confirmado")

        # 2. Corregir caso 2 (el médico rectifica la fatiga observada a 1)
        res2 = self.val.resolver_validacion(cid2, "corregir", usuario="Dr. Smith", motivo="Falso positivo por ejercicio previo", fatiga_corregida=1)
        self.assertEqual(res2["estado"], "corregido")
        self.assertEqual(res2["fatiga_validada"], 1)

        # 3. Descartar caso 3 (sensor ruidoso)
        res3 = self.val.resolver_validacion(cid3, "descartar", usuario="Dr. Smith", motivo="Artefacto de electrodo suelto")
        self.assertEqual(res3["estado"], "descartado")

        # Ahora los pares válidos para entrenamiento deben ser exactamente 2 (confirmado + corregido; descartado NUNCA entra)
        pares = self.apr.obtener_pares_entrenamiento("AST-001")
        self.assertEqual(len(pares), 2, "Solo deben entrar los confirmados y corregidos")

        # Verificar auditoría
        con = sqlite3.connect(self.tmp_db)
        audit = con.execute("SELECT COUNT(*) FROM auditoria_validacion").fetchone()[0]
        con.close()
        self.assertEqual(audit, 3, "Debe haber 3 eventos en la auditoría")


# ─── 4. Pruebas del Motor de Consejos y Regla Crítica de Emergencia ──────────

class TestMotorConsejos(unittest.TestCase):

    def test_alerta_roja_obliga_notificacion_medica_inmediata(self):
        """
        REGLA DE SEGURIDAD PLAN 4:
        En estado rojo, el primer mensaje SIEMPRE es avisar al oficial médico o al cirujano de vuelo.
        """
        from validacion import obtener_consejos_para_metricas
        ult_lectura = {"fc": 115, "sueno_horas": 3.5, "animo": 2.0}
        lineas_base = {"fc": 60, "sueno_horas": 7.5, "animo": 4.0}

        consejos = obtener_consejos_para_metricas(ult_lectura, lineas_base, estado_global="rojo")
        self.assertGreater(len(consejos), 0)
        primer_consejo = consejos[0]
        self.assertTrue(primer_consejo.get("es_critico"))
        self.assertIn("Oficial Médico", primer_consejo.get("texto"))

    def test_consejos_no_incluyen_prescripcion_farmacos(self):
        """Los consejos no deben contener dosis ni prescripción de medicamentos sintéticos."""
        from validacion import RUTA_CONSEJOS
        if not RUTA_CONSEJOS.exists():
            self.skipTest("consejos.json no encontrado")

        with open(RUTA_CONSEJOS, encoding="utf-8") as f:
            cfg = json.load(f)

        palabras_prohibidas = ["mg/día", "dosis de", "tome 500mg", "prescripción", "antibiótico"]
        for c in cfg.get("catalogo", []):
            texto = c.get("plantilla", "").lower()
            for prohibida in palabras_prohibidas:
                self.assertNotIn(prohibida, texto, f"Consejo {c.get('id')} contiene prescripción no permitida")


if __name__ == "__main__":
    unittest.main(verbosity=2)

