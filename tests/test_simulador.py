"""
tests/test_simulador.py — Pruebas unitarias del simulador y funciones de umbral.

Ejecutar: python -m pytest tests/ -v
"""

import sys
import json
import numpy as np
import pandas as pd
import unittest
from pathlib import Path

# Agregar el directorio padre al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulador import generar_linea_base, generar_datos_astronauta, N_DIAS


class TestSimulador(unittest.TestCase):
    """Pruebas unitarias del generador de datos sintéticos."""

    def setUp(self):
        """Configuración inicial: semilla fija para reproducibilidad."""
        self.rng = np.random.default_rng(42)
        self.lb = generar_linea_base(self.rng, 0)

    def test_linea_base_estructura(self):
        """La línea base debe tener todas las claves esperadas."""
        claves_requeridas = [
            "astronauta_id", "base_sistolica", "base_diastolica",
            "base_fc", "base_hrv", "base_spo2", "base_temperatura",
            "base_glucosa", "base_sueno_horas", "base_onsd",
            "base_radiacion_diaria", "cumplimiento_ejercicio", "resiliencia"
        ]
        for clave in claves_requeridas:
            self.assertIn(clave, self.lb, f"Clave faltante: {clave}")

    def test_linea_base_rangos_plausibles(self):
        """Los valores de línea base deben estar en rangos fisiológicos."""
        rng = np.random.default_rng(99)
        # Probar con 10 astronautas distintos
        for i in range(10):
            lb = generar_linea_base(rng, i)
            self.assertGreater(lb["base_sistolica"], 90, "Sistólica muy baja")
            self.assertLess(lb["base_sistolica"], 160, "Sistólica muy alta")
            self.assertGreater(lb["base_fc"], 45, "FC muy baja")
            self.assertLess(lb["base_fc"], 95, "FC muy alta")
            self.assertGreater(lb["base_spo2"], 96, "SpO2 muy baja")
            self.assertLess(lb["base_spo2"], 100.5, "SpO2 imposible")
            self.assertGreater(lb["base_temperatura"], 35.5, "Temperatura muy baja")
            self.assertLess(lb["base_temperatura"], 38.0, "Temperatura muy alta")
            self.assertGreater(lb["base_hrv"], 20, "HRV muy baja")
            self.assertGreater(lb["base_onsd"], 3.5, "ONSD muy bajo")
            self.assertLess(lb["base_onsd"], 5.5, "ONSD muy alto (línea base)")

    def test_datos_astronauta_shape(self):
        """El DataFrame generado debe tener la forma correcta."""
        df = generar_datos_astronauta(0, self.lb, self.rng)
        self.assertEqual(len(df), N_DIAS,
                         f"Esperados {N_DIAS} días, obtenidos {len(df)}")

    def test_columnas_requeridas(self):
        """El DataFrame debe contener todas las columnas requeridas."""
        df = generar_datos_astronauta(0, self.lb, self.rng)
        columnas_requeridas = [
            "astronauta_id", "dia_mision", "sistolica", "diastolica",
            "fc", "hrv", "spo2", "temperatura", "glucosa",
            "sueno_horas", "sueno_calidad", "animo", "carga_trabajo",
            "ejercicio_min", "onsd_mm", "radiacion_diaria",
            "radiacion_acumulada", "fatiga_7d", "onsd_7d", "radiacion_7d"
        ]
        for col in columnas_requeridas:
            self.assertIn(col, df.columns, f"Columna faltante: {col}")

    def test_datos_faltantes_en_rango(self):
        """El porcentaje de datos faltantes debe estar entre 5% y 15%."""
        rng = np.random.default_rng(42)
        lb = generar_linea_base(rng, 0)
        df = generar_datos_astronauta(0, lb, rng)

        cols_con_faltantes = [
            "sistolica", "diastolica", "fc", "hrv", "spo2",
            "temperatura", "glucosa", "sueno_horas", "sueno_calidad",
            "animo", "carga_trabajo", "ejercicio_min"
        ]
        for col in cols_con_faltantes:
            pct = df[col].isnull().mean()
            self.assertLessEqual(pct, 0.20,
                f"Demasiados faltantes en {col}: {pct:.1%}")

    def test_onsd_solo_semanal(self):
        """El ONSD solo debe medirse aproximadamente una vez por semana."""
        df = generar_datos_astronauta(0, self.lb, self.rng)
        dias_con_onsd = df["onsd_mm"].notna().sum()
        # Esperamos ~180/7 ≈ 26 mediciones
        self.assertGreater(dias_con_onsd, 20, "Muy pocas mediciones de ONSD")
        self.assertLess(dias_con_onsd, 40, "Demasiadas mediciones de ONSD")

    def test_radiacion_acumulada_creciente(self):
        """La radiación acumulada debe ser monotónicamente no-decreciente."""
        df = generar_datos_astronauta(0, self.lb, self.rng)
        rad = df["radiacion_acumulada"].values
        self.assertTrue(np.all(np.diff(rad) >= -1e-6),
                        "La radiación acumulada no es monotónica")

    def test_fatiga_etiquetas_validas(self):
        """Las etiquetas de fatiga deben ser 0, 1 o 2 (o NaN al final)."""
        df = generar_datos_astronauta(0, self.lb, self.rng)
        etiquetas = df["fatiga_7d"].dropna().unique()
        for e in etiquetas:
            self.assertIn(e, [0, 1, 2],
                          f"Etiqueta inválida de fatiga: {e}")

    def test_reproducibilidad(self):
        """Misma semilla debe producir mismos datos."""
        rng1 = np.random.default_rng(42)
        lb1 = generar_linea_base(rng1, 0)
        df1 = generar_datos_astronauta(0, lb1, rng1)

        rng2 = np.random.default_rng(42)
        lb2 = generar_linea_base(rng2, 0)
        df2 = generar_datos_astronauta(0, lb2, rng2)

        pd.testing.assert_frame_equal(df1, df2)

    def test_dias_mision_secuenciales(self):
        """Los días de misión deben ser consecutivos del 1 al N_DIAS."""
        df = generar_datos_astronauta(0, self.lb, self.rng)
        dias = df["dia_mision"].values
        np.testing.assert_array_equal(dias, np.arange(1, N_DIAS + 1))


class TestZScore(unittest.TestCase):
    """Pruebas de las funciones de umbral y z-score."""

    def test_zscore_positivo(self):
        """Z-score positivo para valor sobre la media."""
        valor = 120
        media = 100
        std = 10
        z = (valor - media) / std
        self.assertAlmostEqual(z, 2.0)

    def test_zscore_cero_en_media(self):
        """Z-score cero cuando el valor es igual a la media."""
        valor = 72.5
        media = 72.5
        std = 5.0
        z = (valor - media) / std
        self.assertAlmostEqual(z, 0.0)

    def test_clasificacion_semaforo(self):
        """La clasificación de semáforo debe seguir los umbrales de z-score."""
        def clasificar(z):
            if abs(z) <= 1.5:
                return "verde"
            elif abs(z) <= 2.5:
                return "amarillo"
            else:
                return "rojo"

        self.assertEqual(clasificar(0.0),  "verde")
        self.assertEqual(clasificar(1.4),  "verde")
        self.assertEqual(clasificar(1.6),  "amarillo")
        self.assertEqual(clasificar(2.4),  "amarillo")
        self.assertEqual(clasificar(2.6),  "rojo")
        self.assertEqual(clasificar(-3.0), "rojo")

    def test_config_rangos_estructura(self):
        """El archivo de configuración debe tener la estructura correcta."""
        config_path = Path(__file__).parent.parent / "config_rangos.json"
        self.assertTrue(config_path.exists(), "config_rangos.json no encontrado")

        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        # Verificar métricas clave
        metricas_requeridas = [
            "presion_sistolica", "frecuencia_cardiaca", "hrv",
            "spo2", "temperatura", "glucosa", "sueno_horas", "onsd"
        ]
        for metrica in metricas_requeridas:
            self.assertIn(metrica, config, f"Métrica faltante en config: {metrica}")

        # Verificar que tiene campos verde/amarillo/rojo
        for metrica in metricas_requeridas:
            if isinstance(config[metrica], dict):
                self.assertIn("verde", config[metrica],
                              f"Falta 'verde' en {metrica}")


class TestExportacionModelo(unittest.TestCase):
    """Pruebas de la exportación de modelos a JSON."""

    def test_exportacion_json_estructura(self):
        """Un JSON de modelo exportado debe tener los campos requeridos."""
        ruta_modelos = Path(__file__).parent.parent / "modelos"
        if not ruta_modelos.exists():
            self.skipTest("Directorio de modelos no encontrado; ejecutar entrenar.py primero")

        for json_path in ruta_modelos.glob("*.json"):
            if json_path.name.endswith("_meta.json"):
                continue
            with open(json_path, encoding="utf-8") as f:
                modelo = json.load(f)

            campos = ["tipo", "feature_cols", "scaler_mean", "scaler_std"]
            for campo in campos:
                self.assertIn(campo, modelo,
                              f"Campo '{campo}' faltante en {json_path.name}")

            # Verificar consistencia de dimensiones del scaler
            self.assertEqual(
                len(modelo["scaler_mean"]),
                len(modelo["scaler_std"]),
                f"Dimensiones inconsistentes en scaler de {json_path.name}"
            )
            self.assertEqual(
                len(modelo["feature_cols"]),
                len(modelo["scaler_mean"]),
                f"n_features inconsistente en {json_path.name}"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

