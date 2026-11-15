# AstroSalud v4 — Monitor de Salud para Astronautas (Raspberry Pi Touch)

> **⚠️ PROTOTIPO DE INVESTIGACIÓN — DATOS SINTÉTICOS**  
> Sistema de **apoyo a la decisión clínica**, no diagnóstico médico.  
> Diseñado para el **NASA Space Apps Challenge**.

---

## 1. Novedades del Plan 4

El **Plan 4** introduce requisitos estrictos de **seguridad clínica, accesibilidad WCAG y validación humana**:

| Requisito | Plan 2 | Plan 4 |
|---|---|---|
| **Colorimetría y Contraste** | Modo oscuro genérico | Paleta fija de alto contraste con texto y fondo explícitos en todo elemento (WCAG $\ge$ 4.5:1). |
| **Gobernanza del Aprendizaje** | Actualización automática con datos nuevos | **Validación Humana Obligatoria**: el modelo NUNCA aprende de datos sin validar. |
| **Flujo de Validación Médica** | No disponible | Cola protegida con PIN para el Oficial Médico (CMO): **Confirmar**, **Corregir** o **Descartar**. |
| **Aprobación de Modelos** | Rollback automático silencioso | Generación de **versión candidata**, comparativa de métricas y aprobación humana antes de activar. |
| **Motor de Consejos** | Lista estática de contramedidas | Catálogo dinámico `consejos.json`: reglas personalizadas, regla crítica de notificación inmediata en alerta roja y feedback *"Me sirvió" / "No me sirvió"*. |
| **Diseño Gráfico de Métricas** | Grid 2×3 básico | Cuadrícula 3×2 con tarjetas sólidas de 250×200 px, ordenadas por urgencia (Rojo $\rightarrow$ Amarillo $\rightarrow$ Verde), paginables, con sparklines SVG y valores $\ge$ 44 px. |
| **Gráfica de Detalle** | Canvas básico | SVG puro de alta legibilidad, bandas semafóricas (25–30% opacidad), marcadores para daltonismo (círculo, triángulo, cuadrado), selector 7/14/30d y panel fijo de lectura. |
| **Aislamiento Offline** | Dependiente de librerías | **100 % Offline sin excepciones**: sin CDN, sin fuentes web, sin llamadas a APIs externas. |

---

## 2. Paleta de Alto Contraste (WCAG 2.1 AA $\ge$ 4.5:1)

Todos los componentes definen explícitamente sus colores de fondo y texto:

- **Fondo de pantalla:** `#0B1220` | **Texto general:** `#FFFFFF` (Ratio: **18.4:1**)
- **Verde (Normal):** Fondo `#2E7D32` | Texto `#FFFFFF` (Ratio: **5.2:1**)
- **Amarillo (Vigilancia):** Fondo `#FFC107` | Texto `#000000` (Ratio: **12.9:1**)
- **Rojo (Alerta Crítica):** Fondo `#C62828` | Texto `#FFFFFF` (Ratio: **5.6:1**)
- **Botón de Acción:** Fondo `#1565C0` | Texto `#FFFFFF` | Borde `2px solid #FFFFFF` (Ratio: **5.8:1**)
- **Botón Desactivado:** Fondo `#37474F` | Texto `#CFD8DC` (Ratio: **6.5:1**)

---

## 3. Estructura del Proyecto

```
proyecto/
├── simulador.py            # Generador sintético de misiones (180 días, 50 astronautas)
├── entrenar_base.py        # Preentrenamiento supervisado base (SGDClassifier, SGDRegressor, Ridge)
├── aprendizaje.py          # Motor de aprendizaje incremental gobernado
├── validacion.py           # Cola de validación humana con PIN, auditoría y modelos candidatos
├── servidor.py             # Backend local FastAPI + SQLite (100% offline)
├── consejos.json           # Motor de reglas y catálogo de consejos personalizados
├── config_rangos.json      # Rangos de referencia fisiológica
├── datos/
│   └── astrosalud.db       # Base de datos SQLite local
├── modelos/
│   ├── fatiga.joblib       # Modelo activo vigente (SGDClassifier con partial_fit)
│   ├── onsd.joblib         # Modelo activo ONSD (SGDRegressor)
│   ├── radiacion.joblib    # Proyección lineal radiación
│   └── versiones/          # Historial de versiones y modelos candidatos
├── app/
│   └── index.html          # Interfaz táctil 800×480 px (SVG, alto contraste, teclado numérico)
├── tests/
│   ├── test_plan4.py       # Pruebas Plan 4: contraste WCAG, seguridad clínica, PIN, consejos
│   ├── test_plan2.py       # Pruebas Plan 2: persistencia, escalado, joblib
│   └── test_simulador.py   # Pruebas del generador y líneas base
└── README.md
```

---

## 4. Flujo de Validación Humana y Gobernanza del Modelo

```
[ Astronauta / Telemetría ]
          │
          ▼
   (Día t: Predicción a 7 días)
          │
    7 días después
          │
          ▼
[ Cola de Validación Humana ] (Protegida con PIN '1234')
          │
  ┌───────┼───────────────────────────┐
  ▼       ▼                           ▼
Confirmar Corregir                 Descartar
 (Verde)  (Amarillo)                (Rojo)
  └───────┬───────────────────────────┘
          │ (Exclusivamente Confirmados y Corregidos)
          ▼
[ Generar Modelo Candidato ]
          │
   (Comparación F1 Macro y Recall Rojo contra Vigente)
          │
          ▼
[ Aprobación Médica Humana ] ────► Aprobado ──► [ Activar en Producción ]
                                └► Rechazado ──► [ Mantener Modelo Vigente ]
```

---

## 5. Instrucciones de Ejecución

### En PC / Entorno de Pruebas:

```bash
# 1. Instalar dependencias
pip install numpy pandas scikit-learn joblib fastapi uvicorn

# 2. Generar datos sintéticos
python simulador.py

# 3. Entrenar modelos base
python entrenar_base.py

# 4. Iniciar el servidor local
python servidor.py
# Abrir en el navegador: http://localhost:8000
```

### Ejecutar Pruebas Unitarias:

```bash
python -m unittest discover tests
```
*(39 pruebas pasan exitosamente: contraste WCAG, seguridad de datos sin validar, auditoría, modelos y consejos).*

---

## 6. Instalación y Puesta en Marcha en Raspberry Pi (Pantalla 7" 800×480)

### 1. Preparación del Sistema Operativo
- **Hardware recomendado:** Raspberry Pi 4 o 5 (2 GB+ RAM).
- **Pantalla:** Raspberry Pi Touch Display 7" oficial (800×480 px DSI o HDMI).
- **SO:** Raspberry Pi OS (64-bit) con entorno de escritorio.

### 2. Configurar Servicio Autostart (`systemd`)

Crear `/etc/systemd/system/astrosalud.service`:
```ini
[Unit]
Description=AstroSalud Backend Local 100% Offline
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/proyecto
ExecStart=/usr/bin/python3 servidor.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Habilitar el servicio:
```bash
sudo systemctl daemon-reload
sudo systemctl enable astrosalud.service
sudo systemctl start astrosalud.service
```

### 3. Modo Quiosco en Chromium al Arrancar

Editar `~/.config/lxsession/LXDE-pi/autostart` o `~/.config/wayfire.ini`:
```bash
@xset s off
@xset -dpms
@xset s noblank
@chromium-browser --kiosk --disable-restore-session-state --disable-infobars --noerrdialogs --check-for-update-interval=31536000 http://localhost:8000
```

---

## 7. Limitaciones y Próximos Pasos

1. **Datos Sintéticos:** Los datos fueron generados algorítmicamente para validar el flujo completo. Requiere calibración con repositorios abiertos de la NASA (GeneLab, OSDR, LSDA).
2. **Cifrado de Base de Datos:** Para misiones reales, la base de datos SQLite debe cifrarse con SQLCipher (AES-256).
3. **PIN Físico / Criptográfico:** El PIN por defecto es `1234`, modificable mediante configuración protegida por clave criptográfica de tripulación.
