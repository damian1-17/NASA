import sys
from pathlib import Path

# Añadir raíz de proyecto al PYTHONPATH para importar servidor, aprendizaje, validación
ruta_raiz = Path(__file__).parent.parent
if str(ruta_raiz) not in sys.path:
    sys.path.insert(0, str(ruta_raiz))

from servidor import app
