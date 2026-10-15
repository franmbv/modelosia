import os
import sys
import warnings

# 1. Configuración de terminal limpia y soporte UTF-8 en Windows
sys.stdout.reconfigure(encoding="utf-8")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image
from huggingface_hub import utils as hf_hub_utils
from transformers import pipeline, logging
from transformers.utils import logging as hf_logging

hf_hub_utils.logging.set_verbosity_error()
logging.set_verbosity_error()
hf_logging.disable_progress_bar()

# 2. Cargar modelo de profundidad (~99 MB)
estimador = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

# 3. Imagen de entrada: argumento de consola, o 'foto.jpg', o URL de muestra
if len(sys.argv) > 1:
    imagen_entrada = sys.argv[1]
elif os.path.exists("foto.jpg"):
    imagen_entrada = "foto.jpg"
else:
    imagen_entrada = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"

print("\n📷 --- ESTIMACIÓN DE PROFUNDIDAD 3D (DEPTH ANYTHING V2) --- 📷\n")
print(f"🖼️  Procesando: {imagen_entrada}")

resultado = estimador(imagen_entrada)
mapa_gris = resultado["depth"]

# 4. Guardar mapa en escala de grises (formato clásico)
mapa_gris.save("profundidad.png")

# 5. Generar y guardar mapa térmico a color (estilo inferno / infrarrojo)
arr = np.array(mapa_gris)
puntos = [0, 64, 128, 192, 255]
colores = [
    [10, 5, 45],     # Azul/morado oscuro (muy lejos)
    [110, 20, 130],  # Violeta / magenta
    [220, 55, 35],   # Rojo / naranja
    [255, 175, 15],  # Amarillo cálido
    [255, 255, 215]  # Amarillo brillante / blanco (muy cerca)
]

lut = np.zeros((256, 3), dtype=np.uint8)
for c in range(3):
    lut[:, c] = np.interp(np.arange(256), puntos, [col[c] for col in colores])

mapa_color = Image.fromarray(lut[arr])
mapa_color.save("profundidad_color.png")

print("✨ ¡Listo!")
print("   📁 Mapa en grises:      profundidad.png")
print("   🌈 Mapa térmico color:  profundidad_color.png")
print("💡 Tip: Los tonos brillantes/cálidos están cerca; los oscuros/morados, al fondo.\n")
