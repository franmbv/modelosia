import os
import sys
import warnings

# 1. Configuración de terminal limpia y soporte de emojis en Windows
sys.stdout.reconfigure(encoding="utf-8")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

from huggingface_hub import utils as hf_hub_utils
from transformers import pipeline, logging
from transformers.utils import logging as hf_logging

hf_hub_utils.logging.set_verbosity_error()
logging.set_verbosity_error()
hf_logging.disable_progress_bar()

# 2. Cargar modelo de emociones nativo en español (~435 MB)
clasificador = pipeline("text-classification", model="pysentimiento/robertuito-emotion-analysis")

# Diccionario para formatear las emociones en español con emojis
traduccion = {
    "joy": "ALEGRÍA 🥳",
    "anger": "FURIA 🤬",
    "fear": "MIEDO / PÁNICO 😱",
    "sadness": "TRISTEZA 😭",
    "surprise": "SORPRESA 😲",
    "disgust": "ASCO / RECHAZO 🤢",
    "others": "NEUTRAL 😐",
}

# 3. Frases de prueba de situaciones reales de programación
frases = [
    # 1. ANGER (Furia en comentarios de Instagram / Reclamo público)
    "Dejen de subir tiktoks y devuélvanme mi dinero 🤬",

    # 2. JOY (Comentario viral de cliente feliz / Reseña 5 estrellas)
    "¡Me encantó! Llegó en 24 horas y huele increíble 😍",

    # 3. DISGUST (Rechazo total / Reseña destructiva en video)
    "Hice un unboxing y venía lleno de grasa y con una textura asquerosa, qué falta de respeto",

    # 4. SURPRISE (Reacción a un 'haul' o compra barata)
    "¿Cómo es posible que esto cueste solo 5 dólares y funcione mejor que el de marca? Quedé tiesa",

    # 5. FEAR (Alerta de fraude / Pánico en la pasarela de pagos)
    "La página se congeló en el checkout, me debitaron del banco y no me llegó correo de confirmación" ,

    # 6. SADNESS (Decepción clásica de expectativa vs. realidad)
    "El pantalón del video se veía hermoso pero en persona parece un trapo viejo, qué tristeza "
]

print("\n🎭 --- IA DETECTORA DE EMOCIONES EN ESPAÑOL --- 🎭\n")
for frase in frases:
    resultado = clasificador(frase)[0]
    emocion = traduccion.get(resultado["label"], resultado["label"].upper())
    print(f'💬 "{frase}"')
    print(f"👉 Emoción: {emocion} ({resultado['score']:.1%})\n")
