import os
import sys
import warnings

# 1. Terminal limpia y soporte de emojis en Windows
sys.stdout.reconfigure(encoding="utf-8")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from huggingface_hub import utils as hf_hub_utils
from transformers import pipeline, logging
from transformers.utils import logging as hf_logging

hf_hub_utils.logging.set_verbosity_error()
logging.set_verbosity_error()
hf_logging.disable_progress_bar()

# 2. Cargar modelo de embeddings multilingüe (~470 MB)
extractor = pipeline("feature-extraction", model="intfloat/multilingual-e5-small", return_tensors=True)

def vectorizar(texto):
    # Promedio de tokens (mean pooling) + normalización L2
    return F.normalize(extractor(texto)[0].mean(dim=0), p=2, dim=0)

# 3. Base de conocimiento: Runbooks de DevOps
runbooks = [
    "Fallo OOMKilled en Kubernetes: aumentar limits.memory en deployment.yaml y revisar fugas de RAM.",
    "Certificado SSL/HTTPS expirado en Nginx: ejecutar 'sudo certbot renew' para restaurar la conexión segura.",
    "Error 429 Too Many Requests en la API: demasiadas peticiones por segundo, activar caché en Redis.",
    "Almacenamiento lleno en servidor Linux: liberar espacio en disco borrando logs viejos con journalctl.",
]

# Pre-indexar los runbooks como vectores (el núcleo de un RAG)
vectores_docs = torch.stack([vectorizar(f"passage: {doc}") for doc in runbooks])

# 4. Consultas en pánico (sin usar los nombres técnicos de los errores)
consultas = [
    "Chrome bloquea la página diciendo que la conexión no es privada",
    "Los contenedores se caen y reinician porque consumen demasiada memoria",
    "No puedo subir archivos porque el disco duro se quedó sin capacidad",
]

print("\n🔎 --- MINI-RAG DEVOPS: BÚSQUEDA SEMÁNTICA LOCAL --- 🔎\n")
for pregunta in consultas:
    vec_q = vectorizar(f"query: {pregunta}")
    puntajes = torch.matmul(vectores_docs, vec_q)
    mejor_idx = int(torch.argmax(puntajes))
    similitud = float(puntajes[mejor_idx])

    print(f'🆘 Alerta: "{pregunta}"')
    print(f"🛠️  Runbook ({similitud:.1%}): {runbooks[mejor_idx]}\n")
