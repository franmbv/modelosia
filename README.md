# 🤖 Modelos Curiosos de Hugging Face

Colección de implementaciones prácticas, limpias y minimalistas (< 25 líneas de código) de modelos de Inteligencia Artificial de Hugging Face.

El objetivo de este repositorio es explorar modelos accesibles, ligeros y con resultados llamativos o prácticos, listos para ejecutarse de forma local en CPU sin necesidad de configuraciones complejas ni hardware de alta gama.

---

## 📂 Catálogo de Modelos

| # | Nombre / Tarea | Modelo Hugging Face | Modalidad | Script |
|:---:|:---|:---|:---:|:---:|
| **01** | Detector de Emociones en Español | [`pysentimiento/robertuito-emotion-analysis`](https://huggingface.co/pysentimiento/robertuito-emotion-analysis) | NLP / Texto | [`dia1.py`](dia1.py) |
| **02** | Estimación de Profundidad 3D | [`depth-anything/Depth-Anything-V2-Small-hf`](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) | Visión / Imagen | [`dia2.py`](dia2.py) |

---

## 🛠️ Stack y Requisitos

- **Python:** 3.10+
- **Librerías principales:**
  - `transformers` (Hugging Face)
  - `torch` (PyTorch)
  - `huggingface_hub`
  - `pillow` / `opencv-python` *(para modelos de visión)*
  - `soundfile` / `torchaudio` *(para modelos de audio)*

---

## 🚀 Instalación y Puesta en Marcha

### 1. Clonar el repositorio
```bash
git clone https://github.com/TU_USUARIO/modelosia.git
cd modelosia
```

### 2. Crear y activar el entorno virtual

**En Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**En Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install torch transformers huggingface_hub pillow
```

### 4. Ejecutar un script
Por ejemplo, para correr el primer ejemplo:
```bash
python dia1.py
```

> **Nota:** La primera vez que ejecutas un script se descargarán los pesos del modelo automáticamente en la caché local (`~/.cache/huggingface/hub`). Las ejecuciones posteriores serán instantáneas.

---

## 🧹 Liberar Espacio en Disco (Caché)

Si pruebas varios modelos y deseas borrar los pesos descargados para recuperar espacio:
```bash
huggingface-cli delete-cache
```

---

## 📐 Criterios de Selección de Modelos

Cada implementación en este repositorio cumple con las siguientes directrices:
1. **Ligeros:** Pesan entre 50 MB y 800 MB (máximo 1.5 GB).
2. **Compatibilidad:** Ejecución fluida en CPU estándar sin requerir clústeres en la nube ni GPUs dedicadas.
3. **Código minimalista:** Scripts autónomos de menos de 25 líneas con salidas de consola claras y formateadas.
