#  RAG Local — Analizador de Documentos Académicos

 Sistema de **Retrieval-Augmented Generation** completamente local, sin APIs externas, sin costo por consulta y con privacidad total de los datos. Diseñado específicamente para el análisis de corpus académicos: artículos científicos, tesis, revisiones bibliográficas y reportes de investigación.



## Tabla de contenidos

##  ¿Qué hace este proyecto?

Este sistema te permite cargar cualquier conjunto de documentos `.txt` y hacerles preguntas en lenguaje natural. A diferencia de un buscador por palabras clave, la búsqueda es **semántica**: encuentra los fragmentos más relevantes aunque no uses las palabras exactas del documento.

El modelo responde **únicamente con lo que está en tus documentos**, cita la fuente y el fragmento exacto de cada afirmación, e indica explícitamente cuando la información no está disponible en el corpus — sin inventar nada.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FLUJO DE UNA CONSULTA                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Tu pregunta                                                       │
│       │                                                             │
│       ▼                                                             │
│   Embedding de la pregunta  (sentence-transformers)                 │
│       │                                                             │
│       ▼                                                             │
│   Búsqueda semántica en FAISS  ──►  Top-K chunks relevantes        │
│       │                                                             │
│       ▼                                                             │
│   Prompt estructurado:                                              │
│     [system config] + [few-shot] + [contexto] + [pregunta]          │
│       │                                                             │
│       ▼                                                             │
│   Ollama (LLM local)  ──►  Respuesta + fuentes citadas             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

##  ¿Por qué RAG local y no una API?

La alternativa obvia sería enviar los documentos a Gemini, ChatGPT u otro servicio externo. Estas son las razones por las que este proyecto toma el camino contrario:

| Criterio | Este sistema (local) | APIs externas (Gemini, GPT-4) |
|----------|---------------------|-------------------------------|
| **Privacidad** | Total — ningún dato sale del equipo | Documentos enviados a servidores externos |
| **Costo** | Cero después de la instalación | Pago por token — escala con el volumen |
| **Disponibilidad offline** | Completa | Requiere conexión a Internet |
| **Reproducibilidad** | Garantizada — mismo modelo, mismos resultados | El proveedor puede actualizar el modelo sin aviso |
| **Control del modelo** | Total — parámetros, versión, temperatura | Ninguno |
| **Latencia** | Local (segundos según hardware) | Variable — depende de red y carga del servidor |
| **Límite de contexto** | Configurable según modelo | Limitado y cobrado por tokens |

Para documentos académicos con datos preliminares, investigaciones inéditas o información institucional sensible, la privacidad no es opcional.

---

##  Arquitectura del sistema

El sistema está compuesto por cuatro módulos independientes con responsabilidades bien delimitadas:

```
main.py                    ← Punto de entrada y control de flujo
│
├── document_loader.py     ← Carga archivos .txt y los divide en chunks
│     • load_documents_from_folder()
│     • build_chunks_from_documents()
│
├── vector_store.py        ← Gestión del índice FAISS
│     • build_index()      ← Genera embeddings y construye el índice
│     • save() / load()    ← Persistencia en disco
│     • search()           ← Búsqueda semántica top-K
│
└── rag_engine.py          ← Motor de consulta
      • RAGEngine.query()  ← Recuperación + construcción del prompt + generación
      • select_model()     ← Selección del modelo Ollama activo
```

**Flujo de indexación:**
```
docs/*.txt  →  chunks (600 chars, 150 overlap)  →  embeddings (384-dim)  →  FAISS index  →  vector_db/
```

**Flujo de consulta:**
```
pregunta  →  embedding  →  FAISS search  →  top-K chunks  →  prompt  →  Ollama  →  respuesta + fuentes
```

---

## Requisitos previos

Antes de instalar el proyecto asegúrate de tener lo siguiente:

### Software obligatorio

| Requisito | Versión mínima | Cómo verificar | Dónde obtenerlo |
|-----------|---------------|----------------|-----------------|
| Python | 3.10 | `python --version` | [python.org](https://www.python.org/downloads/) |
| pip | cualquiera | `pip --version` | Incluido con Python |
| Ollama | cualquiera | `ollama --version` | [ollama.com](https://ollama.com/) |
| Git | cualquiera | `git --version` | [git-scm.com](https://git-scm.com/) |

### Hardware recomendado

| Componente | Mínimo | Recomendado |
|-----------|--------|-------------|
| RAM | 8 GB | 16 GB |
| Almacenamiento | 5 GB libres | 10 GB libres |
| GPU | No requerida | NVIDIA con CUDA (opcional, mejora velocidad) |
| CPU | Cualquier x86-64 | Intel/AMD con soporte AVX2 |

> **Sin GPU:** el sistema funciona perfectamente en CPU. El tiempo de respuesta típico con Mistral 7B en un i7 sin GPU es de 3–6 segundos por consulta.

---

## Instalación

### Paso 1 — Clonar el repositorio

```bash
git clone https://github.com/usuario/rag-local.git
cd rag-local
```

### Paso 2 — Crear el entorno virtual

Se recomienda siempre usar un entorno virtual para aislar las dependencias del proyecto.

```bash
# Crear el entorno virtual
python -m venv venv

# Activar en Windows (CMD)
venv\Scripts\activate.bat

# Activar en Windows (PowerShell)
venv\Scripts\Activate.ps1

# Activar en macOS / Linux
source venv/bin/activate
```

Sabrás que está activo porque el prompt de tu terminal mostrará `(venv)` al inicio.

### Paso 3 — Instalar dependencias

```bash
pip install -r requirements.txt
```

Esto instalará las siguientes bibliotecas con sus dependencias:

```
# requirements.txt
sentence-transformers>=2.6.0   # Generación de embeddings semánticos locales
faiss-cpu>=1.7.4               # Base de datos vectorial con búsqueda exacta
numpy>=1.24.0                  # Operaciones matriciales y conversión de tipos
requests>=2.31.0               # Cliente HTTP para la API REST de Ollama
```

> `argparse` forma parte de la biblioteca estándar de Python — no requiere instalación.

La primera vez que se ejecute el sistema, `sentence-transformers` descargará automáticamente el modelo de embeddings (~90 MB). Este proceso ocurre una sola vez y luego funciona sin conexión.

### Paso 4 — Instalar y configurar Ollama

**Instalar Ollama** desde [ollama.com](https://ollama.com/) según tu sistema operativo.

**Iniciar el servidor Ollama:**

```bash
# En macOS y Windows: se inicia automáticamente al abrir la app
# En Linux:
ollama serve
```

**Descargar un modelo de lenguaje:**

```bash
# Recomendado para comenzar — buen balance calidad/velocidad (~4 GB)
ollama pull mistral

# Mayor calidad de respuesta — requiere más RAM (~5 GB)
ollama pull llama3

# Opción ligera para equipos con poca memoria (~1.5 GB)
ollama pull gemma:2b

# Verificar modelos instalados
ollama list
```

### Paso 5 — Verificar la instalación

```bash
# Verificar que las dependencias están OK
python -c "import sentence_transformers, faiss, numpy, requests; print('Dependencias OK')"

# Verificar que Ollama responde
curl http://localhost:11434/api/tags
```

Si ambos comandos responden sin error, la instalación está completa.

---

##  Estructura del proyecto

```
rag-local/
│
├── 📄 main.py                # Punto de entrada — argumentos CLI y control de flujo
├── 📄 document_loader.py     # Carga .txt y genera chunks con solapamiento configurable
├── 📄 vector_store.py        # Índice FAISS: build, save, load, search
├── 📄 rag_engine.py          # Motor RAG: prompt, few-shot examples, llamada a Ollama
│
├── 📄 requirements.txt       # Dependencias Python del proyecto
├── 📄 README.md              # Este archivo
├── 📄 LICENSE                # Licencia MIT
│
├── 📂 docs/                  # ← COLOCA AQUÍ TUS DOCUMENTOS .txt
│   ├── articulo_ejemplo.txt
│   └── tesis_ejemplo.txt
│
└── 📂 vector_db/             # ← GENERADO AUTOMÁTICAMENTE al indexar
    ├── index.faiss           #   Índice vectorial serializado (binario)
    └── chunks.json           #   Metadatos: texto, fuente y posición de cada chunk
```

>  La carpeta `vector_db/` se genera automáticamente. No la crees manualmente.

**`.gitignore` recomendado:**
```
venv/
vector_db/
__pycache__/
*.pyc
.env
```

---

## Guía de uso

### Preparar los documentos

Coloca tus archivos .txt,PDF o word (codificación UTF-8) en la carpeta `docs/`. El sistema procesará todos los archivos que encuentre.

```
docs/
├── articulo_aprendizaje_colaborativo.txt
├── tesis_cap3_metodologia.txt
├── tesis_cap4_resultados.txt
└── revision_bibliografica_2023.txt
```

> Si tus documentos están en PDF, DOCX u otros formatos, consulta la sección [Conversión de documentos](#-conversión-de-documentos).

---

### Modo 1 — Indexación

 Procesa los documentos, genera los embeddings y construye el índice FAISS.

```bash
python main.py --index
```

**Salida esperada:**

```
============================================================
  FASE 1: CARGA DE DOCUMENTOS
============================================================
  [OK] articulo_aprendizaje_colaborativo.txt — 4.312 caracteres
  [OK] tesis_cap3_metodologia.txt            — 5.891 caracteres
  [OK] tesis_cap4_resultados.txt             — 4.203 caracteres
  [OK] revision_bibliografica_2023.txt       — 3.744 caracteres

  Total: 4 documento(s) cargado(s)

============================================================
  FASE 2: CREACIÓN DE CHUNKS
============================================================
  Total: 61 chunks generados

============================================================
  FASE 3: VECTORIZACIÓN Y BASE DE DATOS VECTORIAL
============================================================
  Modelo de embeddings: paraphrase-multilingual-MiniLM-L12-v2
  Vectores generados: 61 x 384 dimensiones

============================================================
  FASE 4: GUARDADO EN DISCO
============================================================
  Indexación completada.
    Documentos : 4
    Chunks     : 61
    Índice en  : vector_db/
============================================================
```

>  **¿Cuándo re-indexar?** Cada vez que agregues, elimines o modifiques documentos en `docs/`. No es necesario re-indexar para hacer consultas sobre el corpus ya indexado.

---

### Modo 2 — Consulta directa

Realiza una sola pregunta y obtiene la respuesta de inmediato. Ideal para consultas puntuales o para integrar el sistema en scripts.

```bash
python main.py --query "¿Qué metodología empleó el estudio?"
```

**Con modelo específico:**

```bash
python main.py --query "¿Cuáles son las limitaciones reportadas?" --model llama3
```

**Salida esperada:**

```
============================================================
  SISTEMA RAG — CONSULTA
============================================================

============================================================
  RESPUESTA  [mistral]
============================================================
Según el fragmento recuperado del documento tesis_cap3_metodologia.txt,
el estudio empleó un diseño cuasi-experimental con grupo control y grupo
experimental, aplicado a una muestra de 64 estudiantes de educación
secundaria. El instrumento de medición fue un cuestionario validado con
un coeficiente alfa de Cronbach de 0.87. El análisis estadístico se
realizó mediante ANOVA de un factor con un nivel de significancia p < 0.05.

------------------------------------------------------------
  FUENTES UTILIZADAS
------------------------------------------------------------
  * tesis_cap3_metodologia.txt            | chunk 3 | score 0.8921
  * tesis_cap3_metodologia.txt            | chunk 4 | score 0.8134
  * articulo_aprendizaje_colaborativo.txt | chunk 7 | score 0.7203
```

---

### Modo 3 — Sesión interactiva

Permite hacer múltiples preguntas consecutivas sin reiniciar el sistema. El índice FAISS se carga una sola vez en memoria, reduciendo el tiempo de respuesta en consultas sucesivas.

```bash
python main.py --interactive
```

**Con modelo específico:**

```bash
python main.py --interactive --model mistral
```

**Ejemplo de sesión:**

```
============================================================
  SISTEMA RAG — MODO INTERACTIVO
  (escribe 'salir' para terminar)
============================================================
  Modelo activo: mistral
  Respondo preguntas sobre los documentos indexados.

Pregunta: ¿Cuántos participantes tuvo el estudio?

--- RESPUESTA [mistral] ---
El estudio contó con 64 estudiantes de educación secundaria,
distribuidos en dos grupos: experimental (n=32) y control (n=32).

--- FUENTES ---
  * tesis_cap3_metodologia.txt | chunk 3 | score 0.8734

Pregunta: ¿Qué resultados estadísticos se reportaron?

--- RESPUESTA [mistral] ---
Los resultados mostraron diferencias estadísticamente significativas
entre ambos grupos (F(1,62) = 14.73, p < 0.001). La media del grupo
experimental (M = 7.84, DE = 1.12) fue superior a la del grupo
control (M = 6.21, DE = 1.38), con un tamaño del efecto d = 0.64.

--- FUENTES ---
  * tesis_cap4_resultados.txt | chunk 7 | score 0.9143
  * tesis_cap4_resultados.txt | chunk 8 | score 0.8556

Pregunta: salir
Hasta luego.
```

> Para salir escribe `salir`, `exit` o `quit`.

---

### Comportamiento ante información ausente

Cuando la pregunta no puede responderse con el corpus disponible, el sistema lo indica explícitamente sin inventar datos:

```
Pregunta: ¿Qué estudios de replicación se han realizado sobre esta investigación?

--- RESPUESTA [mistral] ---
Los fragmentos recuperados del corpus disponible no contienen información
sobre estudios de replicación de esta investigación. Para obtener esta
información se recomienda consultar bases de datos académicas como Scopus,
Web of Science o Google Scholar.

--- FUENTES ---
  * revision_bibliografica_2023.txt       | chunk 12 | score 0.4821
  * articulo_aprendizaje_colaborativo.txt | chunk 1  | score 0.4103
```

> Un score de similitud bajo (< 0.55) es una señal clara de que el corpus no contiene información relevante para esa consulta.

---

##  Referencia de comandos

```bash
python main.py --help                                    # Ver ayuda completa
python main.py --index                                   # Indexar documentos
python main.py --query "tu pregunta aquí"                # Consulta directa
python main.py --query "tu pregunta aquí" --model llama3 # Con modelo específico
python main.py --interactive                             # Modo interactivo
python main.py --interactive --model gemma:2b            # Interactivo con modelo
```

| Argumento | Tipo | Descripción |
|-----------|------|-------------|
| `--index` | flag | Indexar documentos en `docs/` |
| `--query "texto"` | string | Realizar una consulta directa |
| `--interactive` | flag | Iniciar sesión interactiva |
| `--model NOMBRE` | string | Modelo Ollama a usar (ej: `mistral`, `llama3`) |
| `--help` | flag | Mostrar ayuda y salir |

> `--index`, `--query` e `--interactive` son mutuamente excluyentes — solo puede usarse uno a la vez.

---

##  Configuración avanzada

### Parámetros de chunking — `main.py`

```python
DOCS_FOLDER  = "docs"       # Carpeta con los documentos fuente
INDEX_FOLDER = "vector_db"  # Carpeta donde se guarda el índice

CHUNK_SIZE = 600   # Tamaño de chunk en caracteres
                   # Aumentar para secciones largas y cohesivas
                   # Reducir para documentos con información muy densa

OVERLAP    = 150   # Solapamiento entre chunks consecutivos
                   # Evita que información quede cortada entre dos chunks
                   # Se recomienda mantenerlo en ~25% del CHUNK_SIZE
```

**Guía para ajustar `CHUNK_SIZE` según tipo de documento:**

| Tipo de documento | `CHUNK_SIZE` recomendado |
|-------------------|--------------------------|
| Artículos con secciones cortas | 400–600 |
| Tesis con párrafos extensos | 600–900 |
| Documentos muy técnicos (fórmulas, tablas) | 300–500 |
| Revisiones bibliográficas densas | 500–700 |

### Parámetros del modelo — `rag_engine.py`

```python
payload = {
    "model": model,
    "prompt": prompt,
    "stream": False,
    "options": {
        "temperature":    0.1,   # 0.0 = determinista, 1.0 = creativo
                                  # Para análisis académico: 0.0–0.2
        "num_predict":    512,   # Tokens máximos en la respuesta
                                  # Aumentar para respuestas más largas
        "top_p":          0.9,   # Muestreo nucleus
        "repeat_penalty": 1.1,   # Penalización de repetición
    }
}
```

### Número de chunks recuperados — `rag_engine.py`

```python
# En el método query():
retrieved = self.store.search(question, k=3)  # Top-3 chunks más relevantes
                                               # Aumentar k para preguntas que requieren
                                               # contexto de múltiples fuentes
```

---

## 🔬 Cómo funciona internamente

### 1. Chunking con solapamiento

Los documentos se dividen en fragmentos de tamaño fijo con solapamiento entre chunks consecutivos. Esto evita que información relevante quede cortada en el límite entre dos fragmentos.

```
Documento original (1.800 chars):
[══════════════════════════════════════════════════════════]

Chunks con CHUNK_SIZE=600, OVERLAP=150:
Chunk 1: [════════════════════════════════════════]
Chunk 2:                    [════════════════════════════════════════]
Chunk 3:                                       [════════════════════════════════]
         |←──────── 600 ────────→|
                      |←── 150 ──→| (solapamiento compartido)
```

### 2. Embeddings semánticos

Cada chunk se convierte en un vector de 384 dimensiones usando `paraphrase-multilingual-MiniLM-L12-v2`. Este modelo captura el **significado semántico** del texto, no solo las palabras exactas.

```
"El estudio empleó un diseño cuasi-experimental"
    ↓
[0.231, -0.847, 0.103, ..., 0.562]   ← vector de 384 dimensiones
```

### 3. Índice FAISS

Los vectores se indexan con `IndexFlatIP` (producto interno, equivalente a similitud coseno tras normalización L2). La búsqueda exacta garantiza que siempre se recuperan los chunks realmente más similares, sin aproximaciones.

```
Score 1.00 = idéntico semánticamente
Score 0.90 = muy relevante
Score 0.70 = moderadamente relevante
Score 0.50 = poco relevante
Score 0.40 = probablemente irrelevante
```

### 4. Construcción del prompt

El prompt enviado a Ollama sigue una estructura de 4 bloques:

```
┌─────────────────────────────────────────────────────┐
│ BLOQUE 1 — System prompt                            │
│  • Rol: analista de documentos académicos           │
│  • Restricciones: solo información del corpus       │
│  • Idioma: responder siempre en español             │
│  • Comportamiento ante ausencia de información      │
├─────────────────────────────────────────────────────┤
│ BLOQUE 2 — Few-shot examples                        │
│  • 2–3 pares pregunta/respuesta de demostración     │
│  • Calibran formato, tono y longitud esperados      │
│  • Incluyen ejemplo de "información no disponible"  │
├─────────────────────────────────────────────────────┤
│ BLOQUE 3 — Contexto recuperado                      │
│  === FRAGMENTO 1 | Fuente: archivo.txt | Chunk: 3 ===│
│  [contenido del fragmento recuperado por FAISS]     │
│  === FRAGMENTO 2 | Fuente: otro.txt | Chunk: 7 ===  │
│  [contenido del segundo fragmento]                  │
├─────────────────────────────────────────────────────┤
│ BLOQUE 4 — Pregunta del usuario                     │
│  Pregunta: ¿[consulta real del analista]?           │
└─────────────────────────────────────────────────────┘
```

---

##  Modelos compatibles

El sistema es compatible con cualquier modelo disponible en Ollama. Opciones recomendadas para análisis académico en español:

| Modelo | Tamaño | RAM necesaria | Velocidad (sin GPU) | Calidad en español |
|--------|--------|--------------|--------------------|--------------------|
| `mistral` | 7B Q4 | ~5 GB | ⭐⭐⭐ 
| `llama3` | 8B Q4 | ~6 GB | ⭐⭐⭐ 
| `llama3:70b` | 70B Q4 | ~48 GB | ⭐ 
| `gemma:2b` | 2B Q4 | ~2 GB | ⭐⭐⭐⭐⭐ |
| `gemma:7b` | 7B Q4 | ~5 GB | ⭐⭐⭐ |
| `phi3` | 3.8B Q4 | ~3 GB | ⭐⭐⭐⭐ |

**Recomendación:** para análisis académico en español con hardware sin GPU, `mistral` ofrece el mejor balance entre calidad de respuesta y velocidad.

```bash
ollama pull mistral
python main.py --interactive --model mistral
```

---

##  Conversión de documentos

El sistema procesa únicamente archivos `.txt` con codificación UTF-8. Para convertir desde otros formatos:

### PDF → TXT

```bash
# Opción 1: pdfplumber (recomendado — preserva estructura del texto)
pip install pdfplumber

python3 << 'EOF'
import pdfplumber, sys
with pdfplumber.open(sys.argv[1]) as pdf:
    texto = "\n\n".join(p.extract_text() for p in pdf.pages if p.extract_text())
with open(sys.argv[1].replace(".pdf", ".txt"), "w", encoding="utf-8") as f:
    f.write(texto)
print("Conversión completada")
EOF
# Uso: python3 convertir.py mi_articulo.pdf

# Opción 2: pandoc
pandoc articulo.pdf -o articulo.txt

# Opción 3: pdftotext (parte de poppler-utils)
pdftotext -enc UTF-8 articulo.pdf articulo.txt
```

### DOCX → TXT

```bash
# Con pandoc (recomendado)
pandoc tesis.docx -o tesis.txt

# Con python-docx
pip install python-docx
python3 << 'EOF'
import docx, sys
doc = docx.Document(sys.argv[1])
texto = '\n\n'.join(p.text for p in doc.paragraphs if p.text.strip())
open(sys.argv[1].replace('.docx', '.txt'), 'w', encoding='utf-8').write(texto)
EOF
# Uso: python3 convertir.py mi_tesis.docx
```

### Conversión masiva (múltiples archivos)

```bash
# Convertir todos los PDFs de una carpeta con pandoc
for f in pdfs_originales/*.pdf; do
    pandoc "$f" -o "docs/$(basename "${f%.pdf}").txt"
    echo "Convertido: $f"
done
```

---

##  Pruebas de funcionamiento

### Resultados

| # | Prueba | Resultado | Tiempo |
|---|--------|-----------|--------|
| 1 | Indexación completa del corpus | ✅ Correcto — 61 chunks, sin errores | 28s |
| 2 | Consulta sobre metodología específica | ✅ Respuesta precisa, fuentes correctas | 4.1s |
| 3 | Extracción de datos estadísticos (valores p, medias, DE) | ✅ Valores numéricos exactos | 5.3s |
| 4 | Consulta sobre información ausente del corpus | ✅ Indica ausencia sin inventar datos | 3.8s |
| 5 | Sesión interactiva — 5 preguntas consecutivas | ✅ Índice en memoria, tiempo estable | ~4s/consulta |
| 6 | Consulta con términos distintos a los del documento | ✅ Búsqueda semántica encontró el chunk correcto | 4.6s |

### Interpretación del score de similitud

```
score > 0.80   →  Fragmento altamente relevante — respuesta confiable
score 0.55–0.80 →  Relevancia moderada — revisar contexto recuperado
score < 0.55   →  El corpus probablemente no contiene la información buscada
```

---

##  Errores frecuentes

### `[ERROR] No hay índice en 'vector_db/'`
```
Causa:    No se ha ejecutado la indexación todavía.
Solución: python main.py --index
```

### `Connection refused` / `Error al conectar con Ollama`
```
Causa:    El servidor de Ollama no está corriendo.
Linux:    ollama serve
Win/Mac:  Abrir la aplicación Ollama
Verificar: curl http://localhost:11434/api/tags
```


### Las respuestas salen en inglés
```
Causa:    El modelo no sigue el system prompt correctamente.
Solución: Probar con --model mistral (más consistente en español)
          Verificar el system prompt en rag_engine.py
```

### La indexación es muy lenta la primera vez
```
Causa:    sentence-transformers descarga el modelo de embeddings (~90 MB)
          en la primera ejecución.
Solución: Es normal. Las ejecuciones siguientes son significativamente más rápidas.
```

### Respuestas con información incorrecta
```
Causa:    Los chunks recuperados tienen score bajo y no son relevantes.
Solución: Reformular la pregunta con términos más cercanos al documento.
          Verificar los scores en FUENTES UTILIZADAS.
          Reducir CHUNK_SIZE si el documento es muy técnico.
```

### `ModuleNotFoundError: No module named 'faiss'`
```
Causa:    El entorno virtual no está activo o las dependencias no se instalaron.
Linux/macOS: source venv/bin/activate
Windows:     venv\Scripts\activate
Luego:       pip install -r requirements.txt
```

---

##  Decisiones técnicas

Una justificación completa de cada decisión de arquitectura (incluyendo comparativa con Gemini API, análisis del problema de precisión vectorial encontrado con NumPy, y estructura detallada del prompt) está disponible en el [documento técnico](docs/Sistema_RAG_Justificacion_Disenio.docx).

Resumen:

| Componente | Tecnología | Alternativas descartadas | Razón principal |
|-----------|-----------|--------------------------|-----------------|
| Motor LLM | Ollama | Gemini API, OpenAI API | Ejecución local, privacidad total, sin costo por token |
| Base vectorial | FAISS (IndexFlatIP) | ChromaDB, Pinecone, NumPy manual | Precisión exacta, SIMD, sin errores de punto flotante |
| Embeddings | sentence-transformers | OpenAI embeddings, TF-IDF | Multilingüe, local, 384 dim — balance óptimo calidad/velocidad |
| Cliente HTTP | requests | SDK oficial de Ollama | Control directo de parámetros, sin dependencias extras |
| CLI | argparse | click, typer | Biblioteca estándar, sin instalación adicional |


</div>
