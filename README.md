#  RAG Local — Analizador de Documentos Académicos

> Sistema de **Retrieval-Augmented Generation** completamente local, sin APIs externas, sin costo por consulta y con privacidad total de los datos.

---

## ¿Qué hace este proyecto?

Carga documentos `.txt,PDF,Word`, los indexa como vectores semánticos y te permite hacerles preguntas en lenguaje natural. El modelo responde **solo con lo que está en tus documentos**, citando la fuente exacta de cada afirmación.

```
Tu pregunta  →  Búsqueda semántica (FAISS)  →  Contexto relevante  →  Respuesta (Ollama)
```

**Pensado para:** análisis de artículos académicos, tesis, revisiones bibliográficas y cualquier corpus de texto que necesites consultar de forma inteligente.

---

## Características principales

-  **100% local** — ningún dato sale de tu equipo
-  **Sin costo por consulta** — sin APIs de pago
-  **Funciona offline** — una vez instalado, no necesitas internet
-  **Búsqueda semántica exacta** con FAISS (no por palabras clave)
-  **Cita las fuentes** — cada respuesta indica el documento y chunk de origen
-  **Multi-modelo** — compatible con Mistral, LLaMA 3, Gemma y cualquier modelo de Ollama
-  **Sin alucinaciones forzadas** — si la información no está en el corpus, el sistema lo indica explícitamente

---

## Requisitos previos

| Requisito | Versión mínima | Notas |
|-----------|---------------|-------|
| Python | 3.10+ | |
| [Ollama](https://ollama.com/) | cualquiera | Debe estar corriendo en background |
| Git | cualquiera | Solo para clonar el repo |
| RAM | 8 GB recomendados | Depende del modelo elegido |

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/usuario/rag-local.git
cd rag-local
```

### 2. Crear y activar el entorno virtual

```bash
# Crear el entorno virtual
python -m venv venv

# Activar — Windows
venv\Scripts\activate

# Activar — macOS / Linux
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

**Contenido de `requirements.txt`:**

```
sentence-transformers>=2.6.0   # Embeddings semánticos locales
faiss-cpu>=1.7.4               # Base de datos vectorial
numpy>=1.24.0                  # Operaciones matriciales
requests>=2.31.0               # Cliente HTTP para Ollama
```

> **Nota:** `argparse` ya viene incluido en la biblioteca estándar de Python, no requiere instalación.

### 4. Descargar un modelo en Ollama

```bash
# Recomendado para hardware sin GPU dedicada (~4 GB)
ollama pull mistral

# Alternativas
ollama pull llama3        # Mayor calidad — requiere ~5 GB de RAM
ollama pull gemma:2b      # Opción ligera para equipos con poca memoria

# Verificar modelos disponibles
ollama list
```

> ⚠️ Ollama debe estar corriendo en segundo plano antes de hacer consultas. En Windows e macOS se inicia automáticamente; en Linux: `ollama serve`

---

## Estructura del proyecto

```
rag-local/
├── main.py               # Punto de entrada — maneja los tres modos de uso
├── document_loader.py    # Carga archivos .txt y genera chunks con solapamiento
├── vector_store.py       # Gestiona el índice FAISS (build, save, load, search)
├── rag_engine.py         # Motor RAG: recuperación semántica + generación con Ollama
├── requirements.txt      # Dependencias Python
├── docs/                 # 📂 Coloca aquí tus documentos .txt
│   └── (tus archivos académicos)
└── vector_db/            # 📂 Generado automáticamente al indexar
    ├── index.faiss       # Índice vectorial serializado
    └── chunks.json       # Metadatos: texto, fuente y posición de cada chunk
```

> `vector_db/` se genera sola al correr `--index`. No la crees manualmente. Se recomienda añadirla al `.gitignore`.

---

## Uso

### Paso 1 — Coloca tus documentos

Copia tus archivos `.txt` en la carpeta `docs/`. El sistema procesará todos los que encuentre.

```
docs/
├── articulo_metodologia.txt
├── tesis_cap3_resultados.txt
└── revision_bibliografica.txt
```

> Si tus documentos están en PDF o DOCX, conviértelos primero a `.txt` con `pdfplumber`, `pandoc` o LibreOffice.

### Paso 2 — Indexar

```bash
python main.py --index
```

Esto genera los embeddings y construye el índice FAISS. **Debes repetir este paso cada vez que agregues o modifiques documentos.**

Salida esperada:

```
============================================================
  FASE 1: CARGA DE DOCUMENTOS
  Total: 3 documento(s) cargado(s)

  FASE 2: CREACIÓN DE CHUNKS
  Total: 47 chunks generados

  FASE 3: VECTORIZACIÓN Y BASE DE DATOS VECTORIAL
  FASE 4: GUARDADO EN DISCO

  Indexación completada.
    Documentos : 3
    Chunks     : 47
    Índice en  : vector_db/
============================================================
```

### Paso 3 — Hacer consultas

**Consulta directa (una sola pregunta):**

```bash
python main.py --query "¿Qué metodología empleó el estudio?"
```

**Con modelo específico:**

```bash
python main.py --query "¿Cuáles son las limitaciones del estudio?" --model llama3
```

**Modo interactivo (múltiples preguntas en sesión continua):**

```bash
python main.py --interactive

# Con modelo específico
python main.py --interactive --model mistral
```

Para salir del modo interactivo escribe `salir`, `exit` o `quit`.

---

## Referencia de comandos

| Comando | Descripción |
|---------|-------------|
| `python main.py --index` | Indexar todos los `.txt` de `docs/` |
| `python main.py --query "pregunta"` | Consulta directa y respuesta inmediata |
| `python main.py --interactive` | Sesión interactiva con múltiples preguntas |
| `--model NOMBRE` | Seleccionar modelo Ollama (ej: `mistral`, `llama3`) |
| `--help` | Mostrar ayuda completa |

---

## Ejemplo de respuesta

```
============================================================
  RESPUESTA  [mistral]
============================================================
El estudio empleó un diseño cuasi-experimental con grupo control y
grupo experimental, aplicado a una muestra de 64 estudiantes de
educación secundaria. El instrumento de medición fue un cuestionario
validado con alfa de Cronbach de 0.87.

------------------------------------------------------------
  FUENTES UTILIZADAS
------------------------------------------------------------
  * articulo_metodologia.txt | chunk 3 | score 0.8921
  * articulo_metodologia.txt | chunk 4 | score 0.8134
```

> El **score** indica la similitud semántica con la consulta (mayor = más relevante). Scores por debajo de 0.55 suelen indicar que el corpus no contiene información suficiente sobre el tema consultado.

---

## Decisiones técnicas

| Componente | Tecnología | Razón principal |
|-----------|-----------|----------------|
| Motor LLM | Ollama | Ejecución local, sin costo, sin envío de datos |
| Búsqueda vectorial | FAISS | Precisión exacta, optimización SIMD, sin errores de punto flotante |
| Embeddings | sentence-transformers | Modelos multilingües preentrenados, funciona offline |
| Cliente HTTP | requests | Control directo sobre parámetros de generación |
| CLI | argparse | Stdlib de Python, sin dependencias adicionales |

Para una justificación completa de cada decisión (incluyendo comparativa con Gemini API, análisis del problema de precisión vectorial encontrado y estructura del prompt), ver el [documento técnico](docs/Sistema_RAG_Justificacion_Disenio.docx).

---

## Parámetros de configuración

Editables directamente en `main.py`:

```python
DOCS_FOLDER  = "docs"       # Carpeta con los documentos fuente
INDEX_FOLDER = "vector_db"  # Carpeta donde se guarda el índice

CHUNK_SIZE = 600   # Tamaño de chunk en caracteres
OVERLAP    = 150   # Solapamiento entre chunks consecutivos
```

Y en `rag_engine.py` los parámetros del modelo:

```python
temperature    = 0.1   # Baja aleatoriedad para respuestas factuales
num_predict    = 512   # Tokens máximos en la respuesta
repeat_penalty = 1.1   # Penalización de repetición
```

---

## Pruebas realizadas

| Prueba | Resultado |
|--------|-----------|
| Indexación de 3 documentos (~12.000 palabras) | ✅ 47 chunks, sin errores |
| Consulta sobre metodología específica | ✅ Respuesta precisa, fuentes correctas |
| Extracción de datos estadísticos (valores p, medias) | ✅ Valores numéricos exactos |
| Consulta fuera del corpus | ✅ Indica ausencia sin inventar datos |
| Sesión interactiva con múltiples preguntas | ✅ Índice persistente, ~4s por consulta |

Hardware de prueba: Intel Core i7 11th Gen, 16 GB RAM, sin GPU dedicada.

---

## Posibles errores frecuentes

**`No hay índice en 'vector_db/'`**
→ Ejecuta primero `python main.py --index`

**`Connection refused` al hacer una consulta**
→ Ollama no está corriendo. Inícialo con `ollama serve` o abre la app de Ollama.

**Respuestas en inglés**
→ El system prompt instruye al modelo a responder en español. Si persiste, prueba con un modelo diferente (`--model mistral` suele ser más consistente).

**`No se encontraron archivos .txt en 'docs/'`**
→ Verifica que tus documentos estén en `docs/` y tengan extensión `.txt` en minúsculas.

---

## Formato de archivos soportado

Actualmente el sistema procesa únicamente archivos `.txt` con codificación UTF-8. Para otros formatos:

```bash
# PDF a TXT con pdfplumber
pip install pdfplumber
python -c "import pdfplumber; f=pdfplumber.open('doc.pdf'); open('doc.txt','w').write('\n'.join(p.extract_text() for p in f.pages if p.extract_text()))"

# DOCX a TXT con pandoc
pandoc documento.docx -o documento.txt
```

---

## Licencia

MIT — libre para uso académico y comercial. Ver [LICENSE](LICENSE).

---

<div align="center">
  <sub>Construido con Ollama · FAISS · sentence-transformers</sub>
</div>
