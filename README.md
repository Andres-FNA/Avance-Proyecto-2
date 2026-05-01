# Sistema RAG Local con Ollama + FAISS

> Asistente experto basado en Retrieval-Augmented Generation que opera completamente de forma local, preservando la privacidad de los datos.

**Avance 2 — Desarrollo de Asistente Experto Basado en RAG**

---

## Descripción General

Este sistema permite "leer" una base de conocimientos específica (manuales técnicos, leyes, historiales médicos, documentos académicos) y responder preguntas sobre ella mediante un pipeline RAG. Todos los componentes corren en la máquina del usuario sin conexión a servicios externos: los embeddings y el modelo de lenguaje se sirven a través de **Ollama**, y el índice vectorial se gestiona con **FAISS**.

**Formatos soportados:** `.txt` · `.pdf` · `.docx`

---

## Arquitectura

```
main.py                  ← Punto de entrada CLI
├── document_loader.py   ← Carga y chunking de documentos
├── vector_store.py      ← Embeddings (Ollama) + índice FAISS
└── rag_engine.py        ← Pipeline completo de consulta RAG
```

| Módulo | Responsabilidad |
|---|---|
| `document_loader.py` | Carga `.txt`, `.pdf`, `.docx` y los divide en chunks con solapamiento |
| `vector_store.py` | Vectoriza con `nomic-embed-text` y almacena en FAISS |
| `rag_engine.py` | Orquesta HyDE → MMR → Dedup → Re-ranking → LLM |
| `main.py` | CLI con modos `--index`, `--query`, `--interactive`, `--models` |

---

## Flujo del Sistema RAG

El pipeline se divide en dos fases: **indexación** (ocurre una vez al agregar documentos) y **consulta** (ocurre en cada pregunta).

### Fase 1 — Indexación

```
docs/
 ├── archivo.pdf
 ├── manual.docx
 └── ley.txt
       │
       ▼
  load_documents_from_folder()
       │
       ▼
  chunk_text()  →  Chunks de 1500 chars, overlap 250
       │
       ▼
  embed_batch()  →  nomic-embed-text via Ollama
       │
       ▼
  VectorStore.build_index()  →  FAISS IndexFlatIP + normalización L2
       │
       ▼
  vector_db/
   ├── faiss.index
   ├── metadata.json
   └── config.json
```

**Paso 1 — Carga de documentos** (`document_loader.py`)

`load_documents_from_folder()` recorre `docs/` y aplica el lector correspondiente:
- `load_txt()` — lectura directa UTF-8
- `load_pdf()` — extracción página a página con `pypdf`
- `load_docx()` — párrafos y tablas con `python-docx`

**Paso 2 — Creación de chunks** (`document_loader.py`)

`chunk_text()` divide cada documento con solapamiento para preservar contexto entre fragmentos consecutivos. El corte busca el último espacio o salto de línea disponible, evitando partir palabras. Cada `Chunk` lleva: `source`, `chunk_id` y `start_char`.

```python
CHUNK_SIZE = 1500  # caracteres por fragmento
OVERLAP    = 250   # solapamiento entre chunks
```

**Paso 3 — Vectorización semántica** (`vector_store.py`)

`embed_batch()` envía los textos en lotes de 32 al endpoint `/api/embed` de Ollama con el modelo `nomic-embed-text`. Los vectores se normalizan con norma L2 para que el producto interno sea equivalente a similitud coseno.

**Paso 4 — Índice FAISS** (`vector_store.py`)

`VectorStore.build_index()` construye un `IndexFlatIP` y persiste tres archivos en `vector_db/`: el índice binario, los metadatos de cada chunk y la configuración del modelo de embeddings.

---

### Fase 2 — Consulta (`RAGEngine.query`)

```
Pregunta del usuario
       │
       ▼
  detect_source_filter()     ← ¿menciona un documento específico?
       │
       ▼
  generate_hypothetical_answer()   ← HyDE: respuesta hipotética
       │
       ├──────────────────────────┐
       ▼                          ▼
  search_mmr(hyde_query)    search_mmr(pregunta_literal)
       │                          │
       └──────────┬───────────────┘
                  ▼
         deduplicate_chunks()    ← Jaccard ≥ 0.60
                  │
                  ▼
         rerank_by_source()      ← boost +0.20 doc detectado
                  │
                  ▼
           build_prompt()
                  │
                  ▼
           call_ollama()
                  │
                  ▼
         Respuesta + fuentes
```

**Paso 0 — Detección de documento específico**

`detect_source_filter()` normaliza la pregunta (minúsculas, sin tildes, sin puntuación) y la compara contra los nombres de los documentos indexados. Si detecta coincidencia (score ≥ 2 tokens), activa un filtro de fuente para priorizar ese documento.

**Paso 1 — HyDE (Hypothetical Document Embedding)**

Antes de buscar, el LLM genera una respuesta hipotética corta (4-6 oraciones, `temperature=0.3`) a la pregunta. Esta respuesta se usa como query de búsqueda porque su embedding se acerca más al espacio vectorial de los documentos que el de una pregunta literal, mejorando el recall semántico.

**Paso 2 — Búsqueda MMR (Maximum Marginal Relevance)**

Se ejecutan dos búsquedas en paralelo: con la query HyDE y con la pregunta literal. MMR selecciona iterativamente el chunk más relevante que sea además suficientemente distinto de los ya seleccionados (`diversity=0.35`), evitando devolver fragmentos casi idénticos del mismo párrafo.

```python
TOP_K     = 5     # chunks máximos en la respuesta
MIN_SCORE = 0.30  # umbral mínimo de similitud coseno
```

**Pasos 3 y 4 — Deduplicación**

Los resultados de ambas búsquedas se combinan eliminando primero duplicados exactos por `chunk_id`. Luego se aplica deduplicación semántica por similitud de Jaccard sobre palabras normalizadas: si dos chunks tienen Jaccard ≥ 0.60 se considera el mismo fragmento y se conserva el más largo.

**Paso 5 — Re-ranking por fuente**

Si se detectó un documento específico, `rerank_by_source()` aplica:
- `+0.20` al score de chunks del documento detectado
- `-0.10` al score de chunks de otros documentos

**Paso 6 — Prompt y LLM**

Se construye el prompt con `build_prompt()` y se llama a Ollama vía `call_ollama()`.

---

## Estructuración del Prompt

### Estructura

```
Eres un asistente que extrae información TEXTUALMENTE desde fragmentos de documentos.

REGLAS ABSOLUTAS:
1. Responde ÚNICAMENTE con texto que aparezca en los fragmentos.
2. Si la respuesta NO está en ningún fragmento, responde SOLO:
   "No tengo información sobre ese tema en los documentos disponibles."
3. USA SOLO EL FRAGMENTO MÁS COMPLETO que responda la pregunta.
4. NO uses "..." ni cortes el texto.
5. NO uses fragmentos de documentos distintos al preguntado.

=== CONTEXTO ===
[Fragmento 1 | Fuente: archivo.pdf | Relevancia: 0.742]
[texto del fragmento]

---

[Fragmento N | Fuente: otro.docx | Relevancia: 0.611]
[texto del fragmento]

=== PREGUNTA ===
[pregunta del usuario]

=== RESPUESTA ===
```

### Configuration system

Las reglas absolutas al inicio del prompt actúan como capa de instrucción del sistema. Para el paso HyDE se aplica configuración adicional explícita:

```python
"options": {"temperature": 0.3, "num_predict": 150}
```

### Formato de salida

El modelo produce respuestas con formato estructurado:

```
Del Fragmento X (nombre_archivo):
[texto copiado literalmente del documento]
```

`RAGEngine.query()` retorna un dict estandarizado con todos los campos para trazabilidad:

```python
{
    "question":         str,   # pregunta original
    "model":            str,   # modelo LLM usado
    "retrieved_chunks": list,  # [(metadata, score), ...]
    "prompt":           str,   # prompt completo enviado
    "answer":           str,   # respuesta del modelo
}
```

---

## Instalación

### Requisitos previos

- Python 3.9+
- [Ollama](https://ollama.com) instalado y ejecutándose

```bash
ollama serve
ollama pull nomic-embed-text   # modelo de embeddings
ollama pull mistral            # modelo de chat (o llama3, etc.)
```

### Dependencias Python

```bash
pip install faiss-cpu numpy requests pypdf python-docx
```

### Estructura de carpetas

```
proyecto/
├── docs/               ← colocar aquí los documentos a indexar
├── vector_db/          ← generado automáticamente al indexar
├── main.py
├── document_loader.py
├── vector_store.py
└── rag_engine.py
```

---

## Uso

```bash
# Indexar documentos en docs/
py main.py --index

# Consulta directa
py main.py --query "Cual es el articulo 22 del reglamento academico"

# Consulta con modelo fijo
py main.py --query "..." --model mistral

# Modo interactivo (múltiples preguntas)
py main.py --interactive

# Listar modelos disponibles en Ollama
py main.py --models
```

> **Windows:** siempre usar comillas en `--query` para consultas con espacios.

---

## Dependencias

| Paquete | Uso |
|---|---|
| `faiss-cpu` | Índice vectorial y búsqueda por similitud coseno |
| `numpy` | Operaciones matriciales sobre embeddings |
| `requests` | Comunicación HTTP con la API de Ollama |
| `pypdf` | Extracción de texto de archivos PDF |
| `python-docx` | Lectura de documentos Word (.docx) |
| `ollama` | Servidor local para embeddings y modelos LLM |
