# Sistema RAG Local con Ollama + FAISS

> Asistente experto basado en Retrieval-Augmented Generation que opera completamente de forma local, preservando la privacidad de los datos.

**Avance 2 — Desarrollo de Asistente Experto Basado en RAG**

---

## Tabla de contenidos

1. [Descripción General](#1-descripción-general)
2. [Arquitectura del Sistema](#2-arquitectura-del-sistema)
3. [Flujo del Sistema RAG](#3-flujo-del-sistema-rag)
   - [Fase 1 — Indexación](#fase-1--indexación)
   - [Fase 2 — Consulta](#fase-2--consulta-ragenginequery)
4. [Estructuración del Prompt](#4-estructuración-del-prompt)
5. [Instalación y Uso](#5-instalación-y-uso)
6. [Dependencias](#6-dependencias)

---

## 1. Descripción General

Este proyecto implementa un sistema de Inteligencia Artificial que opera de forma completamente local, preservando la privacidad de los datos. El sistema es capaz de "leer" una base de conocimientos específica —manuales técnicos, leyes, historiales médicos, documentos académicos— y responder preguntas sobre ella mediante un pipeline **RAG (Retrieval-Augmented Generation)**.

Todos los componentes corren en la máquina del usuario sin conexión a servicios externos: los embeddings semánticos y el modelo de lenguaje se sirven a través de **Ollama**, y el índice vectorial se gestiona con **FAISS**.

**Formatos de documentos soportados:**
- Texto plano (`.txt`)
- PDF con texto embebido (`.pdf`)
- Documentos Word (`.docx`)

---

## 2. Arquitectura del Sistema

El sistema se compone de cuatro módulos principales que interactúan de forma orquestada:

```
main.py                  ← Punto de entrada CLI
├── document_loader.py   ← Carga, limpieza y chunking de documentos
├── vector_store.py      ← Embeddings semánticos (Ollama) + índice FAISS
└── rag_engine.py        ← Pipeline completo de consulta RAG
```

| Módulo | Archivo | Responsabilidad |
|---|---|---|
| Cargador de documentos | `document_loader.py` | Carga y limpia `.txt`, `.pdf` y `.docx`, y los divide en chunks con solapamiento |
| Almacén vectorial | `vector_store.py` | Vectoriza con `nomic-embed-text` via Ollama y construye el índice FAISS |
| Motor RAG | `rag_engine.py` | Orquesta el pipeline completo: HyDE → MMR → Dedup → Re-ranking → LLM |
| Punto de entrada | `main.py` | CLI con modos `--index`, `--query`, `--interactive` y `--models` |

---

## 3. Flujo del Sistema RAG

El pipeline se divide en dos fases bien diferenciadas: la **indexación** (que ocurre una sola vez al agregar documentos) y la **consulta** (que ocurre en cada pregunta del usuario).

---

### Fase 1 — Indexación

Esta fase procesa los documentos crudos y construye la base de datos vectorial persistente en disco.

```
docs/
 ├── reglamento.pdf
 ├── manual.docx
 └── ley.txt
       │
       ▼
  load_documents_from_folder()    ← lee y filtra por extensión
       │
       ▼
  chunk_text()                    ← fragmenta con solapamiento
  CHUNK_SIZE=1500 / OVERLAP=250
       │
       ▼
  embed_batch()                   ← nomic-embed-text via Ollama
  lotes de 32 chunks / norma L2
       │
       ▼
  VectorStore.build_index()       ← FAISS IndexFlatIP
       │
       ▼
  vector_db/
   ├── faiss.index      ← índice binario
   ├── metadata.json    ← texto + metadatos de cada chunk
   └── config.json      ← modelo de embeddings usado
```

#### Paso 1 — Carga de documentos (`document_loader.py`)

La función `load_documents_from_folder()` recorre la carpeta `docs/` y aplica el lector correspondiente según la extensión del archivo:

- **`load_txt()`** — lectura directa con codificación UTF-8 y manejo de errores de encoding.
- **`load_pdf()`** — extracción página a página usando `pypdf`. Si una página no tiene texto embebido (PDF escaneado), se advierte al usuario y se omite esa página. Si el documento completo está vacío, se omite con aviso.
- **`load_docx()`** — extrae texto de párrafos normales y del contenido de tablas (celda a celda) usando `python-docx`.

Los documentos sin contenido útil se omiten con un mensaje de aviso. Los que se cargan exitosamente reportan su tamaño en caracteres.

#### Paso 2 — Creación de chunks (`document_loader.py`)

La función `chunk_text()` divide cada documento en fragmentos con solapamiento para preservar contexto entre chunks consecutivos. La configuración usada en el sistema es:

```python
CHUNK_SIZE = 1500  # caracteres por fragmento
OVERLAP    = 250   # solapamiento entre chunks consecutivos
```

El corte **nunca se realiza en el carácter exacto**: la función busca el último espacio o salto de línea disponible después del punto medio del chunk, evitando cortar palabras a la mitad. El solapamiento se implementa haciendo que cada chunk empiece `OVERLAP` caracteres antes del final del chunk anterior, de forma que ningún concepto quede partido entre dos fragmentos sin contexto.

Cada objeto `Chunk` resultante lleva los siguientes metadatos:

| Campo | Descripción |
|---|---|
| `text` | Contenido textual del fragmento |
| `source` | Nombre del archivo original |
| `chunk_id` | Identificador único secuencial global |
| `start_char` | Posición de inicio en el texto original |

#### Paso 3 — Vectorización semántica (`vector_store.py`)

La función `embed_batch()` envía los textos en lotes de 32 al endpoint `/api/embed` de Ollama usando el modelo `nomic-embed-text` (~274M parámetros). Este modelo genera embeddings de 768 dimensiones que capturan el significado semántico del texto, permitiendo encontrar fragmentos relevantes aunque no compartan palabras exactas con la pregunta —algo que TF-IDF no puede hacer.

Los vectores resultantes se **normalizan con norma L2**: dividir cada vector por su magnitud hace que el producto interno entre dos vectores normalizados sea equivalente a la similitud coseno, lo que mejora la precisión de las búsquedas.

```python
norms = np.linalg.norm(matrix, axis=1, keepdims=True)
norms[norms == 0] = 1.0
matrix /= norms
```

#### Paso 4 — Construcción del índice FAISS (`vector_store.py`)

`VectorStore.build_index()` crea un índice `IndexFlatIP` (producto interno plano) en FAISS y agrega todos los vectores normalizados. La estructura se persiste en tres archivos dentro de `vector_db/`:

- **`faiss.index`** — el índice binario de FAISS con todos los vectores.
- **`metadata.json`** — lista paralela al índice con el texto y metadatos de cada chunk.
- **`config.json`** — registra qué modelo de embeddings se usó, para garantizar coherencia al cargar el índice en consultas futuras.

---

### Fase 2 — Consulta (`RAGEngine.query`)

Esta fase se activa en cada pregunta del usuario. El motor `RAGEngine` orquesta seis pasos en secuencia para maximizar la precisión de la respuesta.

```
Pregunta del usuario
       │
       ▼
  [Paso 0] detect_source_filter()
  Normaliza pregunta y detecta si menciona un documento específico
       │
       ▼
  [Paso 1] generate_hypothetical_answer()   ← HyDE
  El LLM genera una respuesta hipotética para mejorar la búsqueda
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
  [Paso 2a]                         [Paso 2b]
  search_mmr(query_hyde)       search_mmr(pregunta_literal)
  top_k=5, min_score=0.30      top_k=5, min_score=0.30
       │                                 │
       └───────────────┬─────────────────┘
                       ▼
  [Paso 3] Deduplicación por chunk_id
  Elimina duplicados exactos de ambas búsquedas
                       │
                       ▼
  [Paso 4] deduplicate_chunks()   ← Jaccard semántico
  Si similitud ≥ 0.60, conserva solo el chunk más largo
                       │
                       ▼
  [Paso 5] rerank_by_source()
  boost +0.20 al doc detectado / penalización -0.10 a otros
                       │
                       ▼
  [Paso 6] build_prompt()  →  call_ollama()
                       │
                       ▼
            Respuesta + fuentes utilizadas
```

#### Paso 0 — Detección de documento específico

`detect_source_filter()` normaliza la pregunta eliminando tildes, convirtiendo a minúsculas y quitando puntuación, y la compara tokenizada contra los nombres de todos los documentos indexados. Si detecta que la pregunta menciona un documento específico (score ≥ 2 tokens coincidentes), activa un filtro de fuente que priorizará ese documento durante la búsqueda.

La función también maneja sufijos romanos y numéricos: si la pregunta menciona "reglamento I" no activará el filtro para "reglamento II", penalizando con `-5` al score de fuentes con numeración diferente a la mencionada.

Si el filtro activo produce menos de 2 resultados combinados, el sistema amplía automáticamente la búsqueda de forma global y aplica el re-ranking posterior para compensar.

#### Paso 1 — HyDE (Hypothetical Document Embedding)

Antes de buscar en el índice vectorial, el sistema le pide al propio LLM que genere una **respuesta hipotética corta** (4-6 oraciones) a la pregunta del usuario, usando `temperature=0.3` y un límite de 150 tokens.

Esta respuesta hipotética se usa como query de búsqueda en lugar de la pregunta literal. La ventaja es fundamental: el embedding de una *respuesta* se acerca mucho más al espacio vectorial de los fragmentos de documentos —que también son respuestas y afirmaciones— que el embedding de una *pregunta*. Esto mejora significativamente el recall semántico, especialmente para preguntas que usan vocabulario distinto al de los documentos.

```
Pregunta: "¿Cuántos créditos necesito para graduarme?"

HyDE:     "El estudiante debe completar un mínimo de 180 créditos
           distribuidos entre asignaturas obligatorias y electivas..."
           ↑ Este embedding estará mucho más cerca de los artículos
             del reglamento que el embedding de la pregunta original.
```

#### Paso 2 — Búsqueda MMR (Maximum Marginal Relevance)

Se ejecutan **dos búsquedas en paralelo**: una con la query HyDE y otra con la pregunta literal del usuario. Cada búsqueda usa el algoritmo **MMR**, que selecciona iterativamente los chunks de la siguiente forma:

1. El primer chunk seleccionado es siempre el de mayor similitud con la query.
2. Para cada chunk siguiente, calcula: `MMR = (1 - diversity) × relevancia - diversity × max_similitud_con_ya_seleccionados`
3. Selecciona el chunk con mayor MMR score.

El parámetro `diversity=0.35` balancea relevancia y diversidad, evitando devolver múltiples fragmentos casi idénticos del mismo párrafo. Los parámetros de la búsqueda son:

```python
TOP_K     = 5     # chunks máximos en la respuesta final
MIN_SCORE = 0.30  # umbral mínimo de similitud coseno
```

Solo se consideran chunks que superen el umbral mínimo. Si ninguno lo supera, el sistema responde directamente sin llamar al LLM, ahorrando tiempo y recursos.

#### Pasos 3 y 4 — Deduplicación

Los resultados de ambas búsquedas (HyDE + literal) se combinan en una sola lista y se deduplicen en dos etapas:

**Deduplicación exacta por `chunk_id`:** elimina cualquier chunk que aparezca en ambas búsquedas, conservando la primera ocurrencia.

**Deduplicación semántica por similitud de Jaccard:** para cada par de chunks restantes, calcula el índice de Jaccard sobre el conjunto de palabras normalizadas de cada texto:

```
Jaccard(A, B) = |palabras(A) ∩ palabras(B)| / |palabras(A) ∪ palabras(B)|
```

Si `Jaccard ≥ 0.60`, los dos chunks se consideran sustancialmente el mismo fragmento —por ejemplo, el solapamiento entre chunks consecutivos del mismo documento—. En ese caso se conserva únicamente el **más largo** de los dos, ya que contiene más información completa. Esto resuelve el problema de chunks duplicados que el modelo trataría como contexto redundante.

#### Paso 5 — Re-ranking por fuente

Si se detectó un documento específico en el Paso 0, `rerank_by_source()` reajusta los scores de todos los chunks:

| Condición | Ajuste al score |
|---|---|
| Chunk del documento detectado | `+0.20` |
| Chunk de cualquier otro documento | `−0.10` |

Los chunks se reordenan por score ajustado y se limitan al `top_k=5` final. Esto garantiza que si el usuario preguntó sobre un documento específico, los fragmentos de ese documento aparezcan primero en el contexto del prompt.

#### Paso 6 — Construcción del prompt y llamada al LLM

Con los chunks finales seleccionados se construye el prompt estructurado con `build_prompt()` y se envía al LLM a través de `call_ollama()`. El endpoint usado es `/api/generate` de Ollama con `stream=false` y un timeout de 120 segundos. Este paso se describe en detalle en la siguiente sección.

---

## 4. Estructuración del Prompt

La función `build_prompt()` en `rag_engine.py` construye un prompt con tres secciones claramente delimitadas y un conjunto de reglas absolutas que controlan estrictamente el comportamiento del modelo.

### 4.1 Estructura completa del prompt

```
Eres un asistente que extrae información TEXTUALMENTE desde fragmentos de documentos.

REGLAS ABSOLUTAS:
1. Responde ÚNICAMENTE con texto que aparezca en los fragmentos.
2. Si la respuesta NO está en ningún fragmento, responde SOLO:
   "No tengo información sobre ese tema en los documentos disponibles."
3. USA SOLO EL FRAGMENTO MÁS COMPLETO que responda la pregunta.
   Si dos fragmentos dicen lo mismo, usa solo el más largo/completo, ignora el otro.
4. NO uses "..." ni cortes el texto. Si el fragmento termina abruptamente,
   copia exactamente hasta donde llega y NO agregues puntos suspensivos.
5. NO uses fragmentos de documentos distintos al preguntado si se mencionó uno específico.

CÓMO RESPONDER:
- Indica la fuente: "Del Fragmento X (nombre_archivo):"
- Copia el texto relevante tal como aparece, sin modificarlo ni cortarlo con "..."
- Una sola respuesta unificada, sin repetir información

=== CONTEXTO ===
[Fragmento 1 | Fuente: reglamento.pdf | Relevancia: 0.809]
[texto completo del fragmento 1]

---

[Fragmento 2 | Fuente: reglamento.pdf | Relevancia: 0.797]
[texto completo del fragmento 2]

---

[Fragmento N | Fuente: reglamento.pdf | Relevancia: 0.773]
[texto completo del fragmento N]

=== PREGUNTA ===
[pregunta literal del usuario]

=== RESPUESTA ===
```

### 4.2 Configuration system

Aunque Ollama no expone un campo `system` separado en su endpoint `/api/generate`, el **rol del sistema está embebido al inicio del prompt** como texto literal, siguiendo el patrón estándar para modelos locales. Las cinco reglas absolutas funcionan como la capa de instrucción del sistema y definen el comportamiento permitido del modelo:

| Regla | Propósito |
|---|---|
| Solo texto de fragmentos | Evita alucinaciones e invención de contenido |
| Mensaje predefinido si no hay info | Respuesta controlada y honesta ante ausencia de contexto |
| Fragmento más completo | Evita respuestas redundantes con información duplicada |
| Sin puntos suspensivos | Evita que el modelo "complete" fragmentos con contenido inventado |
| Sin mezcla de documentos | Respeta el alcance de la consulta cuando se especificó un documento |

Para el paso HyDE, se aplica una configuración adicional explícita que limita la creatividad y longitud de la respuesta hipotética:

```python
"options": {
    "temperature": 0.3,  # respuestas más deterministas y precisas
    "num_predict": 150   # máximo 150 tokens para la respuesta hipotética
}
```

### 4.3 Formato de salida

El prompt instruye al modelo a producir respuestas con el siguiente formato estructurado y predecible:

```
Del Fragmento X (nombre_archivo):
[texto copiado literalmente del documento sin modificaciones]
```

Adicionalmente, `RAGEngine.query()` retorna un **dict estandarizado** con todos los campos necesarios para trazabilidad, auditoría y depuración del sistema:

```python
{
    "question":         str,   # pregunta original del usuario
    "model":            str,   # nombre del modelo LLM utilizado
    "retrieved_chunks": list,  # lista de (metadata_dict, score_float)
    "prompt":           str,   # prompt completo enviado al LLM
    "answer":           str,   # respuesta generada por el modelo
}
```

El campo `retrieved_chunks` permite al sistema mostrar las fuentes utilizadas con su score de relevancia, aportando transparencia y trazabilidad a cada respuesta generada.

---

## 5. Instalación y Uso

### 5.1 Requisitos previos

- Python 3.9 o superior
- [Ollama](https://ollama.com) instalado y ejecutándose en `http://localhost:11434`

```bash
# Iniciar el servidor de Ollama
ollama serve

# Instalar el modelo de embeddings (obligatorio)
ollama pull nomic-embed-text

# Instalar al menos un modelo de chat
ollama pull mistral    # recomendado: mejor seguimiento de instrucciones
ollama pull llama3     # alternativa disponible
```

### 5.2 Instalación de dependencias Python

```bash
pip install faiss-cpu numpy requests pypdf python-docx
```

### 5.3 Estructura de carpetas

```
proyecto/
├── docs/               ← colocar aquí los documentos a indexar
│    ├── reglamento.pdf
│    ├── manual.docx
│    └── ley.txt
├── vector_db/          ← generado automáticamente al indexar
│    ├── faiss.index
│    ├── metadata.json
│    └── config.json
├── main.py
├── document_loader.py
├── vector_store.py
└── rag_engine.py
```

### 5.4 Comandos disponibles

| Comando | Descripción |
|---|---|
| `py main.py --index` | Indexa todos los documentos en `docs/` |
| `py main.py --query "..."` | Consulta directa con selección interactiva de modelo |
| `py main.py --query "..." --model mistral` | Consulta directa con modelo fijo |
| `py main.py --interactive` | Modo interactivo — múltiples preguntas sin reiniciar |
| `py main.py --interactive --model llama3` | Modo interactivo con modelo fijo |
| `py main.py --models` | Lista los modelos de chat disponibles en Ollama |

### 5.5 Flujo de uso típico

```bash
# 1. Colocar documentos en docs/

# 2. Indexar (solo necesario una vez, o al agregar documentos nuevos)
py main.py --index

# 3. Consultar en modo interactivo
py main.py --interactive
```

> **Nota para Windows:** usar siempre comillas dobles en `--query` para consultas con espacios:
> ```
> py main.py --query "Cual es el articulo 22 del reglamento academico"
> ```

En el modo interactivo se puede escribir `cambiar modelo` en cualquier momento para seleccionar otro modelo sin reiniciar el sistema, y `salir` para terminar.

---

## 6. Dependencias

| Paquete | Versión mínima | Uso en el sistema |
|---|---|---|
| `faiss-cpu` | 1.7+ | Índice vectorial `IndexFlatIP` y búsqueda por similitud coseno |
| `numpy` | 1.24+ | Operaciones matriciales sobre embeddings y normalización L2 |
| `requests` | 2.28+ | Comunicación HTTP con los endpoints de la API de Ollama |
| `pypdf` | 3.0+ | Extracción de texto página a página de archivos PDF |
| `python-docx` | 0.8+ | Lectura de párrafos y tablas en documentos Word (.docx) |
| `ollama` | — | Servidor local para el modelo de embeddings y los modelos LLM |
