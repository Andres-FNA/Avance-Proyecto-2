"""
rag_engine.py
Mecanismo de consulta RAG con Ollama:
  1. Recupera chunks relevantes del VectorStore (con umbral mínimo de score)
  2. Si no hay contexto suficiente → responde que no tiene esa información
  3. Si hay contexto → construye prompt estricto y consulta al LLM via Ollama
"""

import re
import requests
from typing import List, Tuple

from vector_store import VectorStore


# ─────────────────────────────────────────────
# Normalización
# ─────────────────────────────────────────────

def normalize(text: str) -> str:
    """Minúsculas, sin tildes, sin puntuación."""
    text = text.lower()
    for a, b in [("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u"),("ü","u"),("ñ","n")]:
        text = text.replace(a, b)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text


# ─────────────────────────────────────────────
# Detección de documento relevante
# ─────────────────────────────────────────────

def detect_source_filter(question: str, sources: list) -> str | None:
    """
    Detecta si la pregunta menciona un documento específico indexado.
    - Normaliza tildes y puntuación
    - Penaliza fuentes con sufijo romano/numérico distinto al mencionado
    - Requiere score >= 2 para activar filtro
    """
    q_norm   = normalize(question)
    q_tokens = set(t for t in q_norm.split() if len(t) > 3)

    roman_in_q = re.findall(r'\b(i{1,3}|iv|vi{0,3}|ix|\d+)\b', q_norm)

    best_source = None
    best_score  = 0

    for source in sources:
        src_norm   = normalize(source)
        src_tokens = set(t for t in src_norm.split() if len(t) > 3)

        hits = len(q_tokens & src_tokens)
        if hits < 2:
            continue

        score = hits

        if roman_in_q:
            src_romans = re.findall(r'\b(i{1,3}|iv|vi{0,3}|ix|\d+)\b', src_norm)
            if roman_in_q[0] not in src_romans:
                score -= 5
            else:
                score += 2

        if score > best_score:
            best_score  = score
            best_source = source

    return best_source if best_score >= 2 else None


# ─────────────────────────────────────────────
# Configuración Ollama
# ─────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"
TOP_K           = 5       # Reducido a 5 para menos ruido
MIN_SCORE       = 0.30    # Subido a 0.30 para ser más estricto


# ─────────────────────────────────────────────
# Gestión de modelos Ollama
# ─────────────────────────────────────────────

EMBEDDING_ONLY_MODELS = {
    "nomic-embed-text", "mxbai-embed-large",
    "all-minilm", "snowflake-arctic-embed"
}


def get_available_models() -> List[str]:
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        all_models = [m["name"] for m in resp.json().get("models", [])]
        return [
            m for m in all_models
            if not any(emb in m for emb in EMBEDDING_ONLY_MODELS)
        ]
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "No se pudo conectar a Ollama.\n"
            "Asegurate de que Ollama esté corriendo: ollama serve"
        )


def select_model(model_name: str = None) -> str:
    models = get_available_models()

    if not models:
        raise RuntimeError(
            "No hay modelos instalados en Ollama.\n"
            "Instala uno con: ollama pull mistral"
        )

    if model_name:
        match = next((m for m in models if model_name in m), None)
        if match:
            print(f"  [OLLAMA] Usando modelo: {match}")
            return match
        print(f"  [AVISO] Modelo '{model_name}' no encontrado. Selecciona uno de la lista.")

    print("\n  Modelos disponibles en Ollama:")
    for i, name in enumerate(models, 1):
        print(f"    {i}. {name}")

    while True:
        try:
            choice = input(f"\n  Selecciona un modelo [1-{len(models)}]: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                selected = models[idx]
                print(f"  [OLLAMA] Modelo seleccionado: {selected}")
                return selected
            print(f"  Ingresa un numero entre 1 y {len(models)}.")
        except ValueError:
            print("  Ingresa un numero valido.")


# ─────────────────────────────────────────────
# Construcción del prompt
# ─────────────────────────────────────────────

def build_prompt(query: str, retrieved_chunks: List[Tuple[dict, float]]) -> str:
    context_parts = []
    for i, (meta, score) in enumerate(retrieved_chunks, 1):
        context_parts.append(
            f"[Fragmento {i} | Fuente: {meta['source']} | Relevancia: {score:.3f}]\n"
            f"{meta['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    return f"""Eres un asistente que extrae información TEXTUALMENTE desde fragmentos de documentos.

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
{context}

=== PREGUNTA ===
{query}

=== RESPUESTA ==="""


# ─────────────────────────────────────────────
# Llamada al LLM via Ollama
# ─────────────────────────────────────────────

def call_ollama(prompt: str, model: str) -> str:
    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
    }

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Ollama no está disponible. Ejecuta: ollama serve")
    except requests.exceptions.Timeout:
        raise RuntimeError("El modelo tardó demasiado. Intenta con un modelo más pequeño.")

    return resp.json().get("response", "").strip()


# ─────────────────────────────────────────────
# HyDE: Hypothetical Document Embedding
# ─────────────────────────────────────────────

HYDE_PROMPT = """Eres un asistente que genera respuestas hipotéticas para mejorar búsquedas en documentos.
Dado la siguiente pregunta, genera un párrafo corto (4-6 oraciones) que parezca ser el fragmento
de un documento académico o institucional que respondería directamente a esa pregunta.
Escribe SOLO el párrafo, sin introducción ni explicación. y que se enfoque en responder la pregunta que se hace, como si fuera un fragmento de texto que podría estar en los documentos indexados.
no te explayes ni agregues información adicional, solo responde con el párrafo hipotético que responda la pregunta de la mejor manera posible, como si fuera un fragmento de texto que podría estar en los documentos indexados.

Pregunta: {question}

Párrafo hipotético:"""


def generate_hypothetical_answer(question: str, model: str) -> str:
    prompt = HYDE_PROMPT.format(question=question)
    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 150},
    }
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip() or question
    except Exception:
        return question


# ─────────────────────────────────────────────
# Re-ranking por fuente detectada
# ─────────────────────────────────────────────

def rerank_by_source(
    chunks: List[Tuple[dict, float]],
    source_filter: str,
    boost: float = 0.20,
) -> List[Tuple[dict, float]]:
    """
    Boost de +0.20 a chunks del documento detectado.
    Penalización de -0.10 a chunks de otros documentos cuando hay filtro activo.
    """
    if not source_filter:
        return chunks

    reranked = []
    for meta, score in chunks:
        if source_filter.lower() in meta["source"].lower():
            reranked.append((meta, score + boost))
        else:
            reranked.append((meta, score - 0.10))

    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked


# ─────────────────────────────────────────────
# Deduplicación por similitud de texto (Jaccard)
# ─────────────────────────────────────────────

def deduplicate_chunks(
    chunks: List[Tuple[dict, float]],
    similarity_threshold: float = 0.6,
) -> List[Tuple[dict, float]]:
    """
    Elimina chunks cuyo texto sea muy similar al de otro ya seleccionado.
    Cuando dos chunks son similares, conserva el más largo (más completo).
    Esto resuelve el problema de chunks duplicados que dicen lo mismo.
    """
    selected = []

    for meta, score in chunks:
        words_new    = set(normalize(meta["text"]).split())
        is_duplicate = False

        for i, (sel_meta, sel_score) in enumerate(selected):
            words_sel    = set(normalize(sel_meta["text"]).split())
            if not words_new or not words_sel:
                continue

            intersection = len(words_new & words_sel)
            union        = len(words_new | words_sel)
            jaccard      = intersection / union if union > 0 else 0

            if jaccard >= similarity_threshold:
                # Conservar el más largo
                if len(meta["text"]) > len(sel_meta["text"]):
                    selected[i] = (meta, score)
                is_duplicate = True
                break

        if not is_duplicate:
            selected.append((meta, score))

    return selected


# ─────────────────────────────────────────────
# Motor RAG principal
# ─────────────────────────────────────────────

NO_CONTEXT_MESSAGE = (
    "No tengo información sobre ese tema en los documentos disponibles.\n\n"
    "Sugerencias:\n"
    "  • Reformula la pregunta con términos más cercanos al contenido de los documentos\n"
    "  • Verifica que los documentos relevantes estén en docs/ y hayan sido indexados\n"
    "  • Ejecuta python main.py --index si agregaste documentos nuevos"
)


class RAGEngine:
    """
    Orquesta el flujo RAG:
      Consulta → Recuperación → Deduplicación (Jaccard) → Re-ranking → Prompt → LLM
    """

    def __init__(
        self,
        vector_store: VectorStore,
        model: str,
        top_k: int = TOP_K,
        min_score: float = MIN_SCORE,
    ):
        self.vector_store = vector_store
        self.model        = model
        self.top_k        = top_k
        self.min_score    = min_score

    def query(self, question: str, verbose: bool = True) -> dict:
        if verbose:
            print(f"\n[RAG] Buscando contexto para: '{question}'")

        # ── Paso 0: Detectar documento específico ─────────────────
        all_sources   = self.vector_store.list_sources()
        source_filter = detect_source_filter(question, all_sources)
        if verbose:
            if source_filter:
                print(f"[RAG] Documento detectado: '{source_filter}'")
            else:
                print("[RAG] Sin documento específico detectado — búsqueda global")

        # ── Paso 1: HyDE ───────────────────────────────────────────
        if verbose:
            print("[RAG] Generando respuesta hipotética (HyDE)...")
        search_query = generate_hypothetical_answer(question, self.model)
        if verbose:
            preview = search_query[:80].replace("\n", " ")
            print(f"[RAG] Query expandida: '{preview}...'")

        # ── Paso 2: Búsqueda MMR con ambas queries ─────────────────
        retrieved_hyde   = self.vector_store.search_mmr(
            search_query, top_k=self.top_k, min_score=self.min_score,
            source_filter=source_filter,
        )
        retrieved_direct = self.vector_store.search_mmr(
            question, top_k=self.top_k, min_score=self.min_score,
            source_filter=source_filter,
        )

        # Si filtro estricto devuelve muy poco, ampliar globalmente
        if source_filter and len(retrieved_hyde) + len(retrieved_direct) < 2:
            if verbose:
                print("[RAG] Pocos resultados con filtro, ampliando búsqueda global + re-ranking...")
            retrieved_hyde   = self.vector_store.search_mmr(
                search_query, top_k=self.top_k * 2, min_score=self.min_score,
            )
            retrieved_direct = self.vector_store.search_mmr(
                question, top_k=self.top_k * 2, min_score=self.min_score,
            )

        # ── Paso 3: Deduplicar por chunk_id ───────────────────────
        seen_ids = set()
        combined = []
        for meta, score in retrieved_hyde + retrieved_direct:
            cid = meta["chunk_id"]
            if cid not in seen_ids:
                seen_ids.add(cid)
                combined.append((meta, score))

        # ── Paso 4: Deduplicar por similitud de texto (Jaccard) ───
        combined = deduplicate_chunks(combined, similarity_threshold=0.6)

        # ── Paso 5: Re-rankear boosteando el documento detectado ──
        combined = rerank_by_source(combined, source_filter)

        # Limitar a top_k
        retrieved = combined[:self.top_k]

        if verbose:
            if retrieved:
                print(f"[RAG] {len(retrieved)} chunks tras deduplicación y re-ranking:")
                for meta, score in retrieved:
                    print(f"      * [{score:.3f}] {meta['source']} — chunk {meta['chunk_id']}")
            else:
                print(f"[RAG] Sin chunks sobre el umbral ({self.min_score}). No se llama al LLM.")

        if not retrieved:
            return {
                "question":         question,
                "model":            self.model,
                "retrieved_chunks": [],
                "prompt":           "",
                "answer":           NO_CONTEXT_MESSAGE,
            }

        # ── Paso 6: Construir prompt y consultar LLM ──────────────
        prompt = build_prompt(question, retrieved)

        if verbose:
            print(f"[RAG] Consultando modelo '{self.model}'...")

        answer = call_ollama(prompt, self.model)

        return {
            "question":         question,
            "model":            self.model,
            "retrieved_chunks": retrieved,
            "prompt":           prompt,
            "answer":           answer,
        } 
