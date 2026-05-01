"""
vector_store.py
Vectoriza chunks con embeddings semánticos via Ollama (nomic-embed-text)
y los almacena en un índice FAISS.

Por qué embeddings semánticos:
  - Entienden sinónimos y conceptos relacionados (TF-IDF no puede)
  - Reducen drásticamente las alucinaciones por contexto irrelevante
  - nomic-embed-text es local, gratuito y rápido (~274M parámetros)

Requisito previo:
  ollama pull nomic-embed-text
"""

import os
import json
import time
import numpy as np
import faiss
import requests
from typing import List, Tuple

from document_loader import Chunk


# ─────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────

OLLAMA_BASE_URL   = "http://localhost:11434"
EMBEDDING_MODEL   = "nomic-embed-text"   # cambiar aquí si usas otro modelo
EMBED_BATCH_SIZE  = 32                   # chunks por lote (ajustar según RAM)


# ─────────────────────────────────────────────
# Cliente de embeddings Ollama
# ─────────────────────────────────────────────

def embed_single(text: str, model: str = EMBEDDING_MODEL) -> List[float]:
    """
    Obtiene el embedding de un texto via Ollama /api/embeddings.
    Lanza RuntimeError si Ollama no está disponible.
    """
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": model, "input": text},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "No se pudo conectar a Ollama.\n"
            "Asegurate de que Ollama esté corriendo: ollama serve"
        )
    except KeyError:
        raise RuntimeError(
            f"Ollama no devolvió un embedding. ¿Está instalado el modelo '{model}'?\n"
            f"Instálalo con: ollama pull {model}"
        )


def embed_batch(texts: List[str], model: str = EMBEDDING_MODEL) -> np.ndarray:
    """
    Genera embeddings para una lista de textos en lotes.
    Muestra progreso durante la vectorización.
    """
    vectors = []
    total   = len(texts)

    for i in range(0, total, EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        batch_vectors = []

        for text in batch:
            vec = embed_single(text, model)
            batch_vectors.append(vec)

        vectors.extend(batch_vectors)

        done = min(i + EMBED_BATCH_SIZE, total)
        print(f"    Vectorizando... {done}/{total} chunks", end="\r")

    print()  # salto de línea al terminar
    matrix = np.array(vectors, dtype="float32")

    # Normalizar L2 para similitud coseno con IndexFlatIP
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix /= norms

    return matrix


def check_embedding_model(model: str = EMBEDDING_MODEL):
    """
    Verifica que el modelo de embeddings esté disponible en Ollama.
    Muestra advertencia si no lo está.
    """
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        if not any(model in m for m in models):
            print(f"\n  [AVISO] Modelo '{model}' no encontrado en Ollama.")
            print(f"  Instálalo con: ollama pull {model}\n")
    except Exception:
        pass  # Si Ollama no responde, el error real aparecerá al embedear


# ─────────────────────────────────────────────
# Clase principal: VectorStore
# ─────────────────────────────────────────────

class VectorStore:
    """
    Base de datos vectorial usando FAISS + embeddings semánticos de Ollama.
    Guarda embeddings + metadatos de cada chunk.
    Permite búsqueda semántica real por similitud coseno.
    """

    def __init__(self, embedding_model: str = EMBEDDING_MODEL):
        self.embedding_model = embedding_model
        self.index           = None
        self.metadata: List[dict] = []
        print(f"  [MODELO] Embeddings semánticos via Ollama ({embedding_model})")
        check_embedding_model(embedding_model)

    # ─── Construcción del índice ─────────────

    def build_index(self, chunks: List[Chunk]):
        """
        Vectoriza todos los chunks y construye el índice FAISS.
        """
        print(f"\n  [VECTORIZANDO] {len(chunks)} chunks con '{self.embedding_model}'...")
        texts = [chunk.text for chunk in chunks]

        t0         = time.time()
        embeddings = embed_batch(texts, model=self.embedding_model)
        elapsed    = time.time() - t0

        vector_dim = embeddings.shape[1]
        print(f"  [OK] Embeddings generados en {elapsed:.1f}s — dimensión: {vector_dim}")

        # Índice de producto interno (equivale a coseno con vectores normalizados)
        self.index = faiss.IndexFlatIP(vector_dim)
        self.index.add(embeddings)

        # Guardar metadatos en lista paralela al índice
        self.metadata = [
            {
                "chunk_id":   chunk.chunk_id,
                "source":     chunk.source,
                "text":       chunk.text,
                "start_char": chunk.start_char,
            }
            for chunk in chunks
        ]

        print(f"  [ÍNDICE] {self.index.ntotal} vectores almacenados en FAISS")

    # ─── Búsqueda semántica ──────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.25,
        source_filter: str = None,
    ) -> List[Tuple[dict, float]]:
        """
        Busca los `top_k` chunks más similares a la consulta.

        Args:
            query:         texto de búsqueda
            top_k:         número máximo de resultados
            min_score:     score mínimo para incluir un chunk
            source_filter: si se indica, solo devuelve chunks de ese archivo
                           (coincidencia parcial en el nombre del archivo)
        """
        if self.index is None:
            raise RuntimeError("El índice no está construido. Llama build_index() primero.")

        query_vec = np.array(
            [embed_single(query, model=self.embedding_model)],
            dtype="float32"
        )
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec /= norm

        # Pedir más resultados de los necesarios para poder filtrar por fuente
        fetch_k = top_k * 6 if source_filter else top_k
        scores, indices = self.index.search(query_vec, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if score < min_score:
                continue
            meta = self.metadata[idx]
            # Filtrar por documento si se especificó
            if source_filter and source_filter.lower() not in meta["source"].lower():
                continue
            results.append((meta, float(score)))
            if len(results) >= top_k:
                break

        return results

    def list_sources(self) -> List[str]:
        """Devuelve la lista de documentos únicos indexados."""
        return sorted({m["source"] for m in self.metadata})

    def search_mmr(
        self,
        query: str,
        top_k: int = 6,
        fetch_k: int = 20,
        diversity: float = 0.35,
        min_score: float = 0.25,
        source_filter: str = None,
    ) -> List[Tuple[dict, float]]:
        """
        Maximum Marginal Relevance: devuelve chunks relevantes pero diversos.
        Evita devolver múltiples chunks casi idénticos del mismo documento.

        diversity: 0.0 = solo relevancia, 1.0 = solo diversidad (0.3-0.4 es buen balance)
        """
        if self.index is None:
            raise RuntimeError("El índice no está construido.")

        query_vec = np.array(
            [embed_single(query, model=self.embedding_model)],
            dtype="float32"
        )
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec /= norm

        # Recuperar un pool amplio de candidatos
        pool_k = fetch_k * 4 if source_filter else fetch_k
        scores, indices = self.index.search(query_vec, pool_k)

        # Filtrar por score mínimo y por fuente si aplica
        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < min_score:
                continue
            meta = self.metadata[idx]
            if source_filter and source_filter.lower() not in meta["source"].lower():
                continue
            candidates.append((int(idx), meta, float(score)))

        if not candidates:
            return []

        # Obtener los vectores de los candidatos del índice FAISS
        candidate_vecs = np.array([
            self._get_vector(c[0])
            for c in candidates
        ], dtype="float32")

        # MMR iterativo
        selected  = []   # índices dentro de `candidates`
        remaining = list(range(len(candidates)))

        while len(selected) < top_k and remaining:
            if not selected:
                # Primer elemento: el más relevante
                best = max(remaining, key=lambda i: candidates[i][2])
            else:
                best_score = -999.0
                best = None
                for i in remaining:
                    relevance = candidates[i][2]
                    # Similitud coseno con cada ya seleccionado
                    sim_with_selected = max(
                        float(np.dot(candidate_vecs[i], candidate_vecs[s]))
                        for s in selected
                    )
                    mmr_score = (1 - diversity) * relevance - diversity * sim_with_selected
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best = i

            selected.append(best)
            remaining.remove(best)

        return [(candidates[i][1], candidates[i][2]) for i in selected]

    def _get_vector(self, idx: int) -> np.ndarray:
        """Extrae el vector en posición idx del índice FAISS."""
        vec = np.zeros(self.index.d, dtype="float32")
        self.index.reconstruct(int(idx), vec)  # cast explícito a int nativo
        return vec

    # ─── Persistencia ────────────────────────

    def save(self, directory: str):
        """Guarda el índice FAISS, metadatos y config del modelo en disco."""
        os.makedirs(directory, exist_ok=True)

        faiss.write_index(self.index, os.path.join(directory, "faiss.index"))

        with open(os.path.join(directory, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        # Guardar qué modelo se usó para embeddings (importante al cargar)
        config = {"embedding_model": self.embedding_model}
        with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f)

        print(f"  [GUARDADO] Base vectorial en '{directory}/'")

    def load(self, directory: str):
        """Carga el índice FAISS, metadatos y config desde disco."""
        index_path  = os.path.join(directory, "faiss.index")
        meta_path   = os.path.join(directory, "metadata.json")
        config_path = os.path.join(directory, "config.json")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"No se encontró índice en '{directory}'")

        self.index = faiss.read_index(index_path)

        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Restaurar el modelo de embeddings usado al indexar
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            self.embedding_model = config.get("embedding_model", EMBEDDING_MODEL)

        print(f"  [CARGADO] {self.index.ntotal} vectores desde '{directory}/'")
        print(f"  [MODELO]  {self.embedding_model}")

