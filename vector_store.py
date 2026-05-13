"""
vector_store.py

Versión PRO con FAISS + Ollama + embeddings semánticos.

Mejoras:
- Búsqueda vectorial rápida con FAISS
- Similitud coseno real usando:
      IndexFlatIP + normalize_L2
- Compatible con RAGAS
- Compatible con tu RAG actual
- Sin fallback basura
- Mejor retrieval
- Escalable a miles de chunks

Requisito:
    pip install faiss-cpu

Modelo recomendado:
    ollama pull mxbai-embed-large
"""

import os
import json
import requests
import numpy as np
import faiss

from typing import List, Tuple
from document_loader import Chunk
# CONFIGURACIÓN
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "mxbai-embed-large"
# EMBEDDINGS
def get_embedding(
    text: str,
    model: str = EMBEDDING_MODEL
) -> List[float]:
    """
    Obtiene embedding usando Ollama.
    """
    if not text or not text.strip():
        return []
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={
                "model": model,
                "input": text.strip()
            },
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        if "embeddings" not in data or not data["embeddings"]:
            raise RuntimeError(
                f"Ollama no devolvió embeddings.\n"
                f"Verifica que '{model}' esté instalado."
            )
        return data["embeddings"][0]
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "No se pudo conectar con Ollama.\n"
            "Ejecuta primero:\n"
            "ollama serve"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            "Ollama tardó demasiado en responder."
        )
    except Exception as e:
        raise RuntimeError(
            f"Error obteniendo embedding:\n{str(e)}"
        )

# VECTOR STORE 
class VectorStore:
    """

    Usa:
        IndexFlatIP + normalize_L2

    Esto equivale a:
        cosine similarity real
    """

    def __init__(self):
        self.entries: List[dict] = []
        self.index = None
        self.dimension = None

    # BUILD INDEX
    def build_index(
        self,
        chunks: List[Chunk]
    ):
        """
        Vectoriza chunks y construye índice FAISS.
        """

        print(
            f"\nVectorizando {len(chunks)} chunks "
            f"con '{EMBEDDING_MODEL}'..."
        )

        self.entries = []
        vectors = []

        for i, chunk in enumerate(chunks, start=1):
            embedding = get_embedding(chunk.text)

            if not embedding:
                continue

            self.entries.append({
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "text": chunk.text
            })

            vectors.append(embedding)

            print(
                f"  {i}/{len(chunks)} vectorizados",
                end="\r"
            )

        if not vectors:
            raise RuntimeError(
                "No se pudieron generar embeddings."
            )

        vectors = np.array(
            vectors,
            dtype="float32"
        )

        self.dimension = vectors.shape[1]

        # NORMALIZACIÓN L2
        # necesaria para cosine similarity real
        faiss.normalize_L2(vectors)

        # IndexFlatIP = Inner Product
        # con normalize_L2 => cosine similarity
        self.index = faiss.IndexFlatIP(
            self.dimension
        )
        self.index.add(vectors)
        print(
            f"\n[OK] Índice FAISS construido con "
            f"{len(self.entries)} vectores"
        )
    # SEARCH
    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.71
    ) -> List[Tuple[dict, float]]:
        """
        Busca chunks más relevantes usando FAISS.
        """

        if self.index is None:
            raise RuntimeError(
                "El índice está vacío.\n"
                "Ejecuta build_index() primero."
            )

        query_embedding = get_embedding(query)

        if not query_embedding:
            return []

        query_vector = np.array(
            [query_embedding],
            dtype="float32"
        )

        # importante para cosine similarity
        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(
            query_vector,
            top_k
        )

        resultados = []

        for score, idx in zip(
            scores[0],
            indices[0]
        ):
            if idx == -1:
                continue

            if score < min_score:
                continue

            resultados.append(
                (
                    self.entries[idx],
                    float(score)
                )
            )

        if not resultados:
            print(
                "\n[AVISO] No se encontraron chunks "
                "con score suficiente."
            )
            return []

        return resultados
    # SAVE
    
    def save(
        self,
        directory: str
    ):
        """
        Guarda:
        - index.faiss
        - metadata.json
        """

        os.makedirs(
            directory,
            exist_ok=True
        )

        index_path = os.path.join(
            directory,
            "index.faiss"
        )

        metadata_path = os.path.join(
            directory,
            "metadata.json"
        )

        if self.index is None:
            raise RuntimeError(
                "No hay índice para guardar."
            )

        faiss.write_index(
            self.index,
            index_path
        )

        with open(
            metadata_path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                self.entries,
                f,
                ensure_ascii=False,
                indent=2
            )

        print(
            f"[GUARDADO]\n"
            f"FAISS: {index_path}\n"
            f"Metadata: {metadata_path}"
        )
    # LOAD
    def load(
        self,
        directory: str
    ):
        """
        Carga:
        - index.faiss
        - metadata.json
        """

        index_path = os.path.join(
            directory,
            "index.faiss"
        )

        metadata_path = os.path.join(
            directory,
            "metadata.json"
        )

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"No existe:\n{index_path}"
            )

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"No existe:\n{metadata_path}"
            )

        self.index = faiss.read_index(
            index_path
        )

        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as f:
            self.entries = json.load(f)

        self.dimension = self.index.d

        print(
            f"[CARGADO] Índice FAISS con "
            f"{len(self.entries)} vectores"
        )
