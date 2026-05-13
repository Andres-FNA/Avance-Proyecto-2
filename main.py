"""
main.py
Punto de entrada del sistema RAG básico con Ollama.

Formatos soportados: .txt

Uso:
  python main.py --index                        # Indexar documentos de docs/
  python main.py --query "tu pregunta aqui"     # Consulta directa
  python main.py --interactive                  # Modo interactivo
"""

import os
import sys
import argparse

from document_loader import load_documents_from_folder, build_chunks_from_documents
from vector_store import VectorStore
from rag_engine import RAGEngine, select_model


# ─────────────────────────────────────────────
# Configuración de rutas y parámetros
# ─────────────────────────────────────────────

DOCS_FOLDER  = "docs"        # Carpeta con los documentos .txt
INDEX_FOLDER = "vector_db"   # Carpeta donde se guarda la base vectorial

CHUNK_SIZE = 600  # Tamaño de chunk en caracteres — separa mejor secciones cortas
OVERLAP    =150  # Solapamiento entre chunks

# Flujo 1: Indexación

def run_indexing():
    print("=" * 60)
    print("  FASE 1: CARGA DE DOCUMENTOS")
    print("=" * 60)
    documents = load_documents_from_folder(DOCS_FOLDER)

    if not documents:
        print(f"\n[ERROR] No se encontraron archivos .txt en '{DOCS_FOLDER}/'")
        sys.exit(1)

    print(f"\n  Total: {len(documents)} documento(s) cargado(s)")

    print("\n" + "=" * 60)
    print("  FASE 2: CREACIÓN DE CHUNKS")
    print("=" * 60)
    chunks = build_chunks_from_documents(
        documents,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
    )
    print(f"\n  Total: {len(chunks)} chunks generados")

    print("\n" + "=" * 60)
    print("  FASE 3: VECTORIZACIÓN Y BASE DE DATOS VECTORIAL")
    print("=" * 60)
    store = VectorStore()
    store.build_index(chunks)

    print("\n" + "=" * 60)
    print("  FASE 4: GUARDADO EN DISCO")
    print("=" * 60)
    store.save(INDEX_FOLDER)

    print("\n  Indexación completada.")
    print(f"    Documentos : {len(documents)}")
    print(f"    Chunks     : {len(chunks)}")
    print(f"    Índice en  : {INDEX_FOLDER}/")

# Flujo 2: Consulta

def load_store() -> VectorStore:
    """Carga el VectorStore desde disco."""
    if not os.path.exists(INDEX_FOLDER):
        print(f"[ERROR] No hay índice en '{INDEX_FOLDER}/'. Ejecuta primero: python main.py --index")
        sys.exit(1)
    store = VectorStore()
    store.load(INDEX_FOLDER)
    return store


def run_single_query(question: str, model_arg: str = None):
    print("=" * 60)
    print("  SISTEMA RAG — CONSULTA")
    print("=" * 60)

    store  = load_store()
    model  = select_model(model_arg)
    engine = RAGEngine(store, model=model)
    result = engine.query(question)

    print("\n" + "=" * 60)
    print(f"  RESPUESTA  [{result['model']}]")
    print("=" * 60)
    print(result["answer"])

    if result["retrieved_chunks"]:
        print("\n" + "-" * 60)
        print("  FUENTES UTILIZADAS")
        print("-" * 60)
        for entry, score in result["retrieved_chunks"]:
            print(f"  * {entry['source']} | chunk {entry['chunk_id']} | score {score:.4f}")


def run_interactive(model_arg: str = None):
    print("=" * 60)
    print("  SISTEMA RAG — MODO INTERACTIVO")
    print("  (escribe 'salir' para terminar)")
    print("=" * 60)

    store  = load_store()
    model  = select_model(model_arg)
    engine = RAGEngine(store, model=model)

    print(f"\n  Modelo activo: {model}")
    print("  Respondo preguntas sobre los documentos indexados.\n")

    while True:
        question = input("Pregunta: ").strip()
        if not question:
            continue

        if question.lower() in ("salir", "exit", "quit"):
            print("Hasta luego.")
            break

        result = engine.query(question)
        print(f"\n--- RESPUESTA [{result['model']}] ---")
        print(result["answer"])

        if result["retrieved_chunks"]:
            print("\n--- FUENTES ---")
            for entry, score in result["retrieved_chunks"]:
                print(f"  * {entry['source']} | chunk {entry['chunk_id']} | score {score:.4f}")
        print()

# Punto de entrada


def main():
    parser = argparse.ArgumentParser(
        description="Sistema RAG local con Ollama\nFormatos soportados: .txt",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--index",       action="store_true",
                       help="Indexar documentos en docs/")
    group.add_argument("--query",       type=str, metavar="PREGUNTA",
                       help="Realizar una consulta directa")
    group.add_argument("--interactive", action="store_true",
                       help="Modo interactivo (múltiples preguntas)")

    parser.add_argument("--model", type=str, default=None, metavar="NOMBRE",
                        help="Modelo Ollama a usar (ej: mistral, llama3)")

    args = parser.parse_args()

    if args.index:
        run_indexing()
    elif args.query:
        run_single_query(args.query, model_arg=args.model)
    elif args.interactive:
        run_interactive(model_arg=args.model)


if __name__ == "__main__":
    main()
