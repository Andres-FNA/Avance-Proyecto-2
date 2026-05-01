"""
main.py
Punto de entrada del sistema RAG con Ollama.

Formatos soportados: .txt, .pdf, .docx

Uso:
  python main.py --index                        # Indexar documentos de docs/
  python main.py --query "..." --model mistral  # Consulta directa
  python main.py --query "..."                  # Consulta con seleccion interactiva
  python main.py --interactive                  # Modo interactivo
  python main.py --interactive --model llama3   # Modo interactivo con modelo fijo
  python main.py --models                       # Listar modelos de chat disponibles

Dependencias extra para PDF y Word:
  pip install pypdf python-docx
"""

import os
import sys
import argparse

from document_loader import load_documents_from_folder, build_chunks_from_documents
from vector_store import VectorStore
from rag_engine import RAGEngine, select_model, get_available_models


# ─────────────────────────────────────────────
# Rutas del sistema
# ─────────────────────────────────────────────

DOCS_FOLDER  = "docs"       # Carpeta con documentos a indexar (.txt, .pdf, .docx)
INDEX_FOLDER = "vector_db"  # Carpeta donde se guarda el indice FAISS

CHUNK_SIZE = 1500  # Tamaño de chunk en caracteres (mayor = módulos completos)
OVERLAP    = 250   # Solapamiento entre chunks


# ─────────────────────────────────────────────
# Flujo: Indexacion
# ─────────────────────────────────────────────

def run_indexing():
    print("=" * 60)
    print("  FASE 1: CARGA DE DOCUMENTOS")
    print("  (formatos: .txt | .pdf | .docx)")
    print("=" * 60)
    documents = load_documents_from_folder(DOCS_FOLDER)

    if not documents:
        print(f"\n[ERROR] No se encontraron documentos (.txt, .pdf, .docx) en '{DOCS_FOLDER}/'")
        sys.exit(1)

    print(f"\n  Total: {len(documents)} documento(s) cargado(s)")

    print("\n" + "=" * 60)
    print("  FASE 2: CREACION DE CHUNKS")
    print("=" * 60)
    chunks = build_chunks_from_documents(
        documents,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
    )
    print(f"\n  Total: {len(chunks)} chunks generados")

    print("\n" + "=" * 60)
    print("  FASE 3: VECTORIZACION Y BASE DE DATOS VECTORIAL")
    print("=" * 60)
    store = VectorStore()
    store.build_index(chunks)

    print("\n" + "=" * 60)
    print("  FASE 4: PERSISTENCIA EN DISCO")
    print("=" * 60)
    store.save(INDEX_FOLDER)

    print("\n  Indexacion completada.")
    print(f"    Documentos: {len(documents)}")
    print(f"    Chunks:     {len(chunks)}")
    print(f"    Vectores:   {store.index.ntotal}")
    print(f"    Indice en:  {INDEX_FOLDER}/")


# ─────────────────────────────────────────────
# Flujo: Consulta
# ─────────────────────────────────────────────

def load_store() -> VectorStore:
    """Carga el VectorStore desde disco."""
    if not os.path.exists(INDEX_FOLDER):
        print(f"[ERROR] No hay indice en '{INDEX_FOLDER}/'. Ejecuta primero: python main.py --index")
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
        for meta, score in result["retrieved_chunks"]:
            print(f"  * {meta['source']} | chunk {meta['chunk_id']} | score {score:.4f}")


def run_interactive(model_arg: str = None):
    print("=" * 60)
    print("  SISTEMA RAG — MODO INTERACTIVO")
    print("  (escribe 'salir' para terminar)")
    print("=" * 60)

    store  = load_store()
    model  = select_model(model_arg)
    engine = RAGEngine(store, model=model)

    print(f"\n  Modelo activo: {model}")
    print("  Respondo preguntas sobre cualquier documento indexado.")
    print("  Si la información no está en los documentos, lo indicaré.")
    print("  Puedes escribir 'cambiar modelo' en cualquier momento.\n")

    while True:
        question = input("Pregunta: ").strip()
        if not question:
            continue

        if question.lower() in ("salir", "exit", "quit"):
            print("Hasta luego.")
            break

        if question.lower() in ("cambiar modelo", "cambiar", "modelo"):
            model  = select_model()
            engine = RAGEngine(store, model=model)
            print(f"  Modelo cambiado a: {model}\n")
            continue

        result = engine.query(question, verbose=False)
        print(f"\n--- RESPUESTA [{result['model']}] ---")
        print(result["answer"])

        if result["retrieved_chunks"]:
            print("\n--- FUENTES ---")
            for meta, score in result["retrieved_chunks"]:
                print(f"  * {meta['source']} | chunk {meta['chunk_id']} | score {score:.4f}")
        print()


def run_list_models():
    """Lista los modelos de chat disponibles en Ollama (excluye embeddings)."""
    print("=" * 60)
    print("  MODELOS DE CHAT DISPONIBLES EN OLLAMA")
    print("=" * 60)
    try:
        models = get_available_models()
        if not models:
            print("  No hay modelos de chat instalados.")
            print("  Instala uno con: ollama pull mistral")
        else:
            for i, name in enumerate(models, 1):
                print(f"  {i}. {name}")
    except RuntimeError as e:
        print(f"  [ERROR] {e}")


# ─────────────────────────────────────────────
# Punto de entrada
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sistema RAG local con Ollama + FAISS\n"
                    "Formatos soportados: .txt | .pdf | .docx",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--index",       action="store_true",
                       help="Indexar documentos en docs/ (.txt, .pdf, .docx)")
    group.add_argument("--query",       type=str, metavar="PREGUNTA",
                       help="Realizar una consulta directa")
    group.add_argument("--interactive", action="store_true",
                       help="Modo interactivo (multiples preguntas)")
    group.add_argument("--models",      action="store_true",
                       help="Listar modelos de chat disponibles en Ollama")

    parser.add_argument("--model", type=str, default=None, metavar="NOMBRE",
                        help="Nombre del modelo Ollama a usar (ej: mistral, llama3)\n"
                             "Si no se especifica, se muestra seleccion interactiva.")

    args = parser.parse_args()

    if args.index:
        run_indexing()
    elif args.query:
        run_single_query(args.query, model_arg=args.model)
    elif args.interactive:
        run_interactive(model_arg=args.model)
    elif args.models:
        run_list_models()


if __name__ == "__main__":
    main()