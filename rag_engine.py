"""
rag_engine.py

Motor RAG optimizado con Ollama + Few-Shot Prompting


1. Recupera chunks relevantes del VectorStore
2. Evita alucinaciones cuando no hay contexto suficiente
3. Prompt mucho más estricto para mejorar Answer Relevancy
4. Few-shot prompting para enseñar formato de respuesta
5. Menor divagación del LLM
6. Mejor desempeño en RAGAS
"""

import requests
from typing import List, Tuple

from vector_store import VectorStore


# ============================================================
# CONFIGURACIÓN
# ============================================================

OLLAMA_BASE_URL = "http://localhost:11434"

TOP_K = 5
MIN_SCORE = 0.65

def detect_source_filter(question: str) -> str | None:
    """
    Detecta automáticamente qué documento consultar
    según la pregunta del usuario.
    """

    q = question.lower()

    # Ingeniería de Software
    if (
        "ingeniería de software" in q
        or "ingenieria de software" in q
        or "software i" in q
    ):
        return "Ingeniería de Software"

    # Redes de Comunicación
    if (
        "redes de comunicación I" in q
        or "redes de comunicacion I" in q
        or "redes de comunicación I" in q
        or "Redes de Comunicacion I" in q
    ):
        return "Redes de Comunicación i"

    return None
# GESTIÓN DE MODELOS OLLAMA

def get_available_models() -> List[str]:
    """
    Devuelve los modelos instalados en Ollama.
    """

    try:
        response = requests.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=5
        )

        response.raise_for_status()

        return [
            model["name"]
            for model in response.json().get(
                "models",
                []
            )
        ]

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "No se pudo conectar con Ollama.\n"
            "Ejecuta primero:\n"
            "ollama serve"
        )


def select_model(model_name: str = None) -> str:
    """
    Selecciona modelo automáticamente o manualmente.
    """

    models = get_available_models()

    if not models:
        raise RuntimeError(
            "No hay modelos instalados.\n"
            "Instala uno con:\n"
            "ollama pull mistral"
        )

    if model_name:
        match = next(
            (
                model
                for model in models
                if model_name in model
            ),
            None
        )

        if match:
            print(f"[OLLAMA] Usando modelo: {match}")
            return match

    print("\nModelos disponibles:")

    for i, name in enumerate(models, 1):
        print(f"{i}. {name}")

    while True:
        try:
            choice = int(
                input(
                    f"\nSelecciona modelo [1-{len(models)}]: "
                )
            ) - 1

            if 0 <= choice < len(models):
                selected = models[choice]
                print(
                    f"[OLLAMA] Modelo seleccionado: {selected}"
                )
                return selected

        except ValueError:
            pass

        print("Selección inválida.")

# FEW-SHOT EXAMPLES


def get_few_shot_examples() -> str:
    """
    Ejemplos para enseñar al modelo
    cómo responder exactamente.
    """

    return """
EJEMPLOS DE RESPUESTA CORRECTA

Ejemplo 1

Pregunta:
¿Cuántos créditos tiene el curso?

Respuesta:
El curso tiene 3 créditos.

--------------------------------------------------

Ejemplo 2

Pregunta:
¿Qué ruta formativa tiene la asignatura de Redes de comunicacion I?

Respuesta:
La ruta formativa es Infraestructura.

--------------------------------------------------

Ejemplo 3

Pregunta:
¿Cuál es el objetivo de la asignatura?

Respuesta:
El objetivo de la asignatura es presentar metodologías, procesos y herramientas.

--------------------------------------------------

Ejemplo 4

Pregunta:
¿Como me gano un oscar?

Respuesta:
No tengo información sobre ese tema en los documentos disponibles.

--------------------------------------------------
""".strip()


# ============================================================
# PROMPT
# ============================================================

def build_prompt(
    question: str,
    retrieved_chunks: List[Tuple[dict, float]]
) -> str:
    """
    Construye prompt estricto optimizado
    para mejorar Answer Relevancy.
    """

    context_parts = []

    for i, (entry, score) in enumerate(
        retrieved_chunks,
        1
    ):
        context_parts.append(
            f"[Fragmento {i} "
            f"| Fuente: {entry['source']} "
            f"| Score: {score:.3f}]\n"
            f"{entry['text']}"
        )

    context = "\n\n---\n\n".join(
        context_parts
    )

    few_shot = get_few_shot_examples()

    return f"""
Eres un asistente experto en responder preguntas
usando EXCLUSIVAMENTE los fragmentos proporcionados.

Tu objetivo principal es responder de forma:
DIRECTA + EXACTA + BREVE + SIN ALUCINAR.

==================================================
REGLAS OBLIGATORIAS
==================================================
Prioriza definiciones explícitas como:

- objetivo
- finalidad
- propósito
- competencias
- resultados de aprendizaje

1. Usa SOLO información presente en los fragmentos y responde en maximo 2 oraciones.

2. NO inventes información.

3. NO uses conocimiento externo.

4. Si la respuesta existe claramente,
   respóndela directamente.

5. Si preguntan por:

   - créditos
   - ruta formativa
   - número
   - nombre
   - objetivo
   - metodología

   responde SOLO ese dato.

6. Usa las mismas palabras
   presentes en los fragmentos
   cuando sea posible.

7. NO reformules demasiado.

8. Si la respuesta es corta,
   entrégala corta.

9. Si no existe evidencia suficiente,
   responde EXACTAMENTE:

   "No tengo información sobre ese tema
   en los documentos disponibles."

10. Nunca respondas:

   - Según los fragmentos
   - La respuesta es
   - Fragmento 1

11. Nunca seas excesivamente conservador.
    Si la evidencia existe, úsala.
12. Si la pregunta pide un dato puntual
como ruta formativa, nombre, créditos,
correo o valor, responde SOLO con ese dato,
sin explicciones adicionales.
13. Nunca listes contenido técnico
como:

- capas de red
- protocolos
- topologías
- redes LAN/WAN
- infraestructura
- temas de clase

si la pregunta es sobre:

- metodologías didácticas
- evaluación
- objetivo
- finalidad
- ruta formativa

Responde SOLO información académica
de la asignatura.

EJEMPLOS

{few_shot}

FRAGMENTOS

{context}
=
PREGUNTA

{question}

RESPUESTA
""".strip()

# OLLAMA CALL
def call_ollama(
    prompt: str,
    model: str
) -> str:
    """
    Consulta al modelo.
    """

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 0.1,
            "repeat_penalty": 1.0,
            "num_predict": 500,
        }
    }

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=120
        )

        response.raise_for_status()

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Ollama no está disponible.\n"
            "Ejecuta:\n"
            "ollama serve"
        )

    except requests.exceptions.Timeout:
        raise RuntimeError(
            "El modelo tardó demasiado.\n"
            "Prueba con uno más pequeño."
        )

    return response.json().get(
        "response",
        ""
    ).strip()

# RAG ENGINE

class RAGEngine:
    """
    Orquesta el flujo RAG completo.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        model: str,
        top_k: int = TOP_K
    ):
        self.vector_store = vector_store
        self.model = model
        self.top_k = top_k

    def query(
        self,
        question: str,
        source_filter: str = None
    ) -> dict:
        print(
            f"\n[RAG] Buscando contexto para:\n"
            f"'{question}'"
        )

        # 1. SOURCE FILTER AUTOMÁTICO
        if source_filter is None:
            source_filter = detect_source_filter(question)
            if source_filter:
                print(f"[RAG] Source filter automático detectado: {source_filter}")

        # 2. RECUPERACIÓN
        retrieved = self.vector_store.search(
            question,
            top_k=self.top_k,
            min_score=MIN_SCORE
        )

        # 3. FILTRO POR DOCUMENTO
        if source_filter:
            filtered = [
                (entry, score)
                for entry, score in retrieved
                if source_filter.lower() in entry["source"].lower()
            ]

            if filtered:
                retrieved = filtered
                print(f"[RAG] Filtro aplicado: {source_filter}")
            else:
                print(f"[RAG] No hubo matches con filtro '{source_filter}', usando retrieval general.")

        print(f"[RAG] {len(retrieved)} chunks recuperados:")
        for entry, score in retrieved:
            print(f"  * [{score:.3f}] {entry['source']} — chunk {entry['chunk_id']}")

        # 4. SIN CONTEXTO → NO ALUCINAR
        if not retrieved:
            answer = "No tengo información sobre ese tema en los documentos disponibles."
            return {
                "question": question,
                "model": self.model,
                "retrieved_chunks": [],
                "answer": answer,
            }

        # 5. CONSTRUCCIÓN DEL PROMPT Y LLAMADA AL LLM
        prompt = build_prompt(question, retrieved)
        
        print(f"[RAG] Consultando modelo '{self.model}'...")
        answer = call_ollama(prompt, self.model)

        return {
            "question": question,
            "model": self.model,
            "retrieved_chunks": retrieved,
            "answer": answer,
        }
