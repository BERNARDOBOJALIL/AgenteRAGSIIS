"""
agent.py
Agente RAG universitario — interfaz de chat en terminal.

Uso:
    python agent.py
"""

import logging
import os
import re

import chromadb
from dotenv import load_dotenv
from firebase_sources import build_firebase_runtime_context
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq

load_dotenv()
logging.basicConfig(level=logging.WARNING)  # silenciar logs de LangChain en chat

COLLECTION_NAME = "asistente_universidad"
EMBEDDING_MODEL = "models/gemini-embedding-001"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_CLOUD_HOST = os.getenv("CHROMA_CLOUD_HOST", "api.trychroma.com")
CHROMA_CLOUD_PORT = int(os.getenv("CHROMA_CLOUD_PORT", "443"))

SYSTEM_PROMPT = """Eres un asistente virtual amable y preciso de la universidad.
Tu trabajo es responder preguntas de los alumnos usando UNICAMENTE la informacion
disponible en el contexto proporcionado.

Reglas obligatorias:
- Responde siempre en espanol, claro y conciso.
- Si no hay datos suficientes, responde exactamente: "No tengo esa información disponible."
- Nunca inventes datos, horarios, nombres o requisitos.
- Nunca menciones fuentes tecnicas, infraestructura o errores internos (Firebase, Chroma, API, JSON, permisos, logs, etc.).
- Nunca uses frases como "segun el contexto de Firebase", "segun el JSON" o equivalentes tecnicos.
- Para preguntas de salones, prioriza equipamiento y ubicacion aproximada (piso, zona y referencias cercanas).
- Si no hay equipamiento explicito pero existe equipamiento_inferido_conservador, puedes usarlo con lenguaje prudente ("es probable que", "podria contar con") y aclarar que es una estimacion.
- Solo comparte horario/calendario detallado cuando el usuario lo pida de forma explicita.
- Si piden como llegar, da indicaciones aproximadas y humanas; evita una ruta exacta paso a paso.
- Responde siempre en español.
- Se amable y usa un tono cercano pero profesional.
"""


def _compact_frontend_context(frontend_context: str | None, *, max_chars: int = 1500) -> str:
    """Compacta contexto opcional del frontend para no inflar tokens."""
    if not frontend_context:
        return ""

    compact = " ".join(str(frontend_context).split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[: max_chars - 3]}..."


def _sanitize_user_facing_response(text: str) -> str:
    """Elimina phrasing tecnico para mantener respuestas naturales al usuario."""
    if not text:
        return ""

    cleaned = str(text)
    substitutions: list[tuple[str, str]] = [
        (r"seg[uú]n\s+(el\s+)?contexto\s+de\s+firebase", "con la informacion disponible"),
        (r"de\s+acuerdo\s+con\s+firebase", "con la informacion disponible"),
        (r"seg[uú]n\s+(el\s+)?json", "con la informacion disponible"),
        (r"fired?base_operativo", ""),
        (r"datos_operativos_universidad", ""),
        (r"cruce_front_mapa_salones_json", ""),
        (r"referencias_textuales_sugeridas", ""),
        (r"contexto\s+firebase", "informacion disponible"),
        (r"contexto\s+chroma", "informacion disponible"),
        (r"contexto\s+siis\s+frontend", "informacion disponible"),
    ]

    for pattern, replacement in substitutions:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"(?:\s*\n\s*){3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()

    return cleaned or "No tengo esa información disponible."


def inicializar_vector_store() -> Chroma:
    """Carga el vector store desde Chroma Cloud."""
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    client = chromadb.CloudClient(
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE,
        api_key=CHROMA_API_KEY,
        cloud_host=CHROMA_CLOUD_HOST,
        cloud_port=CHROMA_CLOUD_PORT,
        enable_ssl=True,
    )
    return Chroma(
        client=client,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def generar_respuesta_rag(
    vector_store: Chroma,
    pregunta: str,
    historial: list[dict],
    frontend_context: str | None = None,
) -> str:
    """Responde con RAG hibrido: Chroma (estatico) + Firebase en vivo (dinamico)."""
    resultados = vector_store.similarity_search(pregunta, k=3)

    contexto_chroma = []
    for doc in resultados:
        fuente = doc.metadata.get("fuente", "desconocida")
        categoria = doc.metadata.get("categoria", "general")
        contexto_chroma.append(f"[Fuente: {fuente} | Categoría: {categoria}]\n{doc.page_content}")

    firebase_context = build_firebase_runtime_context(
        pregunta,
        frontend_context=frontend_context,
    )
    contexto_firebase = firebase_context.get("context_text", "")
    frontend_context_txt = _compact_frontend_context(frontend_context)
    solicitud_horario_detallado = bool(firebase_context.get("schedule_details_requested", False))

    if not contexto_chroma and not contexto_firebase:
        return "No tengo esa información disponible."

    historial_txt = ""
    if historial:
        lineas = []
        for turno in historial[-3:]:
            lineas.append(f"Usuario: {turno['pregunta']}")
            lineas.append(f"Asistente: {turno['respuesta']}")
        historial_txt = "\n".join(lineas)

    contexto_chroma_txt = "\n\n---\n\n".join(contexto_chroma) if contexto_chroma else "Sin contexto Chroma"
    contexto_frontend_txt = frontend_context_txt if frontend_context_txt else "Sin contexto frontend"

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "Instrucciones adicionales de prioridad:\n"
        "- Usa solo el contexto recuperado para responder.\n"
        "- Para SALONES y para USUARIOS/HORARIOS/CALENDARIOS, prioriza DATOS_OPERATIVOS_UNIVERSIDAD.\n"
        "- Usa cruce_front_mapa_salones_json para cruzar ruta/ubicacion del mapa con salones, equipamiento y responsables.\n"
        "- Si hay referencias_textuales_sugeridas, usalas como base y responde con indicaciones simples tipo: 'a la derecha', 'a la izquierda', 'al fondo'.\n"
        "- Para preguntas de ubicacion, responde con referencias aproximadas (inicio/medio/fondo, planta baja/alta, referencia de escalera si aplica).\n"
        "- Si Contexto SIIS frontend incluye route_guidance, usa esos pasos para mencionar izquierda/derecha con consistencia y no inventes giros.\n"
        "- Si equipamiento viene vacio y existe equipamiento_inferido_conservador, responde en tono conservador "
        "(por ejemplo: 'es probable que cuente con...') y no lo presentes como hecho confirmado.\n"
        "- Evita tecnicismos y evita explicar de donde se obtuvo la informacion.\n"
        "- Si solicitud_horario_detallado=false, no listes bloques completos de horario; solo resume disponibilidad general o confirma si existe horario.\n"
        "- Si solicitud_horario_detallado=true y hay datos, si puedes mostrar detalle de horario/calendario.\n"
        "- Si el contexto no alcanza, responde exactamente: 'No tengo esa información disponible.'\n\n"
        f"solicitud_horario_detallado={str(solicitud_horario_detallado).lower()}\n\n"
        f"Conversación previa:\n{historial_txt if historial_txt else 'Sin historial'}\n\n"
        f"Contexto Chroma:\n{contexto_chroma_txt}\n\n"
        f"Datos operativos (salones/personal):\n{contexto_firebase if contexto_firebase else 'Sin datos operativos'}\n\n"
        f"Contexto SIIS frontend:\n{contexto_frontend_txt}\n\n"
        f"Pregunta: {pregunta}"
    )

    llm = ChatGroq(model=GROQ_MODEL, temperature=0.1)
    respuesta = llm.invoke(prompt)
    contenido = getattr(respuesta, "content", "")
    if not isinstance(contenido, str) or not contenido.strip():
        return "No tengo esa información disponible."
    return _sanitize_user_facing_response(contenido)


def chat_loop(vector_store: Chroma):
    """Bucle de conversación en terminal."""
    print("\n" + "═" * 55)
    print("  🎓 Asistente Universitario  ")
    print("  Escribe 'salir' o 'exit' para terminar")
    print("═" * 55 + "\n")

    historial = []

    while True:
        try:
            pregunta = input("Tú: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n¡Hasta luego!")
            break

        if not pregunta:
            continue

        if pregunta.lower() in {"salir", "exit", "quit", "q"}:
            print("¡Hasta luego!")
            break

        try:
            respuesta = generar_respuesta_rag(vector_store, pregunta, historial)
        except Exception as e:
            respuesta = f"Ocurrió un error al procesar tu pregunta: {e}"

        print(f"\nAsistente: {respuesta}\n")
        historial.append({"pregunta": pregunta, "respuesta": respuesta})


def main():
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: Falta GOOGLE_API_KEY en el archivo .env")
        return
    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: Falta GROQ_API_KEY en el archivo .env")
        return
    if not CHROMA_API_KEY:
        print("ERROR: Falta CHROMA_API_KEY en el archivo .env")
        return
    if not CHROMA_TENANT:
        print("ERROR: Falta CHROMA_TENANT en el archivo .env")
        return
    if not CHROMA_DATABASE:
        print("ERROR: Falta CHROMA_DATABASE en el archivo .env")
        return

    print("Iniciando asistente universitario...")
    try:
        vector_store = inicializar_vector_store()
        total = vector_store._collection.count()
        print(f"Base de datos lista: {total} fragmentos indexados")
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        return

    logging.info(f"Modelo de chat configurado (Groq): {GROQ_MODEL}")
    chat_loop(vector_store)


if __name__ == "__main__":
    main()
