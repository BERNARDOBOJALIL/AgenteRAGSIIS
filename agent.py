"""
agent.py
Agente RAG universitario — interfaz de chat en terminal.

Uso:
    python agent.py
"""

import logging
import os

import chromadb
from dotenv import load_dotenv
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
Tu trabajo es responder preguntas de los alumnos usando ÚNICAMENTE la información
disponible en la base de datos universitaria. 

Reglas:
- Si encuentras la información, respóndela de forma clara y concisa.
- Si NO encuentras la información, dilo honestamente: "No tengo esa información disponible."
- Nunca inventes datos, horarios, nombres o requisitos.
- Responde siempre en español.
- Sé amable y usa un tono cercano pero profesional.
"""


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


def generar_respuesta_rag(vector_store: Chroma, pregunta: str, historial: list[dict]) -> str:
    """Responde con RAG directo (sin tool-calling) para mayor estabilidad."""
    resultados = vector_store.similarity_search(pregunta, k=4)
    if not resultados:
        return "No tengo esa información disponible."

    contexto = []
    for doc in resultados:
        fuente = doc.metadata.get("fuente", "desconocida")
        categoria = doc.metadata.get("categoria", "general")
        contexto.append(f"[Fuente: {fuente} | Categoría: {categoria}]\n{doc.page_content}")

    historial_txt = ""
    if historial:
        lineas = []
        for turno in historial[-3:]:
            lineas.append(f"Usuario: {turno['pregunta']}")
            lineas.append(f"Asistente: {turno['respuesta']}")
        historial_txt = "\n".join(lineas)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "Instrucciones adicionales:\n"
        "- Usa solo el contexto recuperado para responder.\n"
        "- Si el contexto no alcanza, responde exactamente: 'No tengo esa información disponible.'\n\n"
        f"Conversación previa:\n{historial_txt if historial_txt else 'Sin historial'}\n\n"
        f"Contexto:\n{'\n\n---\n\n'.join(contexto)}\n\n"
        f"Pregunta: {pregunta}"
    )

    llm = ChatGroq(model=GROQ_MODEL, temperature=0.1)
    respuesta = llm.invoke(prompt)
    contenido = getattr(respuesta, "content", "")
    return contenido if isinstance(contenido, str) and contenido.strip() else "No tengo esa información disponible."


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
