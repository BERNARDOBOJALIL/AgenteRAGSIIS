"""
ingest.py
Alimenta ChromaDB solo con scraping de URLs estáticas.

Uso:
    python ingest.py
    python ingest.py --reset      # Borra la BD y la recrea desde cero
"""

import argparse
import logging
import os
import shutil

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from scraper import pipeline_completo

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN — Edita aquí tus fuentes
# ══════════════════════════════════════════════════════════════════════════════

URLS_ESTATICAS = [
    "https://www.iberopuebla.mx/IDIT/fabricacion-digital/tecnologias",
    "https://www.iberopuebla.mx/IDIT/emprende-IBERO",
    "https://www.iberopuebla.mx/IDIT/comunidad-IBERO-Puebla",
    "https://www.iberopuebla.mx/IDIT/laboratorios-multidisciplinares",
    "https://www.iberopuebla.mx/IDIT/vinculacion",
    "https://www.iberopuebla.mx/IDIT/soluciones-empresariales",
    "https://www.iberopuebla.mx/IDIT/espacios-y-maquinaria",
    "https://www.iberopuebla.mx/IDIT/IDIT-academy",
    "https://www.iberopuebla.mx/IDIT/economia-social",
    "https://www.iberopuebla.mx/IDIT/innovacion"
]

# Límites para ahorrar requests de embeddings (Gemini)
# Para anexar pendientes: inicia después de las URLs ya indexadas (0-based)
INDICE_INICIO_URLS = 9
MAX_URLS_POR_INGESTA = 3
MAX_FRAGMENTOS_POR_INGESTA = 120

# ══════════════════════════════════════════════════════════════════════════════

DB_PATH = "./chroma_db"
COLLECTION_NAME = "asistente_universidad"
EMBEDDING_MODEL = "models/gemini-embedding-001"


def main(reset: bool = False):
    # Verificar API key
    if not os.getenv("GOOGLE_API_KEY"):
        log.error("Falta GOOGLE_API_KEY en el archivo .env")
        return

    # Opcional: borrar la BD existente
    if reset and os.path.exists(DB_PATH):
        log.info("Borrando base de datos existente...")
        shutil.rmtree(DB_PATH)

    # 1. Recopilar y fragmentar documentos solo de URLs estáticas
    log.info("=== Iniciando recopilación de documentos ===")
    urls_pendientes = URLS_ESTATICAS[INDICE_INICIO_URLS:]
    urls_a_procesar = (
        urls_pendientes[:MAX_URLS_POR_INGESTA]
        if MAX_URLS_POR_INGESTA
        else urls_pendientes
    )
    log.info(f"Procesando {len(urls_a_procesar)} URL(s) estáticas")

    fragmentos = pipeline_completo(
        urls_estaticas=urls_a_procesar or None,
    )

    if not fragmentos:
        log.warning("No hay documentos para indexar. Revisa tus fuentes en ingest.py")
        return

    if MAX_FRAGMENTOS_POR_INGESTA and len(fragmentos) > MAX_FRAGMENTOS_POR_INGESTA:
        log.info(
            f"Aplicando tope de fragmentos: {MAX_FRAGMENTOS_POR_INGESTA} "
            f"(de {len(fragmentos)} generados)"
        )
        fragmentos = fragmentos[:MAX_FRAGMENTOS_POR_INGESTA]

    # 2. Crear embeddings
    log.info("Inicializando modelo de embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    # 3. Insertar en ChromaDB
    log.info(f"Indexando {len(fragmentos)} fragmentos en ChromaDB...")

    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        # BD existente: agregar sin duplicar
        vector_store = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
        vector_store.add_documents(fragmentos)
        total = vector_store._collection.count()
        log.info(f"Documentos agregados. Total en BD: {total}")
    else:
        # BD nueva
        vector_store = Chroma.from_documents(
            documents=fragmentos,
            embedding=embeddings,
            persist_directory=DB_PATH,
            collection_name=COLLECTION_NAME,
        )
        log.info(f"Base de datos creada con {len(fragmentos)} fragmentos.")

    log.info("=== Ingestión completada ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alimenta ChromaDB con documentos universitarios.")
    parser.add_argument("--reset", action="store_true", help="Borra y recrea la base de datos.")
    args = parser.parse_args()
    main(reset=args.reset)
