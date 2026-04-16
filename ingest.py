"""
ingest.py
Alimenta ChromaDB con fuentes de web, PDF y Firestore,
con formato Markdown unificado para todas las entradas.

Uso:
    python ingest.py
    python ingest.py --reset      # Borra la BD y la recrea desde cero
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

import chromadb
from langchain_core.documents import Document

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from docx_sources import load_docx_documents
from firebase_sources import fetch_firebase_documents
from markdown_unifier import export_documents_as_markdown, normalize_documents_to_markdown
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

# Limites para controlar costo de embeddings (Gemini)
# Para anexar pendientes: inicia despues de las URLs ya indexadas (0-based)
INDICE_INICIO_URLS = 9
MAX_URLS_POR_INGESTA = 3
MAX_DOCUMENTOS_POR_INGESTA = 200

PDFS_DIR_DEFAULT = "./pdfs"
DOCX_DIR_DEFAULT = "./"
EXPORT_MD_DIR_DEFAULT = "./exports/markdown_ingesta"

# ══════════════════════════════════════════════════════════════════════════════

DB_PATH = "./chroma_db"
COLLECTION_NAME = "asistente_universidad"
EMBEDDING_MODEL = "models/gemini-embedding-001"

CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_CLOUD_HOST = os.getenv("CHROMA_CLOUD_HOST", "api.trychroma.com")
CHROMA_CLOUD_PORT = int(os.getenv("CHROMA_CLOUD_PORT", "443"))


def _stable_document_id(document: Document) -> str:
    """Genera un ID determinista por contenido+metadatos para evitar duplicados."""
    metadata = document.metadata or {}
    seed = json.dumps(metadata, sort_keys=True, ensure_ascii=True)
    payload = f"{seed}\n{document.page_content}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _collect_web_and_pdf_documents(*, skip_web: bool, pdf_dir: str | None) -> list[Document]:
    """Recopila fragmentos desde URLs estaticas y carpeta de PDFs."""
    urls_a_procesar: list[str] = []
    if not skip_web:
        urls_pendientes = URLS_ESTATICAS[INDICE_INICIO_URLS:]
        urls_a_procesar = (
            urls_pendientes[:MAX_URLS_POR_INGESTA]
            if MAX_URLS_POR_INGESTA
            else urls_pendientes
        )

    if pdf_dir:
        pdf_path = Path(pdf_dir)
        if not pdf_path.exists():
            log.warning("La carpeta de PDFs '%s' no existe. Se omite carga de PDFs.", pdf_dir)
            pdf_dir = None

    if not urls_a_procesar and not pdf_dir:
        return []

    log.info(
        "Recopilando fuentes web/PDF: %s URL(s), PDF dir=%s",
        len(urls_a_procesar),
        pdf_dir or "sin_pdfs",
    )
    return pipeline_completo(
        urls_estaticas=urls_a_procesar or None,
        carpeta_pdfs=pdf_dir,
    )


def _collect_docx_documents(*, docx_dir: str | None) -> list[Document]:
    """Recopila documentos DOCX desde la carpeta indicada."""
    if not docx_dir:
        return []

    docx_path = Path(docx_dir)
    if not docx_path.exists():
        log.warning("La carpeta DOCX '%s' no existe. Se omite carga de DOCX.", docx_dir)
        return []

    return load_docx_documents(docx_dir)


def main(
    reset: bool = False,
    skip_web: bool = False,
    skip_firebase: bool = True,
    skip_pdf: bool = False,
    skip_docx: bool = False,
    pdf_dir: str = PDFS_DIR_DEFAULT,
    docx_dir: str = DOCX_DIR_DEFAULT,
    export_md_dir: str = EXPORT_MD_DIR_DEFAULT,
    no_export_md: bool = False,
    only_export_md: bool = False,
    cloud: bool = False,
):
    # Verificar API key
    if not os.getenv("GOOGLE_API_KEY"):
        log.error("Falta GOOGLE_API_KEY en el archivo .env")
        return

    # Opcional: borrar la BD existente
    if reset and os.path.exists(DB_PATH):
        log.info("Borrando base de datos existente...")
        shutil.rmtree(DB_PATH)

    log.info("=== Iniciando recopilacion de documentos ===")
    all_documents: list[Document] = []

    pdf_dir_to_use = None if skip_pdf else pdf_dir
    docx_dir_to_use = None if skip_docx else docx_dir

    # 1) Web + PDF -> limpiar y convertir a Markdown unificado
    web_pdf_docs = _collect_web_and_pdf_documents(skip_web=skip_web, pdf_dir=pdf_dir_to_use)
    if web_pdf_docs:
        web_pdf_md = normalize_documents_to_markdown(web_pdf_docs, filter_useful=True, preserve_full_text=False)
        all_documents.extend(web_pdf_md)
        log.info("Web/PDF normalizados a Markdown: %s", len(web_pdf_md))

    # 2) Firestore salones + usuarios(profesores)
    if not skip_firebase:
        firestore_docs = fetch_firebase_documents()
        all_documents.extend(firestore_docs)
        log.info("Firestore convertidos a Markdown: %s", len(firestore_docs))

    # 3) DOCX locales
    docx_docs = _collect_docx_documents(docx_dir=docx_dir_to_use)
    if docx_docs:
        docx_md = normalize_documents_to_markdown(docx_docs, filter_useful=False, preserve_full_text=True)
        all_documents.extend(docx_md)
        log.info("DOCX normalizados a Markdown: %s", len(docx_md))

    if not all_documents:
        log.warning("No hay documentos para indexar. Revisa fuentes y variables en .env")
        return

    if MAX_DOCUMENTOS_POR_INGESTA and len(all_documents) > MAX_DOCUMENTOS_POR_INGESTA:
        log.info(
            "Aplicando tope de documentos: %s (de %s generados)",
            MAX_DOCUMENTOS_POR_INGESTA,
            len(all_documents),
        )
        all_documents = all_documents[:MAX_DOCUMENTOS_POR_INGESTA]

    if not no_export_md and export_md_dir:
        exported = export_documents_as_markdown(all_documents, output_dir=export_md_dir)
        log.info("Documentos Markdown exportados: %s en %s", exported, export_md_dir)

    if only_export_md:
        log.info("Modo solo-export-md activo: no se insertaran embeddings en ChromaDB.")
        return

    document_ids = [_stable_document_id(doc) for doc in all_documents]

    # 3. Crear embeddings
    log.info("Inicializando modelo de embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    # 4. Insertar en ChromaDB
    log.info("Indexando %s documentos Markdown en ChromaDB...", len(all_documents))

    if cloud:
        if not CHROMA_API_KEY or not CHROMA_TENANT or not CHROMA_DATABASE:
            log.error("Faltan CHROMA_API_KEY, CHROMA_TENANT o CHROMA_DATABASE para usar --cloud")
            return

        client = chromadb.CloudClient(
            tenant=CHROMA_TENANT,
            database=CHROMA_DATABASE,
            api_key=CHROMA_API_KEY,
            cloud_host=CHROMA_CLOUD_HOST,
            cloud_port=CHROMA_CLOUD_PORT,
            enable_ssl=True,
        )
        vector_store = Chroma(
            client=client,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
        vector_store.add_documents(all_documents, ids=document_ids)
        total = vector_store._collection.count()
        log.info("Documentos agregados en Chroma Cloud. Total en coleccion: %s", total)
    elif os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        # BD existente: agregar sin duplicar
        vector_store = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
        vector_store.add_documents(all_documents, ids=document_ids)
        total = vector_store._collection.count()
        log.info(f"Documentos agregados. Total en BD: {total}")
    else:
        # BD nueva
        vector_store = Chroma.from_documents(
            documents=all_documents,
            embedding=embeddings,
            persist_directory=DB_PATH,
            collection_name=COLLECTION_NAME,
            ids=document_ids,
        )
        log.info("Base de datos creada con %s documentos.", len(all_documents))

    log.info("=== Ingestion completada ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alimenta ChromaDB con documentos IDIT normalizados a Markdown.")
    parser.add_argument("--reset", action="store_true", help="Borra y recrea la base de datos.")
    parser.add_argument("--skip-web", action="store_true", help="Omite scraping web.")
    parser.add_argument("--skip-firebase", action="store_true", help="Omite extraccion desde Firestore para ingesta estatica.")
    parser.add_argument(
        "--include-firebase-static",
        action="store_false",
        dest="skip_firebase",
        help="Incluye Firestore en la ingesta estatica (no recomendado si usas Firebase en vivo en agent.py).",
    )
    parser.add_argument("--skip-pdf", action="store_true", help="Omite carga de PDFs.")
    parser.add_argument("--skip-docx", action="store_true", help="Omite carga de DOCX.")
    parser.add_argument("--pdf-dir", default=PDFS_DIR_DEFAULT, help="Carpeta con PDFs para ingesta.")
    parser.add_argument("--docx-dir", default=DOCX_DIR_DEFAULT, help="Carpeta con DOCX para ingesta.")
    parser.add_argument("--cloud", action="store_true", help="Sube documentos a Chroma Cloud en vez de Chroma local.")
    parser.add_argument(
        "--export-md-dir",
        default=EXPORT_MD_DIR_DEFAULT,
        help="Carpeta destino para exportar documentos Markdown normalizados.",
    )
    parser.add_argument("--no-export-md", action="store_true", help="No exportar archivos Markdown.")
    parser.add_argument(
        "--only-export-md",
        action="store_true",
        help="Solo genera/actualiza archivos Markdown normalizados sin subir a ChromaDB.",
    )
    parser.set_defaults(skip_firebase=True)
    args = parser.parse_args()
    main(
        reset=args.reset,
        skip_web=args.skip_web,
        skip_firebase=args.skip_firebase,
        skip_pdf=args.skip_pdf,
        skip_docx=args.skip_docx,
        pdf_dir=args.pdf_dir,
        docx_dir=args.docx_dir,
        export_md_dir=args.export_md_dir,
        no_export_md=args.no_export_md,
        only_export_md=args.only_export_md,
        cloud=args.cloud,
    )
