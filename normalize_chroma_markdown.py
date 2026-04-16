"""
normalize_chroma_markdown.py
Normaliza documentos existentes en ChromaDB a formato Markdown unificado.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any

import chromadb
from dotenv import load_dotenv
from langchain_core.documents import Document

from markdown_unifier import UNIFIED_FORMAT_VERSION, build_unified_markdown, export_documents_as_markdown

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME = "asistente_universidad"
DEFAULT_LOCAL_DB_PATH = "./chroma_db"
DEFAULT_EXPORT_DIR = "./exports/chroma_normalizado"

CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_CLOUD_HOST = os.getenv("CHROMA_CLOUD_HOST", "api.trychroma.com")
CHROMA_CLOUD_PORT = int(os.getenv("CHROMA_CLOUD_PORT", "443"))


def _looks_like_web_source(metadata: dict[str, Any]) -> bool:
    fuente = str(metadata.get("fuente", "")).lower()
    categoria = str(metadata.get("categoria", "")).lower()
    url = str(metadata.get("url") or metadata.get("source_url") or "").lower()

    web_markers = ("web", "scrap", "crawl", "http", "iberopuebla")
    return any(marker in fuente for marker in web_markers) or any(marker in categoria for marker in web_markers) or url.startswith("http")


def _connect_collection(*, use_cloud: bool, collection_name: str, local_path: str):
    if use_cloud:
        if not CHROMA_API_KEY or not CHROMA_TENANT or not CHROMA_DATABASE:
            raise RuntimeError("Faltan CHROMA_API_KEY, CHROMA_TENANT o CHROMA_DATABASE para usar Cloud.")

        client = chromadb.CloudClient(
            tenant=CHROMA_TENANT,
            database=CHROMA_DATABASE,
            api_key=CHROMA_API_KEY,
            cloud_host=CHROMA_CLOUD_HOST,
            cloud_port=CHROMA_CLOUD_PORT,
            enable_ssl=True,
        )
    else:
        client = chromadb.PersistentClient(path=local_path)

    return client.get_or_create_collection(name=collection_name)


def normalize_collection(
    *,
    use_cloud: bool,
    collection_name: str,
    local_path: str,
    batch_size: int,
    only_web: bool,
    dry_run: bool,
    force: bool,
    export_md_dir: str | None,
):
    collection = _connect_collection(use_cloud=use_cloud, collection_name=collection_name, local_path=local_path)
    total = collection.count()

    if total == 0:
        log.warning("La coleccion esta vacia: %s", collection_name)
        return

    log.info("Coleccion '%s': %s documentos.", collection_name, total)

    offset = 0
    normalized_count = 0
    skipped_count = 0
    to_export: list[Document] = []

    while offset < total:
        payload = collection.get(limit=batch_size, offset=offset, include=["documents", "metadatas"])

        ids = payload.get("ids", [])
        documents = payload.get("documents", [])
        metadatas = payload.get("metadatas", [])

        if not ids:
            break

        updated_ids: list[str] = []
        updated_docs: list[str] = []
        updated_metas: list[dict[str, Any]] = []

        for idx, doc_id in enumerate(ids):
            metadata = dict(metadatas[idx] or {}) if idx < len(metadatas) else {}
            original_text = documents[idx] if idx < len(documents) else ""

            already_unified = metadata.get("formato") == UNIFIED_FORMAT_VERSION
            if already_unified and not force:
                skipped_count += 1
                continue

            if only_web and not _looks_like_web_source(metadata):
                skipped_count += 1
                continue

            title = (
                metadata.get("titulo")
                or metadata.get("title")
                or metadata.get("nombre")
                or metadata.get("name")
                or metadata.get("url")
                or f"documento_{doc_id}"
            )

            markdown, tipo_consulta = build_unified_markdown(
                title=str(title),
                main_text=str(original_text),
                metadata=metadata,
                structured_data={},
            )

            new_metadata = dict(metadata)
            new_metadata["formato"] = UNIFIED_FORMAT_VERSION
            new_metadata["tipo_consulta"] = tipo_consulta

            updated_ids.append(doc_id)
            updated_docs.append(markdown)
            updated_metas.append(new_metadata)
            to_export.append(Document(page_content=markdown, metadata=new_metadata))

        if updated_ids and not dry_run:
            collection.update(ids=updated_ids, documents=updated_docs, metadatas=updated_metas)

        normalized_count += len(updated_ids)
        offset += len(ids)

    if export_md_dir and to_export:
        exported = export_documents_as_markdown(to_export, output_dir=export_md_dir)
        log.info("Documentos exportados a Markdown: %s en %s", exported, export_md_dir)

    action = "normalizables" if dry_run else "normalizados"
    log.info(
        "Proceso terminado: %s=%s | omitidos=%s | total=%s",
        action,
        normalized_count,
        skipped_count,
        total,
    )


def main():
    parser = argparse.ArgumentParser(description="Normaliza documentos de ChromaDB a Markdown unificado.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME, help="Nombre de la coleccion.")
    parser.add_argument("--local-path", default=DEFAULT_LOCAL_DB_PATH, help="Ruta de Chroma local.")
    parser.add_argument("--cloud", action="store_true", help="Usar Chroma Cloud en vez de Chroma local.")
    parser.add_argument("--batch-size", type=int, default=100, help="Tamano de lote para procesamiento.")
    parser.add_argument(
        "--all-sources",
        action="store_true",
        help="Normaliza todas las fuentes. Por defecto solo las que parecen web scraping.",
    )
    parser.add_argument("--dry-run", action="store_true", help="No escribe cambios, solo reporta.")
    parser.add_argument("--force", action="store_true", help="Reprocesa incluso documentos ya normalizados.")
    parser.add_argument(
        "--export-md-dir",
        default=DEFAULT_EXPORT_DIR,
        help="Carpeta para exportar los documentos Markdown resultantes.",
    )
    parser.add_argument("--no-export-md", action="store_true", help="No exportar archivos Markdown.")

    args = parser.parse_args()

    normalize_collection(
        use_cloud=args.cloud,
        collection_name=args.collection,
        local_path=args.local_path,
        batch_size=max(1, args.batch_size),
        only_web=not args.all_sources,
        dry_run=args.dry_run,
        force=args.force,
        export_md_dir=None if args.no_export_md else args.export_md_dir,
    )


if __name__ == "__main__":
    main()
