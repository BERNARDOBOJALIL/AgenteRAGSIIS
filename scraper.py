"""
scraper.py
Web scraping para alimentar el RAG universitario.

Soporta:
  - Sitios estáticos   → WebBaseLoader (requests + BeautifulSoup)
  - Sitios dinámicos   → PlaywrightURLLoader (JS rendering)
  - Crawl automático   → RecursiveUrlLoader (sigue links en el mismo dominio)
  - PDFs               → PyPDFLoader
"""

import time
import logging
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    RecursiveUrlLoader,
    WebBaseLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Configuración de fragmentación ─────────────────────────────────────────────
CHUNK_SIZE = 600
CHUNK_OVERLAP = 120


# ── Limpieza de HTML ───────────────────────────────────────────────────────────

def _limpiar_html(html: str) -> str:
    """Extrae texto limpio de HTML eliminando nav, footer, scripts y publicidad."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["nav", "footer", "header", "script", "style", "aside", "form"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


# ── Scrapers individuales ──────────────────────────────────────────────────────

def scrape_estatico(urls: list[str], categoria: str = "web", delay: float = 1.0) -> list[Document]:
    """
    Scraping de páginas estáticas (HTML servido directamente).
    Ideal para la mayoría de sitios universitarios.

    Args:
        urls:      Lista de URLs a scrapear.
        categoria: Metadato de categoría asignado a todos los documentos.
        delay:     Segundos de espera entre requests (ser amable con el servidor).
    """
    documentos = []
    for url in urls:
        try:
            log.info(f"Scrapeando (estático): {url}")
            loader = WebBaseLoader(url)
            loader.requests_kwargs = {"timeout": 15}
            docs = loader.load()
            for doc in docs:
                doc.metadata["categoria"] = categoria
                doc.metadata["fuente"] = "web_scraping"
            documentos.extend(docs)
            time.sleep(delay)
        except Exception as e:
            log.warning(f"Error en {url}: {e}")
    return documentos


def scrape_dinamico(urls: list[str], categoria: str = "web") -> list[Document]:
    """
    Scraping de páginas que requieren JavaScript (React, Angular, etc.).
    Requiere: pip install playwright && playwright install chromium
    """
    try:
        from langchain_community.document_loaders import PlaywrightURLLoader
    except ImportError:
        log.error("Instala playwright: pip install playwright && playwright install chromium")
        return []

    log.info(f"Scrapeando (dinámico con Playwright): {len(urls)} URLs")
    loader = PlaywrightURLLoader(
        urls=urls,
        remove_selectors=["nav", "footer", "header", "script", "style"],
    )
    docs = loader.load()
    for doc in docs:
        doc.metadata["categoria"] = categoria
        doc.metadata["fuente"] = "web_scraping_js"
    return docs


def scrape_sitio_completo(
    url_base: str,
    max_depth: int = 2,
    categoria: str = "web",
) -> list[Document]:
    """
    Crawl automático: sigue todos los links dentro del mismo dominio.
    Útil para indexar un sitio universitario entero.

    Args:
        url_base:  URL raíz del sitio (ej. "https://www.universidad.edu.mx").
        max_depth: Cuántos niveles de links seguir (2 = página + sus hijos).
        categoria: Metadato asignado a todos los documentos.
    """
    log.info(f"Crawleando sitio completo (depth={max_depth}): {url_base}")

    loader = RecursiveUrlLoader(
        url=url_base,
        max_depth=max_depth,
        extractor=_limpiar_html,
        prevent_outside=True,       # no salir del dominio
        timeout=20,
    )
    docs = loader.load()
    for doc in docs:
        doc.metadata["categoria"] = categoria
        doc.metadata["fuente"] = "web_crawl"
    log.info(f"  → {len(docs)} páginas recopiladas")
    return docs


def cargar_pdfs(carpeta: str = ".", patron: str = "*.pdf", categoria: str = "pdf") -> list[Document]:
    """
    Carga todos los PDFs de una carpeta.

    Args:
        carpeta:  Ruta al directorio con PDFs.
        patron:   Glob para filtrar archivos (default: todos los PDFs).
        categoria: Metadato asignado.
    """
    documentos = []
    archivos = list(Path(carpeta).glob(patron))
    if not archivos:
        log.warning(f"No se encontraron PDFs en '{carpeta}' con patrón '{patron}'")
        return []

    for pdf_path in archivos:
        log.info(f"Cargando PDF: {pdf_path.name}")
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            for doc in docs:
                doc.metadata["categoria"] = categoria
                doc.metadata["fuente"] = str(pdf_path.name)
            documentos.extend(docs)
        except Exception as e:
            log.warning(f"Error leyendo {pdf_path.name}: {e}")

    log.info(f"  → {len(documentos)} páginas de PDF cargadas")
    return documentos


def documentos_manuales(items: list[dict]) -> list[Document]:
    """
    Crea documentos desde texto plano (para información que no está en web ni PDF).

    Cada elemento debe tener 'contenido' y opcionalmente cualquier metadato.

    Ejemplo:
        items = [
            {"contenido": "La cafetería abre a las 7am.", "categoria": "servicios"},
        ]
    """
    docs = []
    for item in items:
        contenido = item.pop("contenido", "")
        item.setdefault("fuente", "manual")
        docs.append(Document(page_content=contenido, metadata=item))
    return docs


# ── Fragmentación ──────────────────────────────────────────────────────────────

def fragmentar(documentos: list[Document], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[Document]:
    """Divide documentos en chunks para mejorar la precisión del RAG."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    fragmentos = splitter.split_documents(documentos)
    log.info(f"Fragmentación: {len(documentos)} docs → {len(fragmentos)} chunks")
    return fragmentos


# ── Pipeline completo ──────────────────────────────────────────────────────────

def pipeline_completo(
    urls_estaticas: Optional[list[str]] = None,
    urls_dinamicas: Optional[list[str]] = None,
    url_crawl: Optional[str] = None,
    carpeta_pdfs: Optional[str] = None,
    datos_manuales: Optional[list[dict]] = None,
    max_depth: int = 2,
) -> list[Document]:
    """
    Ejecuta todas las fuentes configuradas y retorna los chunks listos
    para insertar en ChromaDB.

    Ejemplo de uso en ingest.py:
        fragmentos = pipeline_completo(
            urls_estaticas=["https://uni.edu.mx/becas"],
            carpeta_pdfs="./pdfs",
        )
    """
    todos = []

    if urls_estaticas:
        todos.extend(scrape_estatico(urls_estaticas))

    if urls_dinamicas:
        todos.extend(scrape_dinamico(urls_dinamicas))

    if url_crawl:
        todos.extend(scrape_sitio_completo(url_crawl, max_depth=max_depth))

    if carpeta_pdfs:
        todos.extend(cargar_pdfs(carpeta_pdfs))

    if datos_manuales:
        todos.extend(documentos_manuales(datos_manuales))

    if not todos:
        log.warning("No se recopiló ningún documento.")
        return []

    return fragmentar(todos)
