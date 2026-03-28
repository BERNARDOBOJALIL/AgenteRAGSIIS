"""
crud_chroma.py
Gestor CRUD para ChromaDB — Agente Universitario
"""

import uuid
import os
from typing import Optional
import chromadb
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

# ── Configuración ──────────────────────────────────────────────────────────────
COLLECTION_NAME = "asistente_universidad"
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_CLOUD_HOST = os.getenv("CHROMA_CLOUD_HOST", "api.trychroma.com")
CHROMA_CLOUD_PORT = int(os.getenv("CHROMA_CLOUD_PORT", "443"))

_crud_instance = None


def obtener_crud() -> "CRUDChroma":
    """Retorna la instancia singleton del CRUD."""
    global _crud_instance
    if _crud_instance is None:
        _crud_instance = CRUDChroma()
    return _crud_instance


class CRUDChroma:
    """Operaciones CRUD sobre ChromaDB."""

    def __init__(self):
        if not CHROMA_API_KEY:
            raise RuntimeError("Falta CHROMA_API_KEY en el entorno")
        if not CHROMA_TENANT:
            raise RuntimeError("Falta CHROMA_TENANT en el entorno")
        if not CHROMA_DATABASE:
            raise RuntimeError("Falta CHROMA_DATABASE en el entorno")

        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        client = chromadb.CloudClient(
            tenant=CHROMA_TENANT,
            database=CHROMA_DATABASE,
            api_key=CHROMA_API_KEY,
            cloud_host=CHROMA_CLOUD_HOST,
            cloud_port=CHROMA_CLOUD_PORT,
            enable_ssl=True,
        )
        self.vector_store = Chroma(
            client=client,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
        self._collection = self.vector_store._collection

    # ── CREATE ─────────────────────────────────────────────────────────────────

    def insertar_documento(self, contenido: str, **metadatos) -> str:
        """Inserta un documento con metadatos arbitrarios. Retorna el ID."""
        doc_id = str(uuid.uuid4())
        doc = Document(page_content=contenido, metadata=metadatos)
        self.vector_store.add_documents([doc], ids=[doc_id])
        return doc_id

    def insertar_multiples(self, documentos: list[dict]) -> list[str]:
        """
        Inserta varios documentos a la vez.
        Cada elemento debe tener al menos la clave 'contenido'.
        """
        ids = []
        docs = []
        for item in documentos:
            doc_id = str(uuid.uuid4())
            contenido = item.pop("contenido")
            docs.append(Document(page_content=contenido, metadata=item))
            ids.append(doc_id)
        self.vector_store.add_documents(docs, ids=ids)
        return ids

    # ── READ ───────────────────────────────────────────────────────────────────

    def buscar_similitud(self, consulta: str, k: int = 3) -> list[dict]:
        """Búsqueda semántica. Retorna lista con id, contenido, metadatos y similitud."""
        resultados = self.vector_store.similarity_search_with_relevance_scores(consulta, k=k)
        return [
            {
                "id": doc.metadata.get("_id", ""),
                "contenido": doc.page_content,
                "metadatos": doc.metadata,
                "similitud": score,
            }
            for doc, score in resultados
        ]

    def buscar_por_metadatos(self, filtros: dict) -> list[dict]:
        """Filtra documentos exactamente por metadatos."""
        resultado = self._collection.get(where=filtros)
        docs = []
        for i, doc_id in enumerate(resultado["ids"]):
            docs.append(
                {
                    "id": doc_id,
                    "contenido": resultado["documents"][i],
                    "metadatos": resultado["metadatas"][i],
                }
            )
        return docs

    def obtener_por_id(self, doc_id: str) -> Optional[dict]:
        """Obtiene un documento por su ID. Retorna None si no existe."""
        resultado = self._collection.get(ids=[doc_id])
        if not resultado["ids"]:
            return None
        return {
            "id": resultado["ids"][0],
            "contenido": resultado["documents"][0],
            "metadatos": resultado["metadatas"][0],
        }

    def listar_todos(self, limit: int = 50) -> list[dict]:
        """Lista todos los documentos (hasta el límite indicado)."""
        resultado = self._collection.get(limit=limit)
        return [
            {
                "id": resultado["ids"][i],
                "contenido": resultado["documents"][i],
                "metadatos": resultado["metadatas"][i],
            }
            for i in range(len(resultado["ids"]))
        ]

    def obtener_estadisticas(self) -> dict:
        """Resumen de la base de datos."""
        todos = self._collection.get()
        total = len(todos["ids"])
        categorias: dict[str, int] = {}
        fuentes: dict[str, int] = {}
        for meta in todos["metadatas"]:
            cat = meta.get("categoria", "sin_categoria")
            fuente = meta.get("fuente", "sin_fuente")
            categorias[cat] = categorias.get(cat, 0) + 1
            fuentes[fuente] = fuentes.get(fuente, 0) + 1
        return {"total_documentos": total, "categorias": categorias, "fuentes": fuentes}

    # ── UPDATE ─────────────────────────────────────────────────────────────────

    def actualizar_documento(self, doc_id: str, contenido: Optional[str] = None, **metadatos):
        """Actualiza contenido y/o metadatos de un documento existente."""
        existente = self.obtener_por_id(doc_id)
        if not existente:
            raise ValueError(f"Documento {doc_id} no encontrado.")

        nuevo_contenido = contenido if contenido is not None else existente["contenido"]
        nuevos_meta = {**existente["metadatos"], **metadatos}

        self._collection.update(
            ids=[doc_id],
            documents=[nuevo_contenido],
            metadatas=[nuevos_meta],
        )

    # ── DELETE ─────────────────────────────────────────────────────────────────

    def eliminar_por_id(self, doc_id: str):
        """Elimina un documento por su ID."""
        self._collection.delete(ids=[doc_id])

    def eliminar_por_metadatos(self, filtros: dict) -> int:
        """Elimina todos los documentos que coincidan con los filtros. Retorna cantidad."""
        ids_a_eliminar = [doc["id"] for doc in self.buscar_por_metadatos(filtros)]
        if ids_a_eliminar:
            self._collection.delete(ids=ids_a_eliminar)
        return len(ids_a_eliminar)
