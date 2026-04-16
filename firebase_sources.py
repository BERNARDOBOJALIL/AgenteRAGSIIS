"""
firebase_sources.py
Extraccion de datos desde Firestore via REST para alimentar el agente RAG.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Iterator

import requests
from langchain_core.documents import Document

from markdown_unifier import flatten_data, firestore_record_to_document

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = int(os.getenv("FIREBASE_REQUEST_TIMEOUT", "20"))
DEFAULT_PAGE_SIZE = int(os.getenv("FIREBASE_PAGE_SIZE", "200"))
DEFAULT_MAX_DEPTH = int(os.getenv("FIREBASE_MAX_TRAVERSAL_DEPTH", "2"))

DEFAULT_PROFESSOR_FIELD_MARKERS = {
    "rol",
    "role",
    "tipo",
    "tipo_usuario",
    "tipousuario",
    "perfil",
    "cargo",
    "puesto",
    "ocupacion",
    "position",
    "teacher",
    "is_teacher",
    "is_professor",
    "docente",
    "profesor",
}

DEFAULT_PROFESSOR_VALUE_MARKERS = {
    "profesor",
    "profesora",
    "docente",
    "maestro",
    "maestra",
    "teacher",
    "faculty",
    "catedratico",
    "catedratica",
}


@dataclass
class FirestoreRecord:
    data: dict[str, Any]
    metadata: dict[str, Any]


def _split_csv_env(name: str, default: list[str] | None = None) -> list[str]:
    raw = os.getenv(name, "")
    if not raw.strip():
        return default or []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "si", "on"}


def _optional_int_env(name: str, default: int | None) -> int | None:
    """Lee un entero opcional del entorno; acepta 'none'/'all' como sin limite."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default

    lowered = raw.strip().lower()
    if lowered in {"none", "all", "unlimited", "inf", "infinite"}:
        return None

    try:
        return max(0, int(raw.strip()))
    except ValueError:
        log.warning("Valor invalido para %s='%s'. Se usara default=%s", name, raw, default)
        return default


class FirestoreRESTClient:
    """Cliente REST basico para recorrer documentos en Firestore."""

    def __init__(self, *, project_id: str, api_key: str, timeout: int = DEFAULT_TIMEOUT):
        self.project_id = project_id
        self.api_key = api_key
        self.timeout = timeout
        self.documents_root = f"projects/{project_id}/databases/(default)/documents"

    def _request(self, method: str, url: str, **kwargs) -> dict[str, Any]:
        kwargs.setdefault("timeout", self.timeout)
        response = requests.request(method=method, url=url, **kwargs)
        if response.status_code >= 400:
            detail = response.text.strip().replace("\n", " ")
            raise RuntimeError(f"Firestore REST error {response.status_code}: {detail[:350]}")

        if not response.text:
            return {}
        return response.json()

    def list_collection_ids(self, parent_document: str | None = None) -> list[str]:
        parent = parent_document or self.documents_root
        endpoint = f"https://firestore.googleapis.com/v1/{parent}:listCollectionIds"

        collection_ids: list[str] = []
        page_token = ""

        while True:
            payload: dict[str, Any] = {"pageSize": DEFAULT_PAGE_SIZE}
            if page_token:
                payload["pageToken"] = page_token

            data = self._request(
                "POST",
                endpoint,
                params={"key": self.api_key},
                json=payload,
            )

            collection_ids.extend(data.get("collectionIds", []))
            page_token = data.get("nextPageToken", "")
            if not page_token:
                break

        return collection_ids

    def list_collection_documents(self, collection_path: str) -> list[dict[str, Any]]:
        endpoint = f"https://firestore.googleapis.com/v1/{self.documents_root}/{collection_path}"
        page_token = ""
        docs: list[dict[str, Any]] = []

        while True:
            params = {
                "key": self.api_key,
                "pageSize": DEFAULT_PAGE_SIZE,
            }
            if page_token:
                params["pageToken"] = page_token

            data = self._request("GET", endpoint, params=params)
            docs.extend(data.get("documents", []))
            page_token = data.get("nextPageToken", "")
            if not page_token:
                break

        return docs

    def _to_record(self, doc: dict[str, Any], collection_path: str) -> FirestoreRecord:
        """Convierte un documento crudo REST en FirestoreRecord normalizado."""
        record_data = self._parse_firestore_fields(doc.get("fields", {}))
        doc_name = doc.get("name", "")
        doc_id = doc_name.split("/")[-1] if doc_name else "sin_id"

        metadata = {
            "firebase_project": self.project_id,
            "firebase_collection": collection_path,
            "firebase_doc_id": doc_id,
            "firebase_document_name": doc_name,
            "firebase_create_time": doc.get("createTime", ""),
            "firebase_update_time": doc.get("updateTime", ""),
        }
        return FirestoreRecord(data=record_data, metadata=metadata)

    def get_documents_by_id(self, collection_path: str) -> dict[str, FirestoreRecord]:
        """Obtiene documentos de una coleccion indexados por su document ID."""
        docs = self.list_collection_documents(collection_path)
        by_id: dict[str, FirestoreRecord] = {}
        for doc in docs:
            record = self._to_record(doc, collection_path)
            by_id[record.metadata.get("firebase_doc_id", "sin_id")] = record
        return by_id

    def iter_documents(
        self,
        *,
        collection_allowlist: list[str] | None = None,
        include_subcollections: bool = True,
        max_depth: int | None = DEFAULT_MAX_DEPTH,
    ) -> Iterator[FirestoreRecord]:
        root_collections = collection_allowlist or self.list_collection_ids()
        queue: list[tuple[str, int]] = [(collection_path, 0) for collection_path in root_collections]
        visited_paths: set[str] = set()

        while queue:
            collection_path, depth = queue.pop(0)
            if collection_path in visited_paths:
                continue
            visited_paths.add(collection_path)

            try:
                docs = self.list_collection_documents(collection_path)
            except Exception as exc:
                log.warning("No se pudo leer la coleccion '%s': %s", collection_path, exc)
                continue

            for doc in docs:
                record = self._to_record(doc, collection_path)
                yield record

                doc_name = record.metadata.get("firebase_document_name", "")

                can_descend = max_depth is None or depth < max_depth
                if include_subcollections and can_descend and doc_name:
                    try:
                        subcollections = self.list_collection_ids(parent_document=doc_name)
                    except Exception as exc:
                        log.warning("No se pudieron listar subcolecciones para '%s': %s", doc_name, exc)
                        continue

                    if "/documents/" not in doc_name:
                        continue
                    doc_relative_path = doc_name.split("/documents/", 1)[1]
                    for subcollection in subcollections:
                        queue.append((f"{doc_relative_path}/{subcollection}", depth + 1))

    def _parse_firestore_fields(self, fields: dict[str, Any]) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        for key, value in fields.items():
            parsed[key] = self._parse_firestore_value(value)
        return parsed

    def _parse_firestore_value(self, value: dict[str, Any]) -> Any:
        if "stringValue" in value:
            return value["stringValue"]
        if "integerValue" in value:
            raw = value["integerValue"]
            try:
                return int(raw)
            except (TypeError, ValueError):
                return raw
        if "doubleValue" in value:
            raw = value["doubleValue"]
            try:
                return float(raw)
            except (TypeError, ValueError):
                return raw
        if "booleanValue" in value:
            return bool(value["booleanValue"])
        if "nullValue" in value:
            return None
        if "timestampValue" in value:
            return value["timestampValue"]
        if "referenceValue" in value:
            return value["referenceValue"]
        if "geoPointValue" in value:
            point = value["geoPointValue"]
            return {
                "latitude": point.get("latitude"),
                "longitude": point.get("longitude"),
            }
        if "bytesValue" in value:
            return value["bytesValue"]
        if "arrayValue" in value:
            values = value["arrayValue"].get("values", [])
            return [self._parse_firestore_value(item) for item in values]
        if "mapValue" in value:
            inner_fields = value["mapValue"].get("fields", {})
            return self._parse_firestore_fields(inner_fields)

        return value


def _is_professor_record(record: dict[str, Any]) -> bool:
    field_markers = {
        marker.lower()
        for marker in _split_csv_env("FIREBASE_PROFESSOR_FIELD_MARKERS", list(DEFAULT_PROFESSOR_FIELD_MARKERS))
    }
    value_markers = {
        marker.lower()
        for marker in _split_csv_env("FIREBASE_PROFESSOR_VALUE_MARKERS", list(DEFAULT_PROFESSOR_VALUE_MARKERS))
    }

    flat = flatten_data(record)
    if not flat:
        return False

    # Esquema directo esperado en la coleccion 'usuarios'
    tipo = str(record.get("tipo", "")).strip().lower()
    rol = str(record.get("rol", "")).strip().lower()
    if tipo in value_markers:
        return True
    if rol in {"academico", "académico"} and tipo in {"profesor", "profesora", "docente", "teacher"}:
        return True

    for key, value in flat.items():
        key_l = key.lower()
        value_text = str(value).strip().lower()

        key_has_marker = any(marker in key_l for marker in field_markers)
        if key_has_marker:
            if isinstance(value, bool) and value:
                return True
            if any(marker in value_text for marker in value_markers):
                return True

        if "profesor" in key_l and value_text not in {"", "false", "0", "no"}:
            return True

    compound_text = " ".join(f"{k}:{v}" for k, v in flat.items()).lower()
    has_value_marker = any(marker in compound_text for marker in value_markers)
    has_user_context = any(token in compound_text for token in {"usuario", "user", "docente", "profesor"})
    return has_value_marker and has_user_context


def _build_client_from_env(project_var: str, api_key_var: str) -> FirestoreRESTClient | None:
    project_id = os.getenv(project_var, "").strip()
    api_key = os.getenv(api_key_var, "").strip()

    if not project_id or not api_key:
        log.warning("Faltan variables %s o %s. Se omite esta fuente Firestore.", project_var, api_key_var)
        return None

    return FirestoreRESTClient(project_id=project_id, api_key=api_key)


def _collection_env(name: str, default: str) -> str:
    raw = os.getenv(name, "").strip()
    return raw if raw else default


def fetch_salones_documents() -> list[Document]:
    """
    Extrae toda la informacion disponible de la base Firestore de salones IDIT.
    Hace barrido completo del proyecto: todas las colecciones y documentos.
    """
    client = _build_client_from_env("VITE_FIREBASE_SALONES_PROJECT_ID", "VITE_FIREBASE_SALONES_API_KEY")
    if client is None:
        return []

    include_subcollections = True
    max_depth = _optional_int_env("FIREBASE_SALONES_MAX_TRAVERSAL_DEPTH", None)

    documents: list[Document] = []

    for record in client.iter_documents(
        collection_allowlist=None,
        include_subcollections=include_subcollections,
        max_depth=max_depth,
    ):
        metadata = {
            **record.metadata,
            "fuente": "firebase_salones",
            "categoria": "idit_salones",
        }
        documents.append(firestore_record_to_document(record=record.data, metadata=metadata))

    log.info("Firestore salones: %s documentos convertidos a Markdown.", len(documents))
    return documents


def fetch_profesores_documents() -> list[Document]:
    """
    Extrae solo usuarios que sean profesores desde la base Firestore de usuarios
    y relaciona su horario por ID de documento (coleccion 'horarios').
    """
    client = _build_client_from_env("VITE_FIREBASE_PROJECT_ID", "VITE_FIREBASE_API_KEY")
    if client is None:
        return []

    usuarios_collection = _collection_env("FIREBASE_USUARIOS_COLLECTION", "usuarios")
    configured_horarios = _collection_env("FIREBASE_HORARIOS_COLLECTION", "horarios")
    horarios_candidates = [configured_horarios]
    if "horarios" not in horarios_candidates:
        horarios_candidates.append("horarios")
    if "horrios" not in horarios_candidates:
        horarios_candidates.append("horrios")

    usuarios_by_id = client.get_documents_by_id(usuarios_collection)

    horarios_by_id: dict[str, FirestoreRecord] = {}
    horarios_detectadas: list[str] = []
    for horarios_collection in horarios_candidates:
        current = client.get_documents_by_id(horarios_collection)
        if current:
            horarios_detectadas.append(horarios_collection)
            for doc_id, horario_record in current.items():
                horarios_by_id.setdefault(doc_id, horario_record)

    documentos_profesores: list[Document] = []
    profesores_con_horario = 0

    for doc_id, usuario_record in usuarios_by_id.items():
        if not _is_professor_record(usuario_record.data):
            continue

        horario_record = horarios_by_id.get(doc_id)
        if horario_record:
            profesores_con_horario += 1

        record_data = dict(usuario_record.data)
        record_data["horario"] = horario_record.data if horario_record else None

        metadata = {
            **usuario_record.metadata,
            "fuente": "firebase_usuarios",
            "categoria": "idit_personal",
            "es_profesor": True,
            "firebase_collection": usuarios_collection,
            "firebase_related_collection": ",".join(horarios_detectadas) if horarios_detectadas else configured_horarios,
            "firebase_related_doc_id": doc_id,
            "horario_encontrado": bool(horario_record),
        }
        documentos_profesores.append(firestore_record_to_document(record=record_data, metadata=metadata))

    log.info(
        "Firestore usuarios: profesores=%s | con_horario=%s | docs_markdown=%s",
        len(documentos_profesores),
        profesores_con_horario,
        len(documentos_profesores),
    )
    return documentos_profesores


def fetch_firebase_documents() -> list[Document]:
    """Consolida documentos de salones y profesores para la ingesta RAG."""
    docs_salones = fetch_salones_documents()
    docs_profesores = fetch_profesores_documents()
    total = len(docs_salones) + len(docs_profesores)
    log.info("Firestore total convertido: %s documentos Markdown.", total)
    return docs_salones + docs_profesores
