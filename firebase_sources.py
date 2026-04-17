"""
firebase_sources.py
Extraccion de datos desde Firestore via REST para alimentar el agente RAG.
"""

from __future__ import annotations

import json
import base64
import logging
import os
import re
import time
import unicodedata
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Iterator

import requests
from dotenv import load_dotenv
from langchain_core.documents import Document

try:
    import firebase_admin  # type: ignore[import-not-found]
    from firebase_admin import credentials as firebase_credentials  # type: ignore[import-not-found]
    from firebase_admin import firestore as firebase_firestore  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - depende de entorno
    firebase_admin = None
    firebase_credentials = None
    firebase_firestore = None

from markdown_unifier import flatten_data, firestore_record_to_document

load_dotenv()

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

DEFAULT_TARGET_USER_ROLES = {
    "academico",
    "académico",
}

DEFAULT_TARGET_USER_TYPES = {
    "profesor",
    "profesora",
    "docente",
    "administrativo",
    "administrativa",
    "admin",
    "aminisrativo",   # typo frecuente reportado
    "aminisrativa",
    "adminisrativo",  # typo frecuente
    "adminisrativa",
}

DEFAULT_FIREBASE_CACHE_TTL_SECONDS = int(os.getenv("FIREBASE_RUNTIME_CACHE_TTL_SECONDS", "45"))

_runtime_cache: dict[str, Any] = {
    "snapshot": None,
    "loaded_at": 0.0,
}

SCHEDULE_LITERAL_MARKERS = {
    "horario completo",
    "horarios completos",
    "todos los horarios",
    "todas las clases",
    "calendario completo",
    "calendario detallado",
    "detalle de horario",
    "detalle de horarios",
}

SCHEDULE_TOPIC_MARKERS = {
    "horario",
    "horarios",
    "calendario",
    "calendarios",
    "clase",
    "clases",
}

SCHEDULE_DETAIL_MARKERS = {
    "completo",
    "completa",
    "detallado",
    "detallada",
    "detalle",
    "todos",
    "todas",
    "todo",
    "dia por dia",
    "por dia",
    "semanal",
    "bloques",
}

J_CODE_PATTERN = re.compile(r"\bJ[\s_-]?(\d{1,3})\b", flags=re.IGNORECASE)

CONSERVATIVE_LAB_EQUIPMENT_HINTS: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    (
        ("quimic",),
        (
            "campana de extraccion",
            "tarja o lavabo de laboratorio",
            "material basico para manejo de reactivos",
        ),
    ),
    (
        ("fisic",),
        (
            "bancos de trabajo",
            "instrumentos de medicion",
            "fuentes de alimentacion de laboratorio",
        ),
    ),
    (
        ("biolog", "microbiolog"),
        (
            "microscopios",
            "mesas de preparacion",
            "material de bioseguridad basica",
        ),
    ),
    (
        ("comput", "informat", "inteligencia artificial", "realidad aumentada", "datos"),
        (
            "computadoras de escritorio",
            "proyector o pantalla",
            "conectividad de red",
        ),
    ),
    (
        ("electron", "mecatron", "robot", "automat"),
        (
            "multimetros",
            "fuentes de poder",
            "estaciones de prototipado electronico",
        ),
    ),
    (
        ("fabricacion digital", "maker", "prototip", "impresion 3d", "corte laser"),
        (
            "equipos de prototipado digital",
            "herramientas de taller supervisado",
        ),
    ),
]


@dataclass
class FirestoreRecord:
    data: dict[str, Any]
    metadata: dict[str, Any]


def _split_csv_env(name: str, default: list[str] | None = None) -> list[str]:
    raw = os.getenv(name, "")
    if not raw.strip():
        return default or []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _collection_allowlist_env(*names: str) -> list[str] | None:
    """Lee una lista de colecciones desde variables de entorno CSV."""
    for name in names:
        values = _split_csv_env(name)
        if values:
            return values
    return None


def _datetime_to_iso(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


def _service_account_from_env(
    *,
    file_var: str | None,
    json_var: str | None,
) -> tuple[dict[str, Any] | None, str]:
    """Carga service account desde archivo o JSON/base64 en variables de entorno."""
    candidates: list[tuple[str | None, str]] = []
    if file_var:
        candidates.append((os.getenv(file_var, "").strip(), f"env:{file_var}"))
    candidates.append((os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip(), "env:GOOGLE_APPLICATION_CREDENTIALS"))

    for path, source in candidates:
        if not path:
            continue
        if not os.path.exists(path):
            log.warning("Ruta de credencial no existe (%s): %s", source, path)
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if isinstance(payload, dict):
                return payload, source
        except Exception as exc:
            log.warning("No se pudo leer credencial JSON (%s): %s", source, exc)

    json_candidates: list[tuple[str | None, str]] = []
    if json_var:
        json_candidates.append((os.getenv(json_var, "").strip(), f"env:{json_var}"))
        json_candidates.append((os.getenv(f"{json_var}_BASE64", "").strip(), f"env:{json_var}_BASE64"))
    json_candidates.append((os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip(), "env:FIREBASE_SERVICE_ACCOUNT_JSON"))
    json_candidates.append((os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON_BASE64", "").strip(), "env:FIREBASE_SERVICE_ACCOUNT_JSON_BASE64"))

    for raw, source in json_candidates:
        if not raw:
            continue
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                return payload, source
        except Exception:
            pass

        try:
            decoded = base64.b64decode(raw).decode("utf-8")
            payload = json.loads(decoded)
            if isinstance(payload, dict):
                return payload, source
        except Exception as exc:
            log.warning("No se pudo parsear service account desde %s: %s", source, exc)

    return None, ""


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "si", "on"}


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


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


def _question_requests_detailed_schedule(question: str) -> bool:
    """Detecta si el usuario pidio horario/calendario detallado de forma explicita."""
    normalized = _normalize_text(question)
    if not normalized:
        return False

    if any(marker in normalized for marker in SCHEDULE_LITERAL_MARKERS):
        return True

    has_topic = any(marker in normalized for marker in SCHEDULE_TOPIC_MARKERS)
    has_detail = any(marker in normalized for marker in SCHEDULE_DETAIL_MARKERS)
    return has_topic and has_detail


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _clean_string_list(values: Any, *, max_items: int = 12) -> list[str]:
    if not isinstance(values, list):
        return []

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _clean_text(item)
        if not text:
            continue
        key = _normalize_text(text)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
        if len(cleaned) >= max_items:
            break
    return cleaned


def _extract_responsable_names(values: Any, *, max_items: int = 4) -> list[str]:
    if not isinstance(values, list):
        return []

    names: list[str] = []
    seen: set[str] = set()

    for item in values:
        if isinstance(item, dict):
            text = _clean_text(item.get("nombre") or item.get("name"))
        else:
            text = _clean_text(item)

        if not text:
            continue
        key = _normalize_text(text)
        if key in seen:
            continue
        seen.add(key)
        names.append(text)

        if len(names) >= max_items:
            break

    return names


def _extract_j_code_number(*values: Any) -> int | None:
    for value in values:
        text = _clean_text(value)
        if not text:
            continue
        match = J_CODE_PATTERN.search(text)
        if not match:
            continue
        try:
            return int(match.group(1))
        except ValueError:
            continue
    return None


def _normalize_floor_label(value: Any) -> str:
    raw = _clean_text(value)
    normalized = _normalize_text(raw)

    if normalized in {"pb", "planta baja", "baja", "nivel 0", "0"}:
        return "planta baja"
    if normalized in {"pa", "planta alta", "alta", "nivel 1", "1"}:
        return "planta alta"

    return raw


def _zone_hint_for_j_code(code: int | None) -> str:
    if code is None:
        return ""
    if code <= 8:
        return "zona cercana al acceso principal"
    if code <= 17:
        return "zona media del pasillo principal"
    return "zona del fondo del pasillo principal"


def _stairs_hint_for_j_code(code: int | None) -> str:
    if code is None:
        return ""
    if 10 <= code <= 18:
        return "referencia cercana a la escalera entre plantas"
    return ""


def _special_area_hint(name_blob: str) -> str:
    normalized = _normalize_text(name_blob)
    if not normalized:
        return ""

    if "enfermer" in normalized or "primeros auxilios" in normalized:
        return "zona de servicios de apoyo"
    if "easyplot" in normalized or "oficina" in normalized:
        return "zona administrativa"
    if "laboratorio" in normalized:
        return "bloque de laboratorios"
    return ""


def _build_salon_location_hint(salon_data: dict[str, Any]) -> str:
    nomenclatura = _clean_text(salon_data.get("nomenclatura"))
    nombre = _clean_text(salon_data.get("nombre"))
    piso = _normalize_floor_label(salon_data.get("piso"))
    code = _extract_j_code_number(nomenclatura, nombre)

    hints: list[str] = []
    for hint in [
        piso,
        _zone_hint_for_j_code(code),
        _stairs_hint_for_j_code(code),
        _special_area_hint(f"{nomenclatura} {nombre}"),
    ]:
        if hint and hint not in hints:
            hints.append(hint)

    return ", ".join(hints) if hints else "ubicacion aproximada no disponible"


def _infer_conservative_equipment_from_lab_name(
    *,
    nombre: Any,
    tipo: Any,
    nomenclatura: Any,
    max_items: int = 3,
) -> list[str]:
    """Infiere equipamiento probable de forma conservadora usando solo el nombre/tipo del laboratorio."""
    nombre_txt = _normalize_text(nombre)
    tipo_txt = _normalize_text(tipo)
    nomenclatura_txt = _normalize_text(nomenclatura)
    blob = " ".join(part for part in [nombre_txt, tipo_txt, nomenclatura_txt] if part).strip()

    if not blob:
        return []

    if "laboratorio" not in blob and "laboratorio" not in tipo_txt:
        return []

    inferred: list[str] = []
    seen: set[str] = set()

    for keywords, suggestions in CONSERVATIVE_LAB_EQUIPMENT_HINTS:
        if not any(keyword in blob for keyword in keywords):
            continue

        for suggestion in suggestions:
            key = _normalize_text(suggestion)
            if key in seen:
                continue
            seen.add(key)
            inferred.append(suggestion)
            if len(inferred) >= max_items:
                return inferred

    return inferred


def _compact_nested_payload(value: Any, *, max_items: int = 12, max_depth: int = 2) -> Any:
    if max_depth < 0:
        return "..."

    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= max_items:
                compact["truncated"] = True
                break
            compact[str(key)] = _compact_nested_payload(item, max_items=max_items, max_depth=max_depth - 1)
        return compact

    if isinstance(value, list):
        compact_list: list[Any] = []
        for idx, item in enumerate(value):
            if idx >= max_items:
                compact_list.append({"truncated": True})
                break
            compact_list.append(_compact_nested_payload(item, max_items=max_items, max_depth=max_depth - 1))
        return compact_list

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def _compact_horario_payload(horario: Any, *, max_items: int = 14) -> Any:
    if isinstance(horario, list):
        compact: list[Any] = []
        for item in horario:
            if len(compact) >= max_items:
                break
            if isinstance(item, dict):
                dia = _clean_text(item.get("dia"))
                inicio = _clean_text(item.get("inicio"))
                fin = _clean_text(item.get("fin"))
                if dia or inicio or fin:
                    compact.append(
                        {
                            "dia": dia or None,
                            "inicio": inicio or None,
                            "fin": fin or None,
                        }
                    )
            else:
                text = _clean_text(item)
                if text:
                    compact.append({"valor": text})
        return compact

    if isinstance(horario, dict):
        return _compact_nested_payload(horario, max_items=max_items, max_depth=2)

    return _compact_nested_payload(horario, max_items=max_items, max_depth=1)


def _schedule_presence_summary(horario: Any, calendario: Any) -> str:
    parts: list[str] = []
    if horario:
        parts.append("horario registrado")
    if calendario:
        parts.append("calendario registrado")
    if not parts:
        return "sin horario registrado"
    return " y ".join(parts)


def _compact_salon_runtime_payload(item: dict[str, Any], *, include_schedule: bool) -> dict[str, Any]:
    data = item.get("data") if isinstance(item.get("data"), dict) else {}
    equipamiento_real = _clean_string_list(data.get("equipamiento"), max_items=12)
    equipamiento_inferido = []

    if not equipamiento_real:
        equipamiento_inferido = _infer_conservative_equipment_from_lab_name(
            nombre=data.get("nombre"),
            tipo=data.get("tipo"),
            nomenclatura=data.get("nomenclatura"),
            max_items=3,
        )

    compact = {
        "doc_id": _clean_text(item.get("doc_id")),
        "nomenclatura": _clean_text(data.get("nomenclatura")),
        "nombre": _clean_text(data.get("nombre")),
        "tipo": _clean_text(data.get("tipo")),
        "piso": _normalize_floor_label(data.get("piso")),
        "ubicacion_aproximada": _build_salon_location_hint(data),
        "equipamiento": equipamiento_real,
        "equipamiento_inferido_conservador": equipamiento_inferido,
        "nota_inferencia_equipamiento": (
            "estimacion conservadora basada en el nombre del laboratorio"
            if equipamiento_inferido
            else ""
        ),
        "responsables": _extract_responsable_names(data.get("responsables"), max_items=4),
        "reserva": data.get("reserva"),
        "idConjunto": data.get("idConjunto"),
    }

    horario = data.get("horario")
    tipo_horario = _clean_text(data.get("tipoHorario"))

    if include_schedule and horario:
        compact["tipoHorario"] = tipo_horario
        compact["horario"] = _compact_horario_payload(horario, max_items=16)
    else:
        compact["tipoHorario"] = tipo_horario
        compact["tiene_horario"] = bool(horario)

    cleaned = {
        key: value
        for key, value in compact.items()
        if value not in (None, "", [], {})
    }

    if not cleaned.get("nomenclatura") and cleaned.get("doc_id"):
        cleaned["nomenclatura"] = cleaned["doc_id"]

    return cleaned


def _compact_user_runtime_payload(item: dict[str, Any], *, include_schedule: bool) -> dict[str, Any]:
    usuario = item.get("usuario") if isinstance(item.get("usuario"), dict) else {}
    horario = item.get("horario")
    calendario = item.get("calendario")

    compact = {
        "doc_id": _clean_text(item.get("doc_id")),
        "nombre": _clean_text(usuario.get("nombre") or usuario.get("name")),
        "rol": _clean_text(usuario.get("rol")),
        "tipo": _clean_text(usuario.get("tipo")),
        "correo": _clean_text(usuario.get("correo") or usuario.get("email")),
        "puesto": _clean_text(usuario.get("puesto") or usuario.get("cargo")),
    }

    if include_schedule:
        if horario:
            compact["horario"] = _compact_horario_payload(horario, max_items=16)
        if calendario:
            compact["calendario"] = _compact_nested_payload(calendario, max_items=14, max_depth=3)
    else:
        compact["resumen_horario"] = _schedule_presence_summary(horario, calendario)
        compact["tiene_horario"] = bool(horario)
        compact["tiene_calendario"] = bool(calendario)

    return {
        key: value
        for key, value in compact.items()
        if value not in (None, "", [], {})
    }


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
        if collection_allowlist:
            root_collections = collection_allowlist
        else:
            try:
                root_collections = self.list_collection_ids()
            except Exception as exc:
                raise RuntimeError(
                    "No se pudieron listar colecciones raiz en Firestore REST. "
                    "Define una variable *_COLLECTION_ALLOWLIST con los nombres de coleccion "
                    "o usa credenciales administrativas para descubrimiento completo."
                ) from exc
        queue: list[tuple[str, int]] = [(collection_path, 0) for collection_path in root_collections]
        visited_paths: set[str] = set()
        subcollection_error_count = 0

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
                        subcollection_error_count += 1
                        if subcollection_error_count <= 3:
                            log.warning("No se pudieron listar subcolecciones para '%s': %s", doc_name, exc)
                        elif subcollection_error_count == 4:
                            log.warning(
                                "Se detectaron multiples errores al listar subcolecciones. "
                                "Se omiten advertencias adicionales para reducir ruido."
                            )
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


class FirestoreAdminClient:
    """Cliente Firestore via Firebase Admin SDK (lectura administrativa)."""

    def __init__(self, *, project_id: str, service_account: dict[str, Any], app_name: str):
        if firebase_admin is None or firebase_credentials is None or firebase_firestore is None:
            raise RuntimeError("El paquete firebase-admin no esta disponible. Instala 'firebase-admin'.")

        self.project_id = project_id
        self.app_name = app_name

        existing_apps = getattr(firebase_admin, "_apps", {})
        if app_name in existing_apps:
            app = firebase_admin.get_app(app_name)
        else:
            cred = firebase_credentials.Certificate(service_account)
            app = firebase_admin.initialize_app(cred, options={"projectId": project_id}, name=app_name)

        self._db = firebase_firestore.client(app=app)

    def _doc_name(self, doc_path: str) -> str:
        return f"projects/{self.project_id}/databases/(default)/documents/{doc_path}"

    def _to_record_from_snapshot(self, snapshot: Any, collection_path: str) -> FirestoreRecord:
        data = snapshot.to_dict() or {}
        doc_path = snapshot.reference.path
        metadata = {
            "firebase_project": self.project_id,
            "firebase_collection": collection_path,
            "firebase_doc_id": snapshot.id,
            "firebase_document_name": self._doc_name(doc_path),
            "firebase_create_time": _datetime_to_iso(getattr(snapshot, "create_time", "")),
            "firebase_update_time": _datetime_to_iso(getattr(snapshot, "update_time", "")),
        }
        return FirestoreRecord(data=data, metadata=metadata)

    def _normalize_parent_doc_path(self, parent_document: str) -> str:
        if "/documents/" in parent_document:
            return parent_document.split("/documents/", 1)[1]
        return parent_document

    def list_collection_ids(self, parent_document: str | None = None) -> list[str]:
        if parent_document:
            relative = self._normalize_parent_doc_path(parent_document)
            doc_ref = self._db.document(relative)
            return [collection.id for collection in doc_ref.collections()]

        return [collection.id for collection in self._db.collections()]

    def list_collection_documents(self, collection_path: str) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        for snapshot in self._db.collection(collection_path).stream():
            docs.append(
                {
                    "id": snapshot.id,
                    "name": self._doc_name(snapshot.reference.path),
                    "data": snapshot.to_dict() or {},
                    "createTime": _datetime_to_iso(getattr(snapshot, "create_time", "")),
                    "updateTime": _datetime_to_iso(getattr(snapshot, "update_time", "")),
                }
            )
        return docs

    def get_documents_by_id(self, collection_path: str) -> dict[str, FirestoreRecord]:
        by_id: dict[str, FirestoreRecord] = {}
        for snapshot in self._db.collection(collection_path).stream():
            record = self._to_record_from_snapshot(snapshot, collection_path)
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
                snapshots = list(self._db.collection(collection_path).stream())
            except Exception as exc:
                log.warning("No se pudo leer la coleccion '%s' (admin): %s", collection_path, exc)
                continue

            for snapshot in snapshots:
                record = self._to_record_from_snapshot(snapshot, collection_path)
                yield record

                can_descend = max_depth is None or depth < max_depth
                if not include_subcollections or not can_descend:
                    continue

                try:
                    subcollections = list(snapshot.reference.collections())
                except Exception as exc:
                    log.warning(
                        "No se pudieron listar subcolecciones (admin) para '%s': %s",
                        snapshot.reference.path,
                        exc,
                    )
                    continue

                for subcollection in subcollections:
                    queue.append((f"{snapshot.reference.path}/{subcollection.id}", depth + 1))


def _is_professor_record(record: dict[str, Any]) -> bool:
    """Compatibilidad historica: ahora aplica filtro academico + tipo profesor/administrativo."""
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

    allowed_roles = {
        _normalize_text(marker)
        for marker in _split_csv_env("FIREBASE_TARGET_ROLES", list(DEFAULT_TARGET_USER_ROLES))
    }
    allowed_types = {
        _normalize_text(marker)
        for marker in _split_csv_env("FIREBASE_TARGET_TYPES", list(DEFAULT_TARGET_USER_TYPES))
    }

    # Esquema directo esperado en la coleccion 'usuarios'
    tipo = _normalize_text(record.get("tipo", ""))
    rol = _normalize_text(record.get("rol", ""))
    if rol in allowed_roles and tipo in allowed_types:
        return True

    found_roles: set[str] = set()
    found_types: set[str] = set()

    for key, value in flat.items():
        key_l = _normalize_text(key)
        value_text = _normalize_text(value)

        key_has_marker = any(_normalize_text(marker) in key_l for marker in field_markers)
        if key_has_marker:
            if isinstance(value, bool) and value:
                found_types.add("teacher")
            if any(_normalize_text(marker) in value_text for marker in value_markers):
                found_types.add(value_text)

        if "rol" in key_l or "role" in key_l:
            found_roles.add(value_text)

        if "tipo" in key_l or "type" in key_l or "puesto" in key_l or "cargo" in key_l:
            found_types.add(value_text)

        if "profesor" in key_l and value_text not in {"", "false", "0", "no"}:
            found_types.add("profesor")

    if rol:
        found_roles.add(rol)
    if tipo:
        found_types.add(tipo)

    has_role = any(role in found_roles for role in allowed_roles)

    has_type = any(found_type in allowed_types for found_type in found_types)
    if not has_type:
        has_type = any(
            any(target in found_type for target in allowed_types)
            for found_type in found_types
        )

    return has_role and has_type


def _build_client_from_env(
    project_var: str,
    api_key_var: str,
    *,
    service_account_file_var: str | None = None,
    service_account_json_var: str | None = None,
    admin_app_name: str | None = None,
) -> FirestoreRESTClient | FirestoreAdminClient | None:
    project_id = os.getenv(project_var, "").strip()
    api_key = os.getenv(api_key_var, "").strip()

    service_account, source = _service_account_from_env(
        file_var=service_account_file_var,
        json_var=service_account_json_var,
    )
    if service_account:
        effective_project_id = project_id or str(service_account.get("project_id") or "").strip()
        if not effective_project_id:
            log.warning(
                "No fue posible resolver project_id para cliente admin en %s. Revisa %s y credencial.",
                project_var,
                source,
            )
        else:
            try:
                return FirestoreAdminClient(
                    project_id=effective_project_id,
                    service_account=service_account,
                    app_name=admin_app_name or f"admin_{effective_project_id}",
                )
            except Exception as exc:
                log.warning("No se pudo inicializar cliente Firestore Admin (%s): %s", source, exc)

    if not project_id or not api_key:
        log.warning("Faltan variables %s o %s. Se omite esta fuente Firestore.", project_var, api_key_var)
        return None

    return FirestoreRESTClient(project_id=project_id, api_key=api_key)


def _collection_env(name: str, default: str) -> str:
    raw = os.getenv(name, "").strip()
    return raw if raw else default


def fetch_salones_documents() -> list[Document]:
    """
    Extrae informacion de la base Firestore de salones IDIT.
    Por defecto recorre documentos de colecciones raiz (sin subcolecciones).
    """
    client = _build_client_from_env(
        "VITE_FIREBASE_SALONES_PROJECT_ID",
        "VITE_FIREBASE_SALONES_API_KEY",
        service_account_file_var="FIREBASE_SALONES_SERVICE_ACCOUNT_FILE",
        service_account_json_var="FIREBASE_SALONES_SERVICE_ACCOUNT_JSON",
        admin_app_name="firebase_admin_salones",
    )
    if client is None:
        return []

    include_subcollections = _bool_env("FIREBASE_SALONES_INCLUDE_SUBCOLLECTIONS", True)
    max_depth = _optional_int_env("FIREBASE_SALONES_MAX_TRAVERSAL_DEPTH", None)
    collection_allowlist = _collection_allowlist_env(
        "FIREBASE_SALONES_COLLECTION_ALLOWLIST",
        "FIREBASE_SALONES_COLLECTIONS",
    )

    documents: list[Document] = []

    for record in client.iter_documents(
        collection_allowlist=collection_allowlist,
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
    Extrae usuarios academicos cuyo tipo sea profesor/administrativo
    y relaciona horario + calendario por ID de documento.
    """
    client = _build_client_from_env(
        "VITE_FIREBASE_PROJECT_ID",
        "VITE_FIREBASE_API_KEY",
        service_account_file_var="FIREBASE_USERS_SERVICE_ACCOUNT_FILE",
        service_account_json_var="FIREBASE_USERS_SERVICE_ACCOUNT_JSON",
        admin_app_name="firebase_admin_users",
    )
    if client is None:
        return []

    usuarios_collection = _collection_env("FIREBASE_USUARIOS_COLLECTION", "usuarios")
    configured_horarios = _collection_env("FIREBASE_HORARIOS_COLLECTION", "horarios")
    configured_calendarios = _collection_env("FIREBASE_CALENDARIOS_COLLECTION", "calendarios")
    horarios_candidates = [configured_horarios]
    if "horarios" not in horarios_candidates:
        horarios_candidates.append("horarios")
    if "horrios" not in horarios_candidates:
        horarios_candidates.append("horrios")

    calendarios_candidates = [configured_calendarios]
    if "calendarios" not in calendarios_candidates:
        calendarios_candidates.append("calendarios")
    if "calendario" not in calendarios_candidates:
        calendarios_candidates.append("calendario")

    usuarios_by_id = client.get_documents_by_id(usuarios_collection)

    horarios_by_id: dict[str, FirestoreRecord] = {}
    horarios_detectadas: list[str] = []
    for horarios_collection in horarios_candidates:
        current = client.get_documents_by_id(horarios_collection)
        if current:
            horarios_detectadas.append(horarios_collection)
            for doc_id, horario_record in current.items():
                horarios_by_id.setdefault(doc_id, horario_record)

    calendarios_by_id: dict[str, FirestoreRecord] = {}
    calendarios_detectadas: list[str] = []
    for calendarios_collection in calendarios_candidates:
        current = client.get_documents_by_id(calendarios_collection)
        if current:
            calendarios_detectadas.append(calendarios_collection)
            for doc_id, calendario_record in current.items():
                calendarios_by_id.setdefault(doc_id, calendario_record)

    documentos_profesores: list[Document] = []
    profesores_con_horario = 0
    profesores_con_calendario = 0

    for doc_id, usuario_record in usuarios_by_id.items():
        if not _is_professor_record(usuario_record.data):
            continue

        horario_record = horarios_by_id.get(doc_id)
        if horario_record:
            profesores_con_horario += 1

        calendario_record = calendarios_by_id.get(doc_id)
        if calendario_record:
            profesores_con_calendario += 1

        record_data = dict(usuario_record.data)
        record_data["horario"] = horario_record.data if horario_record else None
        record_data["calendario"] = calendario_record.data if calendario_record else None

        metadata = {
            **usuario_record.metadata,
            "fuente": "firebase_usuarios",
            "categoria": "idit_personal",
            "es_usuario_academico_objetivo": True,
            "firebase_collection": usuarios_collection,
            "firebase_related_collection": ",".join(horarios_detectadas) if horarios_detectadas else configured_horarios,
            "firebase_related_doc_id": doc_id,
            "horario_encontrado": bool(horario_record),
            "firebase_calendar_collection": ",".join(calendarios_detectadas) if calendarios_detectadas else configured_calendarios,
            "calendario_encontrado": bool(calendario_record),
        }
        documentos_profesores.append(firestore_record_to_document(record=record_data, metadata=metadata))

    log.info(
        "Firestore usuarios objetivo: usuarios=%s | con_horario=%s | con_calendario=%s | docs_markdown=%s",
        len(documentos_profesores),
        profesores_con_horario,
        profesores_con_calendario,
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


def _record_payload(record: FirestoreRecord, source_name: str) -> dict[str, Any]:
    return {
        "source": source_name,
        "project": record.metadata.get("firebase_project"),
        "collection": record.metadata.get("firebase_collection"),
        "doc_id": record.metadata.get("firebase_doc_id"),
        "create_time": record.metadata.get("firebase_create_time"),
        "update_time": record.metadata.get("firebase_update_time"),
        "data": record.data,
    }


def fetch_salones_raw_records() -> list[dict[str, Any]]:
    """Obtiene toda la informacion de salones en formato crudo (sin Markdown)."""
    client = _build_client_from_env(
        "VITE_FIREBASE_SALONES_PROJECT_ID",
        "VITE_FIREBASE_SALONES_API_KEY",
        service_account_file_var="FIREBASE_SALONES_SERVICE_ACCOUNT_FILE",
        service_account_json_var="FIREBASE_SALONES_SERVICE_ACCOUNT_JSON",
        admin_app_name="firebase_admin_salones",
    )
    if client is None:
        return []

    include_subcollections = _bool_env("FIREBASE_SALONES_INCLUDE_SUBCOLLECTIONS", True)
    max_depth = _optional_int_env("FIREBASE_SALONES_MAX_TRAVERSAL_DEPTH", None)
    collection_allowlist = _collection_allowlist_env(
        "FIREBASE_SALONES_COLLECTION_ALLOWLIST",
        "FIREBASE_SALONES_COLLECTIONS",
    )

    salones: list[dict[str, Any]] = []
    for record in client.iter_documents(
        collection_allowlist=collection_allowlist,
        include_subcollections=include_subcollections,
        max_depth=max_depth,
    ):
        salones.append(_record_payload(record, "firebase_salones"))

    return salones


def fetch_target_users_raw_records() -> list[dict[str, Any]]:
    """Obtiene usuarios academicos objetivo y vincula horarios/calendarios por doc_id."""
    client = _build_client_from_env(
        "VITE_FIREBASE_PROJECT_ID",
        "VITE_FIREBASE_API_KEY",
        service_account_file_var="FIREBASE_USERS_SERVICE_ACCOUNT_FILE",
        service_account_json_var="FIREBASE_USERS_SERVICE_ACCOUNT_JSON",
        admin_app_name="firebase_admin_users",
    )
    if client is None:
        return []

    usuarios_collection = _collection_env("FIREBASE_USUARIOS_COLLECTION", "usuarios")
    configured_horarios = _collection_env("FIREBASE_HORARIOS_COLLECTION", "horarios")
    configured_calendarios = _collection_env("FIREBASE_CALENDARIOS_COLLECTION", "calendarios")

    horarios_candidates = [configured_horarios]
    if "horarios" not in horarios_candidates:
        horarios_candidates.append("horarios")
    if "horrios" not in horarios_candidates:
        horarios_candidates.append("horrios")

    calendarios_candidates = [configured_calendarios]
    if "calendarios" not in calendarios_candidates:
        calendarios_candidates.append("calendarios")
    if "calendario" not in calendarios_candidates:
        calendarios_candidates.append("calendario")

    usuarios_by_id = client.get_documents_by_id(usuarios_collection)

    horarios_by_id: dict[str, FirestoreRecord] = {}
    for horarios_collection in horarios_candidates:
        current = client.get_documents_by_id(horarios_collection)
        for doc_id, horario_record in current.items():
            horarios_by_id.setdefault(doc_id, horario_record)

    calendarios_by_id: dict[str, FirestoreRecord] = {}
    for calendarios_collection in calendarios_candidates:
        current = client.get_documents_by_id(calendarios_collection)
        for doc_id, calendario_record in current.items():
            calendarios_by_id.setdefault(doc_id, calendario_record)

    payloads: list[dict[str, Any]] = []
    for doc_id, usuario_record in usuarios_by_id.items():
        if not _is_professor_record(usuario_record.data):
            continue

        horario_record = horarios_by_id.get(doc_id)
        calendario_record = calendarios_by_id.get(doc_id)

        payloads.append(
            {
                "source": "firebase_usuarios",
                "project": usuario_record.metadata.get("firebase_project"),
                "collection": usuarios_collection,
                "doc_id": doc_id,
                "usuario": usuario_record.data,
                "horario": horario_record.data if horario_record else None,
                "calendario": calendario_record.data if calendario_record else None,
                "horario_collection": horario_record.metadata.get("firebase_collection") if horario_record else None,
                "calendario_collection": calendario_record.metadata.get("firebase_collection") if calendario_record else None,
            }
        )

    return payloads


def fetch_users_all_raw_records() -> list[dict[str, Any]]:
    """Obtiene toda la informacion del proyecto de usuarios en formato crudo."""
    client = _build_client_from_env(
        "VITE_FIREBASE_PROJECT_ID",
        "VITE_FIREBASE_API_KEY",
        service_account_file_var="FIREBASE_USERS_SERVICE_ACCOUNT_FILE",
        service_account_json_var="FIREBASE_USERS_SERVICE_ACCOUNT_JSON",
        admin_app_name="firebase_admin_users",
    )
    if client is None:
        return []

    include_subcollections = _bool_env("FIREBASE_USERS_INCLUDE_SUBCOLLECTIONS", True)
    max_depth = _optional_int_env("FIREBASE_USERS_MAX_TRAVERSAL_DEPTH", None)
    collection_allowlist = _collection_allowlist_env(
        "FIREBASE_USERS_COLLECTION_ALLOWLIST",
        "FIREBASE_USERS_COLLECTIONS",
    )

    records: list[dict[str, Any]] = []
    for record in client.iter_documents(
        collection_allowlist=collection_allowlist,
        include_subcollections=include_subcollections,
        max_depth=max_depth,
    ):
        records.append(_record_payload(record, "firebase_users_all"))

    return records


def _compact_generic_runtime_payload(item: dict[str, Any], *, include_schedule: bool) -> dict[str, Any]:
    data = item.get("data")
    compact: dict[str, Any] = {
        "source": _clean_text(item.get("source")),
        "collection": _clean_text(item.get("collection")),
        "doc_id": _clean_text(item.get("doc_id")),
    }

    if isinstance(data, dict):
        for key in (
            "nombre",
            "name",
            "nomenclatura",
            "tipo",
            "rol",
            "correo",
            "email",
            "descripcion",
            "estado",
            "area",
            "departamento",
            "ubicacion",
            "piso",
        ):
            value = data.get(key)
            text = _clean_text(value)
            if text:
                compact[key] = text

        horario = data.get("horario")
        calendario = data.get("calendario")
        if include_schedule:
            if horario:
                compact["horario"] = _compact_horario_payload(horario, max_items=12)
            if calendario:
                compact["calendario"] = _compact_nested_payload(calendario, max_items=10, max_depth=2)
        else:
            if "horario" in data:
                compact["tiene_horario"] = bool(horario)
            if "calendario" in data:
                compact["tiene_calendario"] = bool(calendario)

        extras: dict[str, Any] = {}
        for key, value in data.items():
            if key in {
                "horario",
                "calendario",
                "nombre",
                "name",
                "nomenclatura",
                "tipo",
                "rol",
                "correo",
                "email",
                "descripcion",
                "estado",
                "area",
                "departamento",
                "ubicacion",
                "piso",
            }:
                continue
            if len(extras) >= 5:
                break
            extras[str(key)] = _compact_nested_payload(value, max_items=8, max_depth=1)

        if extras:
            compact["campos_adicionales"] = extras
    else:
        compact["data"] = _compact_nested_payload(data, max_items=8, max_depth=1)

    return {
        key: value
        for key, value in compact.items()
        if value not in (None, "", [], {})
    }


def fetch_firebase_live_snapshot(*, force_refresh: bool = False) -> dict[str, Any]:
    """Snapshot en vivo de Firebase para consultas runtime del agente (sin Markdown)."""
    now = time.time()
    cached_snapshot = _runtime_cache.get("snapshot")
    loaded_at = float(_runtime_cache.get("loaded_at", 0.0) or 0.0)
    ttl = max(0, DEFAULT_FIREBASE_CACHE_TTL_SECONDS)

    if not force_refresh and cached_snapshot is not None and ttl > 0 and (now - loaded_at) <= ttl:
        return cached_snapshot

    salones: list[dict[str, Any]] = []
    usuarios: list[dict[str, Any]] = []
    usuarios_all: list[dict[str, Any]] = []
    errors: list[str] = []

    try:
        salones = fetch_salones_raw_records()
    except Exception as exc:
        errors.append(f"salones: {exc}")
        log.warning("No fue posible cargar salones en vivo: %s", exc)

    try:
        usuarios = fetch_target_users_raw_records()
    except Exception as exc:
        errors.append(f"usuarios: {exc}")
        log.warning("No fue posible cargar usuarios en vivo: %s", exc)

    try:
        usuarios_all = fetch_users_all_raw_records()
    except Exception as exc:
        errors.append(f"usuarios_all: {exc}")
        log.warning("No fue posible cargar colecciones completas de usuarios en vivo: %s", exc)

    snapshot = {
        "loaded_at_epoch": now,
        "cache_ttl_seconds": ttl,
        "errors": errors,
        "partial": bool(errors),
        "totals": {
            "salones": len(salones),
            "usuarios_objetivo": len(usuarios),
            "usuarios_all": len(usuarios_all),
        },
        "salones": salones,
        "usuarios": usuarios,
        "usuarios_all": usuarios_all,
    }
    _runtime_cache["snapshot"] = snapshot
    _runtime_cache["loaded_at"] = now
    return snapshot


def _tokenize_for_match(text: str) -> set[str]:
    tokens = [tok for tok in _normalize_text(text).replace("/", " ").replace("-", " ").split() if len(tok) >= 3]
    return set(tokens)


def _record_blob(record: dict[str, Any]) -> str:
    flat = flatten_data(record)
    if not flat:
        return _normalize_text(json.dumps(record, ensure_ascii=False))
    compact = " ".join(f"{k}:{v}" for k, v in flat.items())
    return _normalize_text(compact)


def _score_record(record: dict[str, Any], terms: set[str]) -> int:
    if not terms:
        return 1
    blob = _record_blob(record)
    return sum(1 for term in terms if term in blob)


def _serialize_items_with_char_budget(items: list[dict[str, Any]], max_chars: int) -> tuple[str, int, int]:
    """Serializa una lista JSON sin rebasar un presupuesto de caracteres."""
    selected: list[dict[str, Any]] = []
    consumed = 2  # []
    for item in items:
        chunk = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        extra = len(chunk) + (1 if selected else 0)
        if selected and consumed + extra > max_chars:
            break
        if not selected and len(chunk) + 2 > max_chars:
            selected.append({"truncated": True, "preview": chunk[: max(120, max_chars - 40)]})
            consumed = max_chars
            break
        selected.append(item)
        consumed += extra

    payload = json.dumps(selected, ensure_ascii=False, separators=(",", ":"))
    return payload, len(selected), len(items)


def build_firebase_runtime_context(question: str, *, max_salones: int = 3, max_usuarios: int = 3) -> dict[str, Any]:
    """Construye contexto Firebase en vivo (sin Markdown) relevante para la pregunta."""
    schedule_details_requested = _question_requests_detailed_schedule(question)

    try:
        snapshot = fetch_firebase_live_snapshot()
    except Exception as exc:
        log.warning("No se pudo cargar snapshot Firebase en vivo: %s", exc)
        return {
            "context_text": "",
            "totals": {"salones": 0, "usuarios_objetivo": 0, "usuarios_all": 0},
            "error": str(exc),
            "schedule_details_requested": schedule_details_requested,
        }

    terms = _tokenize_for_match(question)
    snapshot_errors = snapshot.get("errors", []) or []
    error_text = "; ".join(str(err) for err in snapshot_errors if err)
    broad_query_markers = {"todos", "todas", "todo", "completo", "completa", "lista"}
    wants_broad = bool(terms & broad_query_markers)

    salones_all = snapshot.get("salones", [])
    usuarios_all = snapshot.get("usuarios", [])
    usuarios_all_collections = snapshot.get("usuarios_all", [])

    salones_scored = sorted(
        ((item, _score_record(item, terms)) for item in salones_all),
        key=lambda pair: pair[1],
        reverse=True,
    )
    usuarios_scored = sorted(
        ((item, _score_record(item, terms)) for item in usuarios_all),
        key=lambda pair: pair[1],
        reverse=True,
    )
    usuarios_all_collections_scored = sorted(
        ((item, _score_record(item, terms)) for item in usuarios_all_collections),
        key=lambda pair: pair[1],
        reverse=True,
    )

    salones_limit = 6 if wants_broad else max(1, max_salones)
    usuarios_limit = 6 if wants_broad else max(1, max_usuarios)
    usuarios_all_limit = 8 if wants_broad else 3

    selected_salones = [item for item, score in salones_scored if score > 0][:salones_limit]
    selected_usuarios = [item for item, score in usuarios_scored if score > 0][:usuarios_limit]
    selected_usuarios_all = [item for item, score in usuarios_all_collections_scored if score > 0][:usuarios_all_limit]

    if not selected_salones:
        selected_salones = [item for item, _ in salones_scored[: min(2, len(salones_scored))]]
    if not selected_usuarios:
        selected_usuarios = [item for item, _ in usuarios_scored[: min(2, len(usuarios_scored))]]
    if not selected_usuarios_all:
        selected_usuarios_all = [
            item
            for item, _ in usuarios_all_collections_scored[: min(2, len(usuarios_all_collections_scored))]
        ]

    compact_salones = [
        _compact_salon_runtime_payload(item, include_schedule=schedule_details_requested)
        for item in selected_salones
    ]
    compact_usuarios = [
        _compact_user_runtime_payload(item, include_schedule=schedule_details_requested)
        for item in selected_usuarios
    ]
    compact_usuarios_all = [
        _compact_generic_runtime_payload(item, include_schedule=schedule_details_requested)
        for item in selected_usuarios_all
    ]

    salones_json, salones_included, salones_total_selected = _serialize_items_with_char_budget(
        compact_salones,
        max_chars=4600,
    )
    usuarios_json, usuarios_included, usuarios_total_selected = _serialize_items_with_char_budget(
        compact_usuarios,
        max_chars=2600,
    )
    usuarios_all_json, usuarios_all_included, usuarios_all_total_selected = _serialize_items_with_char_budget(
        compact_usuarios_all,
        max_chars=2100,
    )

    context_text = (
        "DATOS_OPERATIVOS_UNIVERSIDAD\n"
        f"totales.salones={snapshot.get('totals', {}).get('salones', 0)}\n"
        f"totales.usuarios_objetivo={snapshot.get('totals', {}).get('usuarios_objetivo', 0)}\n"
        f"totales.usuarios_all={snapshot.get('totals', {}).get('usuarios_all', 0)}\n"
        f"solicitud_horario_detallado={str(schedule_details_requested).lower()}\n"
        f"salones_relevantes_incluidos={salones_included}/{salones_total_selected}\n"
        f"usuarios_relevantes_incluidos={usuarios_included}/{usuarios_total_selected}\n"
        f"usuarios_all_relevantes_incluidos={usuarios_all_included}/{usuarios_all_total_selected}\n"
        "salones_relevantes_json=\n"
        f"{salones_json}\n"
        "usuarios_relevantes_json=\n"
        f"{usuarios_json}\n"
        "usuarios_all_relevantes_json=\n"
        f"{usuarios_all_json}"
    )

    return {
        "context_text": context_text,
        "totals": snapshot.get("totals", {"salones": 0, "usuarios_objetivo": 0, "usuarios_all": 0}),
        "error": error_text,
        "schedule_details_requested": schedule_details_requested,
    }
