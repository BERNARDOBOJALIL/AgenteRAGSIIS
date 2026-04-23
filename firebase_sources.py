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
    "encargado"
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

DEFAULT_STUDENT_MARKERS = {
    "alumno",
    "alumna",
    "estudiante",
    "student",
}

DEFAULT_SALONES_COLLECTION_ALLOWLIST = ["salones"]
DEFAULT_USERS_COLLECTION_ALLOWLIST = [
    "usuarios",
    "horarios",
    "horrios",
]
DEFAULT_USERS_ALLOWED_LEAF_COLLECTIONS = {"usuarios", "horarios", "horrios"}

DEFAULT_CITAS_COLLECTION_CANDIDATES = ["citas", "cita", "appointments", "appointment"]
DEFAULT_NOTIFICACIONES_COLLECTION_CANDIDATES = [
    "notificaciones",
    "notificacion",
    "notifications",
    "notification",
]

DEFAULT_LOCAL_SALONES_SERVICE_ACCOUNT_FILE = "siis-d3571-336618a9b5d1.json"
DEFAULT_LOCAL_USERS_SERVICE_ACCOUNT_FILE = "siis-9593c-611399213026.json"

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

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

SCHEDULE_QUERY_MARKERS = {
    "que horario",
    "que horarios",
    "cual es el horario",
    "cuales son los horarios",
    "a que hora",
    "disponible",
    "disponibilidad",
    "abre",
    "cierra",
    "cuando",
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


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value)
        if not text:
            continue
        key = _normalize_lookup_token(text)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    return cleaned


def _collection_candidates(configured: str, defaults: list[str]) -> list[str]:
    return _dedupe_preserve_order([configured, *defaults])


def _collection_leaf_name(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    leaf = text.split("/")[-1]
    return _normalize_lookup_token(leaf)


def _resolve_path_candidates(path_value: str) -> list[str]:
    expanded = os.path.expandvars(os.path.expanduser(path_value))
    if os.path.isabs(expanded):
        return [expanded]
    return [
        expanded,
        os.path.join(MODULE_DIR, expanded),
    ]


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
    fallback_file_names: list[str] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Carga service account desde archivo o JSON/base64 en variables de entorno."""
    candidates: list[tuple[str | None, str]] = []
    if file_var:
        candidates.append((os.getenv(file_var, "").strip(), f"env:{file_var}"))

    for file_name in fallback_file_names or []:
        candidates.append((file_name, f"local_default:{file_name}"))

    candidates.append((os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip(), "env:GOOGLE_APPLICATION_CREDENTIALS"))

    for path, source in candidates:
        if not path:
            continue
        resolved_path = ""
        for candidate_path in _resolve_path_candidates(path):
            if os.path.exists(candidate_path):
                resolved_path = candidate_path
                break

        if not resolved_path:
            log.warning("Ruta de credencial no existe (%s): %s", source, path)
            continue
        try:
            with open(resolved_path, "r", encoding="utf-8") as fh:
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


def _int_env(name: str, default: int) -> int:
    """Lee un entero positivo del entorno."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default

    try:
        return max(1, int(raw.strip()))
    except ValueError:
        log.warning("Valor invalido para %s='%s'. Se usara default=%s", name, raw, default)
        return default


def _question_requests_schedule(question: str) -> bool:
    """Detecta si el usuario pidio informacion de horario/calendario."""
    normalized = _normalize_text(question)
    if not normalized:
        return False

    if any(marker in normalized for marker in SCHEDULE_LITERAL_MARKERS):
        return True

    if any(marker in normalized for marker in SCHEDULE_QUERY_MARKERS):
        return True

    return any(marker in normalized for marker in SCHEDULE_TOPIC_MARKERS)


def _question_requests_detailed_schedule(question: str) -> bool:
    """Detecta si el usuario pidio horario/calendario detallado de forma explicita."""
    normalized = _normalize_text(question)
    if not normalized:
        return False

    if any(marker in normalized for marker in SCHEDULE_LITERAL_MARKERS):
        return True

    has_topic = _question_requests_schedule(question)
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


def _schedule_presence_summary(
    horario: Any,
    calendario: Any,
    citas: Any,
    notificaciones: Any,
) -> str:
    parts: list[str] = []
    if horario:
        parts.append("horario registrado")
    if calendario:
        parts.append("calendario registrado")
    if citas:
        parts.append("citas registradas")
    if notificaciones:
        parts.append("notificaciones registradas")
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
    citas = item.get("citas")
    notificaciones = item.get("notificaciones")
    salones_relacionados = item.get("salones_relacionados") if isinstance(item.get("salones_relacionados"), list) else []
    salon_principal = item.get("salon_principal") if isinstance(item.get("salon_principal"), dict) else {}

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
        if citas:
            compact["citas"] = _compact_nested_payload(citas, max_items=12, max_depth=3)
        if notificaciones:
            compact["notificaciones"] = _compact_nested_payload(notificaciones, max_items=12, max_depth=3)
    else:
        compact["resumen_horario"] = _schedule_presence_summary(horario, calendario, citas, notificaciones)
        compact["tiene_horario"] = bool(horario)
        compact["tiene_calendario"] = bool(calendario)
        compact["tiene_citas"] = bool(citas)
        compact["tiene_notificaciones"] = bool(notificaciones)

    if salon_principal:
        compact["salon_principal"] = _compact_nested_payload(salon_principal, max_items=10, max_depth=2)
    if salones_relacionados:
        compact["salones_relacionados"] = [
            _compact_nested_payload(salon, max_items=10, max_depth=2)
            for salon in salones_relacionados[:3]
        ]

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


def _is_student_record(record: dict[str, Any]) -> bool:
    """Detecta registros de alumno/estudiante para excluirlos del contexto del agente."""
    flat = flatten_data(record)
    if not flat:
        return False

    student_markers = {
        _normalize_text(marker)
        for marker in _split_csv_env("FIREBASE_EXCLUDED_STUDENT_MARKERS", list(DEFAULT_STUDENT_MARKERS))
    }

    tipo = _normalize_text(record.get("tipo", ""))
    rol = _normalize_text(record.get("rol", ""))
    if any(marker in tipo for marker in student_markers):
        return True
    if any(marker in rol for marker in student_markers):
        return True

    for key, value in flat.items():
        key_norm = _normalize_text(key)
        value_norm = _normalize_text(value)

        key_is_user_related = any(
            token in key_norm
            for token in ("rol", "role", "tipo", "type", "perfil", "cargo", "puesto", "usuario", "user")
        )
        if key_is_user_related and any(marker in value_norm for marker in student_markers):
            return True

        if any(marker in key_norm for marker in student_markers):
            if value_norm not in {"", "false", "0", "no", "none", "null"}:
                return True

    return False


def _redact_student_branches(value: Any) -> Any:
    """Remueve ramas que parezcan datos de alumnos dentro de payloads anidados."""
    student_markers = {
        _normalize_text(marker)
        for marker in _split_csv_env("FIREBASE_EXCLUDED_STUDENT_MARKERS", list(DEFAULT_STUDENT_MARKERS))
    }

    if isinstance(value, dict):
        if _is_student_record(value):
            return {}

        cleaned_dict: dict[str, Any] = {}
        for key, item in value.items():
            key_norm = _normalize_text(key)
            if any(marker in key_norm for marker in student_markers):
                continue

            cleaned_item = _redact_student_branches(item)
            if cleaned_item in (None, "", [], {}):
                continue
            cleaned_dict[key] = cleaned_item

        return cleaned_dict

    if isinstance(value, list):
        cleaned_list: list[Any] = []
        for item in value:
            if isinstance(item, dict) and _is_student_record(item):
                continue

            cleaned_item = _redact_student_branches(item)
            if cleaned_item in (None, "", [], {}):
                continue
            cleaned_list.append(cleaned_item)

        return cleaned_list

    return value


def _is_professor_record(record: dict[str, Any]) -> bool:
    """Compatibilidad historica: ahora aplica filtro academico + tipo profesor/administrativo."""
    if _is_student_record(record):
        return False

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


def _merge_documents_for_collections(
    client: Any,
    collection_candidates: list[str],
) -> tuple[dict[str, FirestoreRecord], list[str]]:
    by_id: dict[str, FirestoreRecord] = {}
    detected: list[str] = []

    for collection_name in collection_candidates:
        try:
            current = client.get_documents_by_id(collection_name)
        except Exception as exc:
            log.warning("No se pudo leer la coleccion relacionada '%s': %s", collection_name, exc)
            continue

        if current:
            detected.append(collection_name)
            for doc_id, record in current.items():
                by_id.setdefault(doc_id, record)

    return by_id, detected


def _build_users_collection_config() -> dict[str, Any]:
    strict_core_only = _bool_env("FIREBASE_USERS_STRICT_CORE_ONLY", True)
    allowed_leaf_collections = set(DEFAULT_USERS_ALLOWED_LEAF_COLLECTIONS)

    if not strict_core_only:
        custom_allowed = {
            _collection_leaf_name(item)
            for item in _split_csv_env("FIREBASE_USERS_ALLOWED_COLLECTION_LEAVES")
            if _collection_leaf_name(item)
        }
        if custom_allowed:
            allowed_leaf_collections = custom_allowed

    def _filter_candidates(candidates: list[str]) -> list[str]:
        filtered: list[str] = []
        seen_leaves: set[str] = set()
        for candidate in candidates:
            leaf = _collection_leaf_name(candidate)
            if not leaf or leaf not in allowed_leaf_collections or leaf in seen_leaves:
                continue
            seen_leaves.add(leaf)
            filtered.append(candidate)
        return filtered

    usuarios_collection = _collection_env("FIREBASE_USUARIOS_COLLECTION", "usuarios")
    if _collection_leaf_name(usuarios_collection) not in allowed_leaf_collections:
        usuarios_collection = "usuarios"

    configured_horarios = _collection_env("FIREBASE_HORARIOS_COLLECTION", "horarios")
    configured_calendarios = _collection_env("FIREBASE_CALENDARIOS_COLLECTION", "calendarios")
    configured_citas = _collection_env("FIREBASE_CITAS_COLLECTION", "citas")
    configured_notificaciones = _collection_env("FIREBASE_NOTIFICACIONES_COLLECTION", "notificaciones")

    horarios_candidates = _filter_candidates(
        _collection_candidates(configured_horarios, ["horarios", "horrios"])
    )
    calendarios_candidates = _filter_candidates(
        _collection_candidates(configured_calendarios, ["calendarios", "calendario"])
    )
    citas_candidates = _filter_candidates(
        _collection_candidates(configured_citas, list(DEFAULT_CITAS_COLLECTION_CANDIDATES))
    )
    notificaciones_candidates = _filter_candidates(
        _collection_candidates(configured_notificaciones, list(DEFAULT_NOTIFICACIONES_COLLECTION_CANDIDATES))
    )

    return {
        "usuarios_collection": usuarios_collection,
        "configured_horarios": configured_horarios,
        "configured_calendarios": configured_calendarios,
        "configured_citas": configured_citas,
        "configured_notificaciones": configured_notificaciones,
        "horarios_candidates": horarios_candidates,
        "calendarios_candidates": calendarios_candidates,
        "citas_candidates": citas_candidates,
        "notificaciones_candidates": notificaciones_candidates,
        "allowed_leaf_collections": sorted(allowed_leaf_collections),
        "strict_core_only": strict_core_only,
    }


def _build_users_collections_bundle(client: Any) -> dict[str, Any]:
    config = _build_users_collection_config()
    usuarios_collection = config["usuarios_collection"]
    horarios_candidates = config["horarios_candidates"]
    calendarios_candidates = config["calendarios_candidates"]
    citas_candidates = config["citas_candidates"]
    notificaciones_candidates = config["notificaciones_candidates"]

    usuarios_by_id = client.get_documents_by_id(usuarios_collection)
    horarios_by_id, horarios_detectadas = _merge_documents_for_collections(client, horarios_candidates)
    calendarios_by_id, calendarios_detectadas = _merge_documents_for_collections(client, calendarios_candidates)
    citas_by_id, citas_detectadas = _merge_documents_for_collections(client, citas_candidates)
    notificaciones_by_id, notificaciones_detectadas = _merge_documents_for_collections(
        client,
        notificaciones_candidates,
    )

    return {
        **config,
        "usuarios_by_id": usuarios_by_id,
        "horarios_by_id": horarios_by_id,
        "calendarios_by_id": calendarios_by_id,
        "citas_by_id": citas_by_id,
        "notificaciones_by_id": notificaciones_by_id,
        "horarios_detectadas": horarios_detectadas,
        "calendarios_detectadas": calendarios_detectadas,
        "citas_detectadas": citas_detectadas,
        "notificaciones_detectadas": notificaciones_detectadas,
    }


def _extract_salon_lookup_keys(value: Any) -> set[str]:
    text = _clean_text(value)
    if not text:
        return set()

    keys: set[str] = set()
    text_key = _normalize_lookup_token(text)
    if text_key:
        keys.add(text_key)

    for match in J_CODE_PATTERN.finditer(text):
        try:
            keys.add(f"j{int(match.group(1))}")
        except ValueError:
            continue

    return keys


def _build_salones_lookup_indexes(
    salones_records: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    by_lookup_key: dict[str, list[dict[str, Any]]] = {}
    by_responsable: dict[str, list[dict[str, Any]]] = {}

    for item in salones_records:
        data = item.get("data") if isinstance(item.get("data"), dict) else {}

        item_keys: set[str] = set()
        for value in (
            item.get("doc_id"),
            data.get("nomenclatura"),
            data.get("codigo"),
            data.get("codigoSalon"),
            data.get("codigo_salon"),
            data.get("idConjunto"),
            data.get("nombre"),
        ):
            item_keys.update(_extract_salon_lookup_keys(value))

        for key in item_keys:
            by_lookup_key.setdefault(key, []).append(item)

        for responsable_name in _extract_responsable_names(data.get("responsables"), max_items=16):
            normalized = _normalize_text(responsable_name)
            if not normalized:
                continue
            by_responsable.setdefault(normalized, []).append(item)

    return by_lookup_key, by_responsable


def _collect_user_name_candidates(usuario_data: dict[str, Any]) -> list[str]:
    return _dedupe_preserve_order(
        [
            _clean_text(usuario_data.get("nombre")),
            _clean_text(usuario_data.get("name")),
            _clean_text(usuario_data.get("displayName")),
            _clean_text(usuario_data.get("responsable")),
        ]
    )


def _names_likely_match(a: str, b: str) -> bool:
    a_norm = _normalize_text(a)
    b_norm = _normalize_text(b)

    if not a_norm or not b_norm:
        return False
    if a_norm == b_norm:
        return True

    if len(a_norm) >= 8 and (a_norm in b_norm or b_norm in a_norm):
        return True

    a_tokens = {token for token in a_norm.split() if len(token) >= 3}
    b_tokens = {token for token in b_norm.split() if len(token) >= 3}
    return len(a_tokens.intersection(b_tokens)) >= 2


def _resolve_related_salones_for_user(
    *,
    doc_id: str,
    usuario_data: dict[str, Any],
    horario_data: dict[str, Any] | None,
    calendario_data: dict[str, Any] | None,
    citas_data: dict[str, Any] | None,
    notificaciones_data: dict[str, Any] | None,
    salones_by_lookup_key: dict[str, list[dict[str, Any]]],
    salones_by_responsable: dict[str, list[dict[str, Any]]],
    max_items: int = 3,
) -> list[dict[str, Any]]:
    lookup_keys: set[str] = set()

    for record in [usuario_data, horario_data, calendario_data, citas_data, notificaciones_data]:
        if not isinstance(record, dict):
            continue
        flat = flatten_data(record)
        for key, value in flat.items():
            value_text = _clean_text(value)
            if not value_text:
                continue

            key_norm = _normalize_text(key)
            value_keys = _extract_salon_lookup_keys(value_text)
            if not value_keys:
                continue

            if any(
                marker in key_norm
                for marker in ("salon", "aula", "nomenclatura", "codigo", "room", "laboratorio", "ubicacion")
            ):
                lookup_keys.update(value_keys)
            else:
                lookup_keys.update({candidate for candidate in value_keys if candidate.startswith("j")})

    lookup_keys.update(_extract_salon_lookup_keys(doc_id))

    related: list[dict[str, Any]] = []
    seen_doc_ids: set[str] = set()

    def _append_candidate(item: dict[str, Any]) -> None:
        salon_doc_id = _clean_text(item.get("doc_id"))
        if salon_doc_id and salon_doc_id in seen_doc_ids:
            return
        if salon_doc_id:
            seen_doc_ids.add(salon_doc_id)
        related.append(item)

    for key in lookup_keys:
        for item in salones_by_lookup_key.get(key, []):
            _append_candidate(item)
            if len(related) >= max_items:
                return related

    user_names = _collect_user_name_candidates(usuario_data)
    for user_name in user_names:
        user_norm = _normalize_text(user_name)
        if not user_norm:
            continue

        for responsable_name, items in salones_by_responsable.items():
            if not _names_likely_match(user_norm, responsable_name):
                continue
            for item in items:
                _append_candidate(item)
                if len(related) >= max_items:
                    return related

    return related


def _build_client_from_env(
    project_var: str,
    api_key_var: str,
    *,
    service_account_file_var: str | None = None,
    service_account_json_var: str | None = None,
    fallback_service_account_files: list[str] | None = None,
    admin_app_name: str | None = None,
) -> FirestoreRESTClient | FirestoreAdminClient | None:
    project_id = os.getenv(project_var, "").strip()
    api_key = os.getenv(api_key_var, "").strip()

    service_account, source = _service_account_from_env(
        file_var=service_account_file_var,
        json_var=service_account_json_var,
        fallback_file_names=fallback_service_account_files,
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
        fallback_service_account_files=[DEFAULT_LOCAL_SALONES_SERVICE_ACCOUNT_FILE],
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
    y relaciona horario + calendario + citas + notificaciones por ID de documento,
    ademas de salones donde opera.
    """
    client = _build_client_from_env(
        "VITE_FIREBASE_PROJECT_ID",
        "VITE_FIREBASE_API_KEY",
        service_account_file_var="FIREBASE_USERS_SERVICE_ACCOUNT_FILE",
        service_account_json_var="FIREBASE_USERS_SERVICE_ACCOUNT_JSON",
        fallback_service_account_files=[DEFAULT_LOCAL_USERS_SERVICE_ACCOUNT_FILE],
        admin_app_name="firebase_admin_users",
    )
    if client is None:
        return []

    bundle = _build_users_collections_bundle(client)
    usuarios_collection = bundle["usuarios_collection"]
    usuarios_by_id = bundle["usuarios_by_id"]

    horarios_by_id = bundle["horarios_by_id"]
    calendarios_by_id = bundle["calendarios_by_id"]
    citas_by_id = bundle["citas_by_id"]
    notificaciones_by_id = bundle["notificaciones_by_id"]

    try:
        salones_records = fetch_salones_raw_records()
    except Exception as exc:
        salones_records = []
        log.warning("No fue posible cargar salones para relacionar personal: %s", exc)

    salones_by_lookup_key, salones_by_responsable = _build_salones_lookup_indexes(salones_records)

    documentos_profesores: list[Document] = []
    profesores_con_horario = 0
    profesores_con_calendario = 0
    profesores_con_citas = 0
    profesores_con_notificaciones = 0
    profesores_con_salon = 0

    for doc_id, usuario_record in usuarios_by_id.items():
        if not _is_professor_record(usuario_record.data):
            continue

        horario_record = horarios_by_id.get(doc_id)
        horario_data = _redact_student_branches(horario_record.data) if horario_record else None
        if horario_data:
            profesores_con_horario += 1

        calendario_record = calendarios_by_id.get(doc_id)
        calendario_data = _redact_student_branches(calendario_record.data) if calendario_record else None
        if calendario_data:
            profesores_con_calendario += 1

        citas_record = citas_by_id.get(doc_id)
        citas_data = _redact_student_branches(citas_record.data) if citas_record else None
        if citas_data:
            profesores_con_citas += 1

        notificaciones_record = notificaciones_by_id.get(doc_id)
        notificaciones_data = _redact_student_branches(notificaciones_record.data) if notificaciones_record else None
        if notificaciones_data:
            profesores_con_notificaciones += 1

        salones_relacionados_raw = _resolve_related_salones_for_user(
            doc_id=doc_id,
            usuario_data=usuario_record.data,
            horario_data=horario_data if isinstance(horario_data, dict) else None,
            calendario_data=calendario_data if isinstance(calendario_data, dict) else None,
            citas_data=citas_data if isinstance(citas_data, dict) else None,
            notificaciones_data=notificaciones_data if isinstance(notificaciones_data, dict) else None,
            salones_by_lookup_key=salones_by_lookup_key,
            salones_by_responsable=salones_by_responsable,
            max_items=3,
        )
        salones_relacionados = [
            _compact_salon_runtime_payload(item, include_schedule=False)
            for item in salones_relacionados_raw
        ]
        if salones_relacionados:
            profesores_con_salon += 1

        record_data = dict(usuario_record.data)
        record_data["horario"] = horario_data
        record_data["calendario"] = calendario_data
        record_data["citas"] = citas_data
        record_data["notificaciones"] = notificaciones_data
        record_data["salones_relacionados"] = salones_relacionados or None
        record_data["salon_principal"] = salones_relacionados[0] if salones_relacionados else None

        metadata = {
            **usuario_record.metadata,
            "fuente": "firebase_usuarios",
            "categoria": "idit_personal",
            "es_usuario_academico_objetivo": True,
            "firebase_collection": usuarios_collection,
            "firebase_related_collection": ",".join(bundle["horarios_detectadas"]) if bundle["horarios_detectadas"] else bundle["configured_horarios"],
            "firebase_related_doc_id": doc_id,
            "horario_encontrado": bool(horario_data),
            "firebase_calendar_collection": ",".join(bundle["calendarios_detectadas"]) if bundle["calendarios_detectadas"] else bundle["configured_calendarios"],
            "calendario_encontrado": bool(calendario_data),
            "firebase_citas_collection": ",".join(bundle["citas_detectadas"]) if bundle["citas_detectadas"] else bundle["configured_citas"],
            "citas_encontradas": bool(citas_data),
            "firebase_notificaciones_collection": ",".join(bundle["notificaciones_detectadas"]) if bundle["notificaciones_detectadas"] else bundle["configured_notificaciones"],
            "notificaciones_encontradas": bool(notificaciones_data),
            "salones_relacionados_total": len(salones_relacionados),
        }
        documentos_profesores.append(firestore_record_to_document(record=record_data, metadata=metadata))

    log.info(
        "Firestore usuarios objetivo: usuarios=%s | con_horario=%s | con_calendario=%s | con_citas=%s | con_notificaciones=%s | con_salon=%s | docs_markdown=%s",
        len(documentos_profesores),
        profesores_con_horario,
        profesores_con_calendario,
        profesores_con_citas,
        profesores_con_notificaciones,
        profesores_con_salon,
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
        fallback_service_account_files=[DEFAULT_LOCAL_SALONES_SERVICE_ACCOUNT_FILE],
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


def fetch_target_users_raw_records(
    *,
    salones_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Obtiene usuarios academicos objetivo y vincula sus datos operativos por doc_id."""
    client = _build_client_from_env(
        "VITE_FIREBASE_PROJECT_ID",
        "VITE_FIREBASE_API_KEY",
        service_account_file_var="FIREBASE_USERS_SERVICE_ACCOUNT_FILE",
        service_account_json_var="FIREBASE_USERS_SERVICE_ACCOUNT_JSON",
        fallback_service_account_files=[DEFAULT_LOCAL_USERS_SERVICE_ACCOUNT_FILE],
        admin_app_name="firebase_admin_users",
    )
    if client is None:
        return []

    bundle = _build_users_collections_bundle(client)
    usuarios_collection = bundle["usuarios_collection"]
    usuarios_by_id = bundle["usuarios_by_id"]
    horarios_by_id = bundle["horarios_by_id"]
    calendarios_by_id = bundle["calendarios_by_id"]
    citas_by_id = bundle["citas_by_id"]
    notificaciones_by_id = bundle["notificaciones_by_id"]

    if salones_records is None:
        try:
            salones_records = fetch_salones_raw_records()
        except Exception as exc:
            salones_records = []
            log.warning("No fue posible cargar salones para relacionar usuarios objetivo: %s", exc)

    salones_by_lookup_key, salones_by_responsable = _build_salones_lookup_indexes(salones_records)

    payloads: list[dict[str, Any]] = []
    for doc_id, usuario_record in usuarios_by_id.items():
        if not _is_professor_record(usuario_record.data):
            continue

        horario_record = horarios_by_id.get(doc_id)
        calendario_record = calendarios_by_id.get(doc_id)
        citas_record = citas_by_id.get(doc_id)
        notificaciones_record = notificaciones_by_id.get(doc_id)

        horario_data = _redact_student_branches(horario_record.data) if horario_record else None
        calendario_data = _redact_student_branches(calendario_record.data) if calendario_record else None
        citas_data = _redact_student_branches(citas_record.data) if citas_record else None
        notificaciones_data = _redact_student_branches(notificaciones_record.data) if notificaciones_record else None

        salones_relacionados_raw = _resolve_related_salones_for_user(
            doc_id=doc_id,
            usuario_data=usuario_record.data,
            horario_data=horario_data if isinstance(horario_data, dict) else None,
            calendario_data=calendario_data if isinstance(calendario_data, dict) else None,
            citas_data=citas_data if isinstance(citas_data, dict) else None,
            notificaciones_data=notificaciones_data if isinstance(notificaciones_data, dict) else None,
            salones_by_lookup_key=salones_by_lookup_key,
            salones_by_responsable=salones_by_responsable,
            max_items=3,
        )
        salones_relacionados = [
            _compact_salon_runtime_payload(item, include_schedule=False)
            for item in salones_relacionados_raw
        ]

        payloads.append(
            {
                "source": "firebase_usuarios",
                "project": usuario_record.metadata.get("firebase_project"),
                "collection": usuarios_collection,
                "doc_id": doc_id,
                "usuario": usuario_record.data,
                "horario": horario_data,
                "calendario": calendario_data,
                "citas": citas_data,
                "notificaciones": notificaciones_data,
                "salones_relacionados": salones_relacionados,
                "salon_principal": salones_relacionados[0] if salones_relacionados else None,
                "horario_collection": horario_record.metadata.get("firebase_collection") if horario_record else None,
                "calendario_collection": calendario_record.metadata.get("firebase_collection") if calendario_record else None,
                "citas_collection": citas_record.metadata.get("firebase_collection") if citas_record else None,
                "notificaciones_collection": notificaciones_record.metadata.get("firebase_collection") if notificaciones_record else None,
            }
        )

    return payloads


def fetch_users_all_raw_records(*, target_user_ids: set[str] | None = None) -> list[dict[str, Any]]:
    """Obtiene toda la informacion del proyecto de usuarios en formato crudo."""
    client = _build_client_from_env(
        "VITE_FIREBASE_PROJECT_ID",
        "VITE_FIREBASE_API_KEY",
        service_account_file_var="FIREBASE_USERS_SERVICE_ACCOUNT_FILE",
        service_account_json_var="FIREBASE_USERS_SERVICE_ACCOUNT_JSON",
        fallback_service_account_files=[DEFAULT_LOCAL_USERS_SERVICE_ACCOUNT_FILE],
        admin_app_name="firebase_admin_users",
    )
    if client is None:
        return []

    config = _build_users_collection_config()
    allowed_leaf_collections = set(config.get("allowed_leaf_collections") or DEFAULT_USERS_ALLOWED_LEAF_COLLECTIONS)
    normalized_target_user_ids = {
        _clean_text(doc_id)
        for doc_id in (target_user_ids or set())
        if _clean_text(doc_id)
    }
    apply_target_filter = bool(normalized_target_user_ids)

    related_leaf_collections = {
        _collection_leaf_name(config["usuarios_collection"]),
        *(_collection_leaf_name(name) for name in config["horarios_candidates"]),
        *(_collection_leaf_name(name) for name in config["calendarios_candidates"]),
        *(_collection_leaf_name(name) for name in config["citas_candidates"]),
        *(_collection_leaf_name(name) for name in config["notificaciones_candidates"]),
    }
    related_leaf_collections = {name for name in related_leaf_collections if name}
    usuarios_leaf = _collection_leaf_name(config["usuarios_collection"])

    include_subcollections = _bool_env("FIREBASE_USERS_INCLUDE_SUBCOLLECTIONS", True)
    max_depth = _optional_int_env("FIREBASE_USERS_MAX_TRAVERSAL_DEPTH", None)
    collection_allowlist = _collection_allowlist_env(
        "FIREBASE_USERS_COLLECTION_ALLOWLIST",
        "FIREBASE_USERS_COLLECTIONS",
    )
    if not collection_allowlist:
        collection_allowlist = list(DEFAULT_USERS_COLLECTION_ALLOWLIST)
    else:
        filtered_allowlist: list[str] = []
        for path in collection_allowlist:
            if _collection_leaf_name(path) in allowed_leaf_collections:
                filtered_allowlist.append(path)
        collection_allowlist = filtered_allowlist or list(DEFAULT_USERS_COLLECTION_ALLOWLIST)

    records: list[dict[str, Any]] = []
    for record in client.iter_documents(
        collection_allowlist=collection_allowlist,
        include_subcollections=include_subcollections,
        max_depth=max_depth,
    ):
        payload = _record_payload(record, "firebase_users_all")
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        doc_id = _clean_text(payload.get("doc_id"))
        collection_leaf = _collection_leaf_name(payload.get("collection"))

        if collection_leaf and collection_leaf not in allowed_leaf_collections:
            continue

        if apply_target_filter:
            if _is_student_record(data):
                continue

            if collection_leaf in related_leaf_collections and doc_id not in normalized_target_user_ids:
                continue

            if collection_leaf == usuarios_leaf and doc_id not in normalized_target_user_ids:
                continue

        records.append(payload)

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
        usuarios = fetch_target_users_raw_records(salones_records=salones)
    except Exception as exc:
        errors.append(f"usuarios: {exc}")
        log.warning("No fue posible cargar usuarios en vivo: %s", exc)

    users_all_filter_to_targets = _bool_env("FIREBASE_USERS_ALL_FILTER_TO_TARGETS", False)
    target_user_ids: set[str] | None = None
    if users_all_filter_to_targets:
        target_user_ids = {
            _clean_text(item.get("doc_id"))
            for item in usuarios
            if isinstance(item, dict) and _clean_text(item.get("doc_id"))
        }

    try:
        usuarios_all = fetch_users_all_raw_records(target_user_ids=target_user_ids)
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


def _normalize_lookup_token(value: Any) -> str:
    return _normalize_text(value).replace(" ", "")


def _parse_frontend_context(frontend_context: str | None) -> dict[str, Any]:
    """Parsea el payload SIIS_FRONTEND_CONTEXT enviado por el frontend."""
    raw = str(frontend_context or "").strip()
    if not raw:
        return {}

    if raw.startswith("SIIS_FRONTEND_CONTEXT"):
        _, sep, tail = raw.partition("\n")
        raw = tail.strip() if sep else ""

    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
    except Exception:
        return {}

    return parsed if isinstance(parsed, dict) else {}


def _extract_frontend_candidate_labels(front_payload: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()

    def _push(value: Any) -> bool:
        text = _clean_text(value)
        if not text:
            return False
        key = _normalize_lookup_token(text)
        if not key or key in seen:
            return False
        seen.add(key)
        labels.append(text)
        return True

    route_guidance = front_payload.get("route_guidance")
    if isinstance(route_guidance, dict):
        has_destination = _push(route_guidance.get("destination"))
        if not has_destination:
            _push(route_guidance.get("origin"))

    selected = front_payload.get("last_selected_salon")
    if isinstance(selected, dict):
        _push(selected.get("nomenclatura"))
        _push(selected.get("nombre"))
        _push(selected.get("name"))
        _push(selected.get("rawName"))

    relevant = front_payload.get("relevant_salones")
    if isinstance(relevant, list):
        for item in relevant:
            if not isinstance(item, dict):
                continue
            _push(item.get("nomenclatura"))
            _push(item.get("nombre"))

    return labels


def _build_front_route_summary(route_guidance: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    origin = _clean_text(route_guidance.get("origin"))
    destination = _clean_text(route_guidance.get("destination"))
    floor = _clean_text(route_guidance.get("floor"))

    if origin:
        summary["origen"] = origin
    if destination:
        summary["destino"] = destination
    if floor:
        summary["piso_ruta"] = floor

    left_turns = route_guidance.get("left_turns")
    right_turns = route_guidance.get("right_turns")
    if isinstance(left_turns, (int, float)):
        summary["giros_izquierda"] = int(left_turns)
    if isinstance(right_turns, (int, float)):
        summary["giros_derecha"] = int(right_turns)

    direction_steps = route_guidance.get("direction_steps")
    if isinstance(direction_steps, list):
        cleaned_steps = [_clean_text(step) for step in direction_steps if _clean_text(step)]
        if cleaned_steps:
            summary["pasos_direccion"] = cleaned_steps[:4]

    steps = route_guidance.get("steps")
    if isinstance(steps, list):
        cleaned_steps = [_clean_text(step) for step in steps if _clean_text(step)]
        if cleaned_steps:
            summary["pasos_ruta"] = cleaned_steps[:6]

    return summary


def _lateral_hint_from_route_summary(route_summary: dict[str, Any]) -> str:
    steps = route_summary.get("pasos_direccion")
    joined_steps = _normalize_text(" ".join(steps)) if isinstance(steps, list) else ""

    has_left = "izquierda" in joined_steps
    has_right = "derecha" in joined_steps

    left_turns = int(route_summary.get("giros_izquierda", 0) or 0)
    right_turns = int(route_summary.get("giros_derecha", 0) or 0)

    if has_left and not has_right:
        return "hacia la izquierda"
    if has_right and not has_left:
        return "hacia la derecha"
    if has_left and has_right:
        return "siguiendo giros a izquierda y derecha"

    if right_turns > left_turns and right_turns > 0:
        return "hacia la derecha"
    if left_turns > right_turns and left_turns > 0:
        return "hacia la izquierda"
    return ""


def _zone_hint_from_location_text(location_text: Any) -> str:
    normalized = _normalize_text(location_text)
    if not normalized:
        return ""

    if "fondo" in normalized:
        return "al fondo"
    if "zona media" in normalized or "media del pasillo" in normalized:
        return "a media altura del pasillo"
    if "acceso principal" in normalized or "entrada" in normalized:
        return "cerca de la entrada"
    if "escalera" in normalized:
        return "cerca de la escalera"
    return ""


def _build_simple_reference_for_salon(route_summary: dict[str, Any], salon_payload: dict[str, Any]) -> str:
    if not isinstance(salon_payload, dict):
        return ""

    origin = _clean_text(route_summary.get("origen"))
    lateral = _lateral_hint_from_route_summary(route_summary)
    zone_hint = _zone_hint_from_location_text(salon_payload.get("ubicacion_aproximada"))
    floor = _clean_text(salon_payload.get("piso"))

    guidance_parts = [part for part in [lateral, zone_hint] if part]
    guidance = ", ".join(guidance_parts)

    main = ""
    if origin and guidance:
        main = f"Desde {origin}, avanza por el pasillo y sigue {guidance}."
    elif origin:
        main = f"Desde {origin}, avanza por el pasillo hasta la zona indicada."
    elif guidance:
        main = f"Ubicalo {guidance}."

    if floor:
        if main:
            main += f" Esta en {floor}."
        else:
            main = f"Esta en {floor}."

    return main.strip()


def _build_frontend_route_db_cross_context(
    front_payload: dict[str, Any],
    salones_all: list[dict[str, Any]],
    *,
    include_schedule: bool,
    max_matches: int = 3,
) -> dict[str, Any]:
    """Cruza contexto de ruta del frontend con salones de la BD para respuestas mas precisas."""
    if not front_payload:
        return {}

    route_guidance = front_payload.get("route_guidance") if isinstance(front_payload.get("route_guidance"), dict) else {}
    candidates = _extract_frontend_candidate_labels(front_payload)
    candidate_norms = [_normalize_lookup_token(label) for label in candidates if _normalize_lookup_token(label)]
    candidate_terms = _tokenize_for_match(" ".join(candidates))

    scored: list[tuple[int, dict[str, Any]]] = []
    for item in salones_all:
        data = item.get("data") if isinstance(item.get("data"), dict) else {}
        nomenclatura = _clean_text(data.get("nomenclatura"))
        nombre = _clean_text(data.get("nombre"))
        nom_norm = _normalize_lookup_token(nomenclatura)
        name_norm = _normalize_lookup_token(nombre)
        blob = " ".join(part for part in [nom_norm, name_norm] if part)
        if not blob:
            continue

        score = 0
        for cand in candidate_norms:
            if cand == nom_norm or cand == name_norm:
                score += 10
            elif cand and (cand in blob or blob in cand):
                score += 5

        for term in candidate_terms:
            if term in blob:
                score += 1

        if score > 0:
            scored.append((score, item))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    matched_items: list[dict[str, Any]] = []
    seen_doc_ids: set[str] = set()
    for _, item in scored:
        doc_id = _clean_text(item.get("doc_id"))
        if doc_id and doc_id in seen_doc_ids:
            continue
        if doc_id:
            seen_doc_ids.add(doc_id)
        matched_items.append(item)
        if len(matched_items) >= max_matches:
            break

    cross_context: dict[str, Any] = {}
    front_floor = _clean_text(front_payload.get("active_floor"))
    if front_floor:
        cross_context["piso_activo_front"] = front_floor

    route_summary = _build_front_route_summary(route_guidance) if route_guidance else {}
    if route_summary:
        cross_context["ruta_front"] = route_summary

    if matched_items:
        matched_compact = [
            _compact_salon_runtime_payload(item, include_schedule=include_schedule)
            for item in matched_items
        ]
        cross_context["salones_ruta_cruzados"] = matched_compact

        referencias = []
        for salon in matched_compact:
            referencia = _build_simple_reference_for_salon(route_summary, salon)
            if not referencia:
                continue
            referencias.append(
                {
                    "salon": _clean_text(salon.get("nomenclatura") or salon.get("nombre") or salon.get("doc_id")),
                    "referencia_simple": referencia,
                }
            )

        if referencias:
            cross_context["referencias_textuales_sugeridas"] = referencias

    return cross_context


def _serialize_object_with_char_budget(payload: dict[str, Any], max_chars: int) -> str:
    """Serializa un objeto JSON sin exceder presupuesto; recorta listas grandes si hace falta."""
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(encoded) <= max_chars:
        return encoded

    compact = dict(payload)
    salones = compact.get("salones_ruta_cruzados")
    if isinstance(salones, list) and len(salones) > 1:
        compact["salones_ruta_cruzados"] = salones[:2]
        compact["truncated"] = True
        encoded = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        if len(encoded) <= max_chars:
            return encoded

    route = compact.get("ruta_front")
    if isinstance(route, dict) and "pasos_ruta" in route:
        route = dict(route)
        pasos = route.get("pasos_ruta")
        if isinstance(pasos, list):
            route["pasos_ruta"] = pasos[:3]
        compact["ruta_front"] = route
        compact["truncated"] = True

    encoded = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    if len(encoded) <= max_chars:
        return encoded

    return json.dumps({"truncated": True}, ensure_ascii=False, separators=(",", ":"))


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


def build_firebase_runtime_context(
    question: str,
    *,
    max_salones: int = 10,
    max_usuarios: int = 10,
    frontend_context: str | None = None,
) -> dict[str, Any]:
    """Construye contexto Firebase en vivo (sin Markdown) relevante para la pregunta."""
    schedule_requested = _question_requests_schedule(question)
    schedule_details_requested = _question_requests_detailed_schedule(question)
    include_schedule_in_context = schedule_details_requested
    front_payload = _parse_frontend_context(frontend_context)

    try:
        snapshot = fetch_firebase_live_snapshot()
    except Exception as exc:
        log.warning("No se pudo cargar snapshot Firebase en vivo: %s", exc)
        return {
            "context_text": "",
            "totals": {"salones": 0, "usuarios_objetivo": 0, "usuarios_all": 0},
            "error": str(exc),
            "schedule_requested": schedule_requested,
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

    max_salones_context = _int_env("FIREBASE_RUNTIME_MAX_SALONES_CONTEXT", 8)
    max_usuarios_context = _int_env("FIREBASE_RUNTIME_MAX_USUARIOS_CONTEXT", 8)
    max_usuarios_all_context = _int_env("FIREBASE_RUNTIME_MAX_USUARIOS_ALL_CONTEXT", 12)

    salones_limit = max(max_salones_context, max(1, max_salones))
    usuarios_limit = max(max_usuarios_context, max(1, max_usuarios))
    usuarios_all_limit = max(6, max_usuarios_all_context)

    if wants_broad:
        salones_limit = max(salones_limit, max_salones_context * 2)
        usuarios_limit = max(usuarios_limit, max_usuarios_context * 2)
        usuarios_all_limit = max(usuarios_all_limit, max_usuarios_all_context * 2)

    selected_salones = [item for item, score in salones_scored if score > 0][:salones_limit]
    selected_usuarios = [item for item, score in usuarios_scored if score > 0][:usuarios_limit]
    selected_usuarios_all = [item for item, score in usuarios_all_collections_scored if score > 0][:usuarios_all_limit]

    if not selected_salones:
        selected_salones = [item for item, _ in salones_scored[: min(salones_limit, len(salones_scored))]]
    if not selected_usuarios:
        selected_usuarios = [item for item, _ in usuarios_scored[: min(usuarios_limit, len(usuarios_scored))]]
    if not selected_usuarios_all:
        selected_usuarios_all = [
            item
            for item, _ in usuarios_all_collections_scored[: min(usuarios_all_limit, len(usuarios_all_collections_scored))]
        ]

    compact_salones = [
        _compact_salon_runtime_payload(item, include_schedule=include_schedule_in_context)
        for item in selected_salones
    ]
    compact_usuarios = [
        _compact_user_runtime_payload(item, include_schedule=include_schedule_in_context)
        for item in selected_usuarios
    ]
    compact_usuarios_all = [
        _compact_generic_runtime_payload(item, include_schedule=include_schedule_in_context)
        for item in selected_usuarios_all
    ]

    front_route_db_cross = _build_frontend_route_db_cross_context(
        front_payload,
        salones_all,
        include_schedule=include_schedule_in_context,
    )
    front_route_db_cross_json = _serialize_object_with_char_budget(
        front_route_db_cross,
        max_chars=_int_env("FIREBASE_RUNTIME_MAX_CHARS_FRONT_ROUTE_DB_CROSS", 1600),
    )

    salones_json, salones_included, salones_total_selected = _serialize_items_with_char_budget(
        compact_salones,
        max_chars=_int_env("FIREBASE_RUNTIME_MAX_CHARS_SALONES", 4800),
    )
    usuarios_json, usuarios_included, usuarios_total_selected = _serialize_items_with_char_budget(
        compact_usuarios,
        max_chars=_int_env("FIREBASE_RUNTIME_MAX_CHARS_USUARIOS", 2800),
    )
    usuarios_all_json, usuarios_all_included, usuarios_all_total_selected = _serialize_items_with_char_budget(
        compact_usuarios_all,
        max_chars=_int_env("FIREBASE_RUNTIME_MAX_CHARS_USUARIOS_ALL", 2600),
    )

    context_text = (
        "DATOS_OPERATIVOS_UNIVERSIDAD\n"
        f"totales.salones={snapshot.get('totals', {}).get('salones', 0)}\n"
        f"totales.usuarios_objetivo={snapshot.get('totals', {}).get('usuarios_objetivo', 0)}\n"
        f"totales.usuarios_all={snapshot.get('totals', {}).get('usuarios_all', 0)}\n"
        "politica.excluir_usuarios_alumno=true\n"
        f"solicitud_horario={str(schedule_requested).lower()}\n"
        f"solicitud_horario_detallado={str(schedule_details_requested).lower()}\n"
        f"salones_relevantes_incluidos={salones_included}/{salones_total_selected}\n"
        f"usuarios_relevantes_incluidos={usuarios_included}/{usuarios_total_selected}\n"
        f"usuarios_all_relevantes_incluidos={usuarios_all_included}/{usuarios_all_total_selected}\n"
        "salones_relevantes_json=\n"
        f"{salones_json}\n"
        "usuarios_relevantes_json=\n"
        f"{usuarios_json}\n"
        "cruce_front_mapa_salones_json=\n"
        f"{front_route_db_cross_json}\n"
        "usuarios_all_relevantes_json=\n"
        f"{usuarios_all_json}"
    )

    return {
        "context_text": context_text,
        "totals": snapshot.get("totals", {"salones": 0, "usuarios_objetivo": 0, "usuarios_all": 0}),
        "error": error_text,
        "schedule_requested": schedule_requested,
        "schedule_details_requested": schedule_details_requested,
    }
