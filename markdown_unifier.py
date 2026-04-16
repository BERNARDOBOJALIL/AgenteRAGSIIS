"""
markdown_unifier.py
Utilidades para transformar distintas fuentes de datos a un formato Markdown unificado.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

log = logging.getLogger(__name__)

MISSING_INFO_TEXT = "No poseemos esa infromacion."
UNIFIED_FORMAT_VERSION = "markdown_unificado_v1"

CONSULTA_GENERAL = "Informacion general de IDIT"
CONSULTA_PERSONAL = "Informacion sobre personal, equipamiento y ubicaciones dentro del IDIT"
CONSULTA_SERVICIOS = "Informacion sobre servicios del IDIT"

_WEB_NOISE_PATTERNS = [
    r"\bcookies?\b",
    r"\baceptar(\s+todas)?\b",
    r"\baviso\s+de\s+privacidad\b",
    r"\bpolitica\s+de\s+privacidad\b",
    r"\bterminos\s+y\s+condiciones\b",
    r"\binicio\b",
    r"\bmenu\b",
    r"\bfacebook\b",
    r"\binstagram\b",
    r"\blinkedin\b",
    r"\byoutube\b",
    r"\bpasar\s+al\s+contenido\s+principal\b",
    r"\bmain\s+navigation\b",
    r"\bmen[uú]\b",
    r"\badmisiones\b",
    r"\baccesos\b",
    r"\boferta\s+acad[eé]mica\b",
    r"\blicenciaturas\b",
    r"\bposgrados\b",
    r"\bpreparatoria\b",
    r"\beducaci[oó]n\s+continua\b",
    r"\blenguas\b",
    r"\bcon[oó]cenos\b",
    r"\bvida\s+universitaria\b",
    r"\bdifusi[oó]n\s+y\s+medios\b",
    r"\bprocuraci[oó]n\s+de\s+fondos\b",
    r"\bnormativa\s+de\s+la\s+ibero\s+puebla\b",
    r"\bprotecci[oó]n\s+civil\b",
    r"\brecorridos\s+virtuales\b",
]

_NAV_EXACT_LINES = {
    "main navigation",
    "pasar al contenido principal",
    "menu",
    "menú",
    "accesos",
    "admisiones",
    "oferta academica",
    "oferta académica",
    "licenciaturas",
    "posgrados",
    "preparatoria",
    "educacion continua",
    "educación continua",
    "lenguas",
    "conocenos",
    "conócenos",
    "campus",
    "servicios",
    "vida universitaria",
    "difusion y medios",
    "difusión y medios",
    "procuracion de fondos",
    "procuración de fondos",
}

_NAV_WORDS = {
    "menu",
    "menú",
    "inicio",
    "campus",
    "servicios",
    "admisiones",
    "accesos",
    "licenciaturas",
    "posgrados",
    "preparatoria",
    "lenguas",
    "conocenos",
    "conócenos",
    "difusion",
    "difusión",
    "medios",
}

_SIGNAL_KEYWORDS = {
    "idit",
    "laboratorio",
    "laboratorios",
    "fabricacion",
    "fabricación",
    "digital",
    "fablab",
    "innovacion",
    "innovación",
    "economia social",
    "economía social",
    "academy",
    "vinculacion",
    "vinculación",
    "empresarial",
    "maquinaria",
    "equipamiento",
    "emprende",
    "comunidad",
    "modeva",
}

_SERVICE_KEYWORDS = {
    "servicio",
    "servicios",
    "atencion",
    "asesoria",
    "soporte",
    "prestamo",
    "tramite",
    "vinculacion",
    "academy",
}

_PERSONAL_OR_LOCATION_KEYWORDS = {
    "profesor",
    "docente",
    "maestro",
    "personal",
    "encargado",
    "coordinador",
    "equipo",
    "equipamiento",
    "maquinaria",
    "laboratorio",
    "laboratorios",
    "salon",
    "salones",
    "aula",
    "ubicacion",
    "edificio",
    "planta",
    "piso",
}

_SECTION_KEYWORDS = {
    "personal": {"profesor", "docente", "maestro", "personal", "encargado", "coordinador", "usuario", "email", "telefono"},
    "equipamiento": {"equipo", "equipamiento", "maquina", "maquinaria", "herramienta", "tecnologia", "laboratorio"},
    "ubicacion": {"salon", "aula", "ubicacion", "edificio", "planta", "piso", "mapa", "zona", "campus"},
    "servicios": {"servicio", "servicios", "atencion", "asesoria", "prestamo", "tramite", "vinculacion", "academy"},
}


def clean_text(value: Any, *, filter_noise: bool = True, dedupe_lines: bool = True) -> str:
    """Limpia texto y opcionalmente elimina ruido tipico de scraping."""
    if value is None:
        return ""

    text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\xa0", " ")

    lines = []
    for line in text.split("\n"):
        normalized = re.sub(r"\s+", " ", line).strip()
        if not normalized:
            continue
        if filter_noise and _is_noise_line(normalized):
            continue
        lines.append(normalized)

    if dedupe_lines:
        deduped: list[str] = []
        seen = set()
        for line in lines:
            fingerprint = line.lower()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            deduped.append(line)
        lines = deduped

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _light_clean_text(value: Any) -> str:
    """Limpieza suave para metadatos sin descartar contenido corto."""
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _line_has_signal(lowered_line: str) -> bool:
    return any(keyword in lowered_line for keyword in _SIGNAL_KEYWORDS)


def _is_noise_line(line: str) -> bool:
    lowered = line.lower().strip(" -•|:;,.\t")
    if not lowered:
        return True

    if any(re.search(pattern, lowered) for pattern in _WEB_NOISE_PATTERNS):
        return True

    if lowered in _NAV_EXACT_LINES:
        return True

    words = re.findall(r"[a-z0-9áéíóúüñ]+", lowered)
    if not words:
        return True

    # Etiquetas cortas de navegacion no aportan al RAG.
    if len(words) <= 2 and not _line_has_signal(lowered):
        return True

    if len(words) <= 4 and not _line_has_signal(lowered):
        nav_hits = sum(1 for word in words if word in _NAV_WORDS)
        if nav_hits >= max(2, len(words) - 1):
            return True

    # Titulos de pagina repetitivos de branding tampoco aportan.
    if "| ibero puebla" in lowered and len(words) <= 14:
        return True

    return False


def is_document_useful(raw_text: str) -> bool:
    """Evalua si un chunk contiene informacion suficiente para indexar."""
    cleaned = clean_text(raw_text, filter_noise=True, dedupe_lines=True)
    if not cleaned:
        return False

    words = re.findall(r"[a-z0-9áéíóúüñ]+", cleaned.lower())
    if len(words) < 8 and not _line_has_signal(cleaned.lower()):
        return False

    lines = [ln for ln in cleaned.split("\n") if ln.strip()]
    if len(lines) < 2 and len(cleaned) < 90:
        return False

    return True


def flatten_data(data: Any, prefix: str = "") -> dict[str, Any]:
    """Aplana diccionarios y listas para poder tabular datos en Markdown."""
    flat: dict[str, Any] = {}

    if isinstance(data, dict):
        for key, value in data.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, (dict, list)):
                flat.update(flatten_data(value, next_prefix))
            else:
                flat[next_prefix] = value
        return flat

    if isinstance(data, list):
        for idx, item in enumerate(data):
            next_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            if isinstance(item, (dict, list)):
                flat.update(flatten_data(item, next_prefix))
            else:
                flat[next_prefix] = item
        return flat

    if prefix:
        flat[prefix] = data
    return flat


def _safe_text(value: Any) -> str:
    if value is None:
        return MISSING_INFO_TEXT
    if isinstance(value, bool):
        return "Si" if value else "No"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, dict)):
        serialized = json.dumps(value, ensure_ascii=False)
        return _light_clean_text(serialized) or MISSING_INFO_TEXT
    cleaned = _light_clean_text(value)
    return cleaned if cleaned else MISSING_INFO_TEXT


def _select_title(metadata: dict[str, Any], fallback: str) -> str:
    for key in ("titulo", "title", "nombre", "name", "firebase_doc_id", "url", "source_url"):
        value = metadata.get(key)
        if value:
            selected = _light_clean_text(value)
            if selected:
                return selected
    return fallback


def _summarize(text: str, limit: int = 5000) -> str:
    if not text:
        return MISSING_INFO_TEXT
    one_line = re.sub(r"\s+", " ", text).strip()
    if not one_line:
        return MISSING_INFO_TEXT
    if len(one_line) <= limit:
        return one_line
    return f"{one_line[:limit].rstrip()}..."


def infer_tipo_consulta(main_text: str, metadata: dict[str, Any] | None = None) -> str:
    """Clasifica el documento en uno de los 3 tipos de consulta requeridos."""
    metadata = metadata or {}
    blob_parts = [main_text]
    blob_parts.extend(str(v) for v in metadata.values())
    blob = clean_text(" ".join(blob_parts)).lower()

    if any(keyword in blob for keyword in _SERVICE_KEYWORDS):
        return CONSULTA_SERVICIOS

    if any(keyword in blob for keyword in _PERSONAL_OR_LOCATION_KEYWORDS):
        return CONSULTA_PERSONAL

    return CONSULTA_GENERAL


def _build_section(flat_data: dict[str, Any], section_key: str, limit: int = 6) -> str:
    keywords = _SECTION_KEYWORDS[section_key]
    matches = []
    for key, value in flat_data.items():
        lowered_key = key.lower()
        if any(keyword in lowered_key for keyword in keywords):
            matches.append((key, _safe_text(value)))

    if not matches:
        return MISSING_INFO_TEXT

    lines = [f"- {key}: {value}" for key, value in matches[:limit]]
    return "\n".join(lines)


def _build_structured_table(flat_data: dict[str, Any], limit: int = 30) -> str:
    if not flat_data:
        return f"| Campo | Valor |\n| --- | --- |\n| informacion_disponible | {MISSING_INFO_TEXT} |"

    rows = ["| Campo | Valor |", "| --- | --- |"]
    count = 0
    for key, value in flat_data.items():
        count += 1
        if count > limit:
            rows.append("| observacion | Se omitieron campos para mantener legibilidad. |")
            break
        safe_key = _light_clean_text(key).replace("|", "/")
        safe_value = _safe_text(value).replace("|", "/")
        rows.append(f"| {safe_key} | {safe_value} |")

    return "\n".join(rows)


def build_unified_markdown(
    *,
    title: str,
    main_text: str,
    metadata: dict[str, Any] | None = None,
    structured_data: dict[str, Any] | None = None,
    tipo_consulta: str | None = None,
    preserve_full_text: bool = False,
) -> tuple[str, str]:
    """Construye un documento Markdown consistente para ingesta RAG."""
    metadata = metadata or {}
    structured_data = structured_data or {}

    cleaned_text = clean_text(
        main_text,
        filter_noise=not preserve_full_text,
        dedupe_lines=not preserve_full_text,
    )
    normalized_title = _light_clean_text(title) or MISSING_INFO_TEXT
    detected_tipo = tipo_consulta or infer_tipo_consulta(cleaned_text, metadata)

    flat_data = flatten_data(structured_data)

    fuente_lines = [
        f"- origen: {_safe_text(metadata.get('fuente', 'desconocida'))}",
        f"- categoria: {_safe_text(metadata.get('categoria', 'sin_categoria'))}",
        f"- proyecto: {_safe_text(metadata.get('firebase_project'))}",
        f"- coleccion: {_safe_text(metadata.get('firebase_collection'))}",
        f"- documento_id: {_safe_text(metadata.get('firebase_doc_id'))}",
        f"- url: {_safe_text(metadata.get('url') or metadata.get('source_url'))}",
        f"- fecha_normalizacion_utc: {_safe_text(metadata.get('fecha_normalizacion_utc'))}",
    ]

    markdown = (
        "# Ficha de conocimiento IDIT\n\n"
        "## Tipo de consulta\n"
        f"{detected_tipo}\n\n"
        "## Titulo\n"
        f"{normalized_title}\n\n"
        "## Resumen\n"
        f"{_summarize(cleaned_text)}\n\n"
        "## Informacion principal\n"
        f"{cleaned_text if cleaned_text else MISSING_INFO_TEXT}\n\n"
        "## Personal relacionado\n"
        f"{_build_section(flat_data, 'personal')}\n\n"
        "## Equipamiento\n"
        f"{_build_section(flat_data, 'equipamiento')}\n\n"
        "## Ubicacion\n"
        f"{_build_section(flat_data, 'ubicacion')}\n\n"
        "## Servicios\n"
        f"{_build_section(flat_data, 'servicios')}\n\n"
        "## Datos estructurados\n"
        f"{_build_structured_table(flat_data)}\n\n"
        "## Fuente\n"
        f"{'\n'.join(fuente_lines)}\n"
    )

    return markdown.strip(), detected_tipo


def normalize_documents_to_markdown(
    documents: list[Document],
    *,
    filter_useful: bool = True,
    preserve_full_text: bool = False,
) -> list[Document]:
    """Convierte documentos de LangChain a Markdown unificado."""
    normalized_docs: list[Document] = []
    dropped = 0

    for idx, doc in enumerate(documents):
        if filter_useful and not is_document_useful(doc.page_content):
            dropped += 1
            continue

        metadata = dict(doc.metadata or {})
        title = _select_title(metadata, fallback=f"documento_{idx + 1}")

        markdown, tipo = build_unified_markdown(
            title=title,
            main_text=doc.page_content,
            metadata=metadata,
            structured_data={},
            preserve_full_text=preserve_full_text,
        )

        metadata["formato"] = UNIFIED_FORMAT_VERSION
        metadata["tipo_consulta"] = tipo
        normalized_docs.append(Document(page_content=markdown, metadata=metadata))

    if dropped:
        log.info("Descartados por ruido/baja utilidad: %s de %s documentos", dropped, len(documents))

    return normalized_docs


def firestore_record_to_document(
    *,
    record: dict[str, Any],
    metadata: dict[str, Any],
) -> Document:
    """Transforma un registro de Firestore en un documento Markdown unificado."""
    text_lines = []
    flat = flatten_data(record)
    for key, value in flat.items():
        text_lines.append(f"{key}: {_safe_text(value)}")

    main_text = "\n".join(text_lines) if text_lines else MISSING_INFO_TEXT
    title = _select_title(metadata, fallback=f"registro_{metadata.get('firebase_doc_id', 'sin_id')}")

    markdown, tipo = build_unified_markdown(
        title=title,
        main_text=main_text,
        metadata=metadata,
        structured_data=record,
    )

    enriched_meta = dict(metadata)
    enriched_meta["formato"] = UNIFIED_FORMAT_VERSION
    enriched_meta["tipo_consulta"] = tipo

    return Document(page_content=markdown, metadata=enriched_meta)


def export_documents_as_markdown(documents: list[Document], output_dir: str) -> int:
    """Exporta los documentos ya normalizados a archivos .md."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved = 0
    for idx, doc in enumerate(documents, start=1):
        metadata = doc.metadata or {}
        file_seed = _select_title(metadata, fallback=f"documento_{idx}")
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", file_seed).strip("-") or f"documento-{idx}"

        destination = output_path / f"{idx:04d}_{slug[:70]}.md"
        destination.write_text(doc.page_content.strip() + "\n", encoding="utf-8")
        saved += 1

    return saved
