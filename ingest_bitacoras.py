"""
Ingesta de bitacoras XLSX -> Markdown normalizado -> Chroma Cloud.

Uso:
    python ingest_bitacoras.py --input-dir ./bitacoras --output-dir ./bitacoras_md
    python ingest_bitacoras.py --input-dir ./bitacoras --output-dir ./bitacoras_md --dry-run
"""

import argparse
import hashlib
import logging
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


COLLECTION_NAME = "asistente_universidad"
EMBEDDING_MODEL = "models/gemini-embedding-001"

CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_CLOUD_HOST = os.getenv("CHROMA_CLOUD_HOST", "api.trychroma.com")
CHROMA_CLOUD_PORT = int(os.getenv("CHROMA_CLOUD_PORT", "443"))


@dataclass
class InternalActivity:
    actividad: str
    partes: str
    herramientas: str
    mensual: str
    semestral: str
    anual: str


@dataclass
class ExternalMaintenanceRow:
    fecha: str
    descripcion: str
    refacciones: str


@dataclass
class BitacoraData:
    maquina: str
    seccion: str
    ubicacion: str
    responsable: str
    actualizacion: str
    especificaciones: str
    materiales: str
    caracteristicas: str
    uso: str
    archivo_origen: str
    internal_sections: dict[str, list[InternalActivity]]
    external_rows: list[ExternalMaintenanceRow]
    elaboro: str
    reviso: str
    autorizo: str


@dataclass
class ChunkPayload:
    chunk_type: str
    text: str
    subsection: str = ""


def _normalize_for_match(value: str) -> str:
    text = str(value or "").strip().lower()
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def _clean_cell(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")

    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value).strip()

    text = str(value).strip()
    if _normalize_for_match(text) == "nan":
        return ""
    return text


def _safe_get(df: pd.DataFrame, row_idx: int, col_idx: int) -> str:
    if row_idx < 0 or col_idx < 0:
        return ""
    if row_idx >= df.shape[0] or col_idx >= df.shape[1]:
        return ""
    return _clean_cell(df.iat[row_idx, col_idx])


def _value_or_unspecified(value: str) -> str:
    return value if value else "No especificado"


def _is_x(value: str) -> bool:
    return _normalize_for_match(value) == "x"


def _to_checkmark(value: str) -> str:
    return "✓" if _is_x(value) else ""


def _is_subsection_row(
    actividad: str,
    partes: str,
    herramientas: str,
    mensual: str,
    semestral: str,
    anual: str,
) -> bool:
    if not actividad:
        return False

    letters = "".join(ch for ch in actividad if ch.isalpha())
    if not letters:
        return False

    is_upper = letters == letters.upper()
    detail_cols_empty = not any([partes, herramientas, mensual, semestral, anual])
    return is_upper and detail_cols_empty


def _is_activity_row(mensual: str, semestral: str, anual: str) -> bool:
    return _is_x(mensual) or _is_x(semestral) or _is_x(anual)


def _escape_md_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


def _extract_actualizacion(value: str) -> str:
    if not value:
        return ""

    match = re.search(r"actualizaci[oó]n\s*:\s*(.+)", value, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()

    if ":" in value:
        return value.split(":", 1)[1].strip()

    return value.strip()


def _find_row_with_labels(
    df: pd.DataFrame,
    label_left: str,
    label_right: str,
    *,
    search_start: int = 0,
    search_end: int | None = None,
    max_cols: int = 12,
) -> tuple[int, int, int] | None:
    end_row = search_end if search_end is not None else min(df.shape[0], 30)
    scan_cols = min(df.shape[1], max_cols)

    left_norm = _normalize_for_match(label_left)
    right_norm = _normalize_for_match(label_right)

    for row_idx in range(search_start, end_row):
        left_col = -1
        right_col = -1

        for col_idx in range(scan_cols):
            value = _normalize_for_match(_safe_get(df, row_idx, col_idx))
            if not value:
                continue
            if left_col < 0 and left_norm in value:
                left_col = col_idx
            if right_col < 0 and right_norm in value:
                right_col = col_idx

        if left_col >= 0 and right_col >= 0:
            return row_idx, left_col, right_col

    return None


def _find_maintenance_columns(df: pd.DataFrame) -> tuple[int, dict[str, int]]:
    scan_cols = min(df.shape[1], 16)

    for row_idx in range(df.shape[0]):
        row_map: dict[str, int] = {}

        for col_idx in range(scan_cols):
            value = _normalize_for_match(_safe_get(df, row_idx, col_idx))
            if not value:
                continue

            if "actividad" in value:
                row_map.setdefault("actividad", col_idx)
            elif "partes" in value:
                row_map.setdefault("partes", col_idx)
            elif "herramient" in value:
                row_map.setdefault("herramientas", col_idx)
            elif "mensual" in value:
                row_map.setdefault("mensual", col_idx)
            elif "semestral" in value:
                row_map.setdefault("semestral", col_idx)
            elif "anual" in value:
                row_map.setdefault("anual", col_idx)
            elif value == "fecha" or "fecha" in value:
                row_map.setdefault("fecha", col_idx)
            elif "descrip" in value:
                row_map.setdefault("descripcion", col_idx)
            elif "refaccion" in value:
                row_map.setdefault("refacciones", col_idx)

        has_internal = all(key in row_map for key in ["actividad", "partes", "herramientas"])
        has_freq = any(key in row_map for key in ["mensual", "semestral", "anual"])
        if has_internal and has_freq:
            return row_idx, row_map

    fallback_row = min(max(df.shape[0] - 1, 0), 18)
    first_nonempty_col = 0
    for col_idx in range(min(df.shape[1], 12)):
        if _safe_get(df, fallback_row, col_idx):
            first_nonempty_col = col_idx
            break

    fallback_map = {
        "actividad": first_nonempty_col,
        "partes": first_nonempty_col + 1,
        "herramientas": first_nonempty_col + 2,
        "mensual": first_nonempty_col + 3,
        "semestral": first_nonempty_col + 4,
        "anual": first_nonempty_col + 5,
        "fecha": first_nonempty_col + 6,
        "descripcion": first_nonempty_col + 7,
        "refacciones": first_nonempty_col + 8,
    }
    return fallback_row, fallback_map


def _find_signature_header_row(df: pd.DataFrame) -> int | None:
    scan_cols = min(df.shape[1], 12)
    for row_idx in range(df.shape[0]):
        row_values = [_normalize_for_match(_safe_get(df, row_idx, c)) for c in range(scan_cols)]
        has_elaboro = any("elaboro" in value for value in row_values)
        has_reviso = any("reviso" in value for value in row_values)
        has_autorizo = any("autorizo" in value for value in row_values)
        if has_elaboro and has_reviso and has_autorizo:
            return row_idx
    return None


def _find_label_column(df: pd.DataFrame, row_idx: int, label: str, default_col: int) -> int:
    scan_cols = min(df.shape[1], 12)
    normalized_label = _normalize_for_match(label)
    for col_idx in range(scan_cols):
        value = _normalize_for_match(_safe_get(df, row_idx, col_idx))
        if normalized_label in value:
            return col_idx
    return default_col


def _slug_component(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value.strip().lower())
    ascii_only = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_only)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "sin_nombre"


def _build_output_filename(section_name: str, file_stem: str) -> str:
    return f"{_slug_component(section_name)}__{_slug_component(file_stem)}.md"


def _build_vector_store() -> Chroma:
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("Falta GOOGLE_API_KEY en el archivo .env")

    if not CHROMA_API_KEY or not CHROMA_TENANT or not CHROMA_DATABASE:
        raise RuntimeError("Faltan CHROMA_API_KEY, CHROMA_TENANT o CHROMA_DATABASE en el .env")

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


def parse_bitacora_xlsx(file_path: Path, section_name: str) -> BitacoraData:
    df = pd.read_excel(file_path, header=None, engine="openpyxl")

    maquina = ""
    for candidate_col in [1, 0, 2, 3]:
        maquina = _safe_get(df, 5, candidate_col)
        if maquina:
            break
    if not maquina:
        for candidate_col in range(min(df.shape[1], 8)):
            maquina = _safe_get(df, 4, candidate_col)
            if maquina:
                break
    if not maquina:
        maquina = file_path.stem.replace("_", " ").strip()

    actualizacion = ""
    for row_idx in range(min(df.shape[0], 12)):
        for col_idx in range(min(df.shape[1], 12)):
            cell = _safe_get(df, row_idx, col_idx)
            if "actualiz" in _normalize_for_match(cell):
                actualizacion = _extract_actualizacion(cell)
                break
        if actualizacion:
            break

    ubicacion = ""
    especificaciones = ""
    responsable = ""
    materiales = ""
    caracteristicas = ""
    uso = ""

    ubi_row = _find_row_with_labels(df, "ubicacion", "especificaciones", search_start=5, search_end=20)
    if ubi_row is not None:
        row_idx, col_left, col_right = ubi_row
        ubicacion = _safe_get(df, row_idx + 1, col_left)
        especificaciones = _safe_get(df, row_idx + 1, col_right)

    resp_row = _find_row_with_labels(df, "responsable", "materiales", search_start=7, search_end=24)
    if resp_row is not None:
        row_idx, col_left, col_right = resp_row
        responsable = _safe_get(df, row_idx + 1, col_left)
        materiales = _safe_get(df, row_idx + 1, col_right)

    car_row = _find_row_with_labels(df, "caracteristicas", "uso", search_start=8, search_end=26)
    if car_row is not None:
        row_idx, col_left, col_right = car_row
        caracteristicas = _safe_get(df, row_idx + 1, col_left)
        uso = _safe_get(df, row_idx + 1, col_right)

    table_header_row, table_cols = _find_maintenance_columns(df)
    signature_header_row = _find_signature_header_row(df)
    table_end_row = signature_header_row if signature_header_row is not None else df.shape[0]

    internal_sections: dict[str, list[InternalActivity]] = {}
    external_rows: list[ExternalMaintenanceRow] = []
    current_subsection = "General"

    for row_idx in range(table_header_row + 1, table_end_row):
        actividad = _safe_get(df, row_idx, table_cols.get("actividad", 0))
        partes = _safe_get(df, row_idx, table_cols.get("partes", 1))
        herramientas = _safe_get(df, row_idx, table_cols.get("herramientas", 2))
        mensual = _safe_get(df, row_idx, table_cols.get("mensual", 3))
        semestral = _safe_get(df, row_idx, table_cols.get("semestral", 4))
        anual = _safe_get(df, row_idx, table_cols.get("anual", 5))

        fecha_ext = _safe_get(df, row_idx, table_cols.get("fecha", 6))
        descripcion_ext = _safe_get(df, row_idx, table_cols.get("descripcion", 7))
        refacciones_ext = _safe_get(df, row_idx, table_cols.get("refacciones", 8))

        if not any([
            actividad,
            partes,
            herramientas,
            mensual,
            semestral,
            anual,
            fecha_ext,
            descripcion_ext,
            refacciones_ext,
        ]):
            continue

        normalized_activity = _normalize_for_match(actividad)
        if normalized_activity in {
            "actividad",
            "subencabezados",
            "mantenimiento interno",
            "mantenimiento externo",
            "mensual",
            "semestral",
            "anual",
            "frecuencia",
        }:
            continue

        if _is_subsection_row(actividad, partes, herramientas, mensual, semestral, anual):
            current_subsection = actividad.strip()
            internal_sections.setdefault(current_subsection, [])
            continue

        if _is_activity_row(mensual, semestral, anual):
            internal_sections.setdefault(current_subsection, [])
            internal_sections[current_subsection].append(
                InternalActivity(
                    actividad=actividad,
                    partes=partes,
                    herramientas=herramientas,
                    mensual=_to_checkmark(mensual),
                    semestral=_to_checkmark(semestral),
                    anual=_to_checkmark(anual),
                )
            )

        has_external_data = any([fecha_ext, descripcion_ext, refacciones_ext])
        fecha_norm = _normalize_for_match(fecha_ext)
        descripcion_norm = _normalize_for_match(descripcion_ext)
        refacciones_norm = _normalize_for_match(refacciones_ext)
        looks_like_external_header = (
            fecha_norm in {"fecha", "anual", "mensual", "semestral", "frecuencia", "mantenimiento externo"}
            or descripcion_norm in {"fecha", "descripcion", "descripcon", "descripción"}
            or "refaccion" in refacciones_norm
        )
        if has_external_data and not looks_like_external_header:
            external_rows.append(
                ExternalMaintenanceRow(
                    fecha=fecha_ext,
                    descripcion=descripcion_ext,
                    refacciones=refacciones_ext,
                )
            )

    if not internal_sections:
        internal_sections = {"General": []}

    elaboro = ""
    reviso = ""
    autorizo = ""

    if signature_header_row is not None and signature_header_row + 1 < df.shape[0]:
        value_row = signature_header_row + 1
        elaboro_col = _find_label_column(df, signature_header_row, "elaboro", 0)
        reviso_col = _find_label_column(df, signature_header_row, "reviso", 1)
        autorizo_col = _find_label_column(df, signature_header_row, "autorizo", 2)

        elaboro = _safe_get(df, value_row, elaboro_col)
        reviso = _safe_get(df, value_row, reviso_col)
        autorizo = _safe_get(df, value_row, autorizo_col)

    return BitacoraData(
        maquina=_value_or_unspecified(maquina),
        seccion=_value_or_unspecified(section_name),
        ubicacion=_value_or_unspecified(ubicacion),
        responsable=_value_or_unspecified(responsable),
        actualizacion=_value_or_unspecified(actualizacion),
        especificaciones=_value_or_unspecified(especificaciones),
        materiales=_value_or_unspecified(materiales),
        caracteristicas=_value_or_unspecified(caracteristicas),
        uso=_value_or_unspecified(uso),
        archivo_origen=file_path.name,
        internal_sections=internal_sections,
        external_rows=external_rows,
        elaboro=_value_or_unspecified(elaboro),
        reviso=_value_or_unspecified(reviso),
        autorizo=_value_or_unspecified(autorizo),
    )


def render_markdown(data: BitacoraData) -> str:
    lines = [
        f"# Bitácora de Mantenimiento: {data.maquina}",
        "",
        f"**Sección:** {data.seccion}",
        f"**Ubicación:** {data.ubicacion}",
        f"**Responsable:** {data.responsable}",
        f"**Actualización:** {data.actualizacion}",
        "",
        "---",
        "",
        "## Descripción técnica",
        "",
        f"**Especificaciones de trabajo:** {data.especificaciones}",
        f"**Materiales de trabajo:** {data.materiales}",
        f"**Características técnicas:** {data.caracteristicas}",
        f"**Uso:** {data.uso}",
        "",
        "---",
        "",
        "## Plan de mantenimiento interno",
        "",
    ]

    for subsection, activities in data.internal_sections.items():
        lines.extend([
            f"### {subsection}",
            "",
            "| Actividad | Partes | Herramientas | Mensual | Semestral | Anual |",
            "|-----------|--------|--------------|---------|-----------|-------|",
        ])
        if activities:
            for activity in activities:
                lines.append(
                    "| "
                    f"{_escape_md_cell(activity.actividad)} | "
                    f"{_escape_md_cell(activity.partes)} | "
                    f"{_escape_md_cell(activity.herramientas)} | "
                    f"{_escape_md_cell(activity.mensual)} | "
                    f"{_escape_md_cell(activity.semestral)} | "
                    f"{_escape_md_cell(activity.anual)} |"
                )
        else:
            lines.append("|  |  |  |  |  |  |")
        lines.append("")

    if data.external_rows:
        lines.extend([
            "---",
            "",
            "## Mantenimiento externo",
            "",
            "| Fecha | Descripción | Refacciones |",
            "|-------|-------------|-------------|",
        ])
        for row in data.external_rows:
            lines.append(
                "| "
                f"{_escape_md_cell(row.fecha)} | "
                f"{_escape_md_cell(row.descripcion)} | "
                f"{_escape_md_cell(row.refacciones)} |"
            )
        lines.append("")

    lines.extend([
        "---",
        "",
        "## Firmas",
        "",
        "| Elaboró | Revisó | Autorizó |",
        "|---------|--------|----------|",
        (
            "| "
            f"{_escape_md_cell(data.elaboro)} | "
            f"{_escape_md_cell(data.reviso)} | "
            f"{_escape_md_cell(data.autorizo)} |"
        ),
    ])

    return "\n".join(lines).strip() + "\n"


def _chunk_context_line(data: BitacoraData, chunk_type: str) -> str:
    return f"Máquina: {data.maquina} | Sección: {data.seccion} | Tipo: {chunk_type}"


def _render_internal_subsection_table(subsection: str, activities: list[InternalActivity]) -> str:
    lines = [
        f"### {subsection}",
        "",
        "| Actividad | Partes | Herramientas | Mensual | Semestral | Anual |",
        "|-----------|--------|--------------|---------|-----------|-------|",
    ]

    if activities:
        for activity in activities:
            lines.append(
                "| "
                f"{_escape_md_cell(activity.actividad)} | "
                f"{_escape_md_cell(activity.partes)} | "
                f"{_escape_md_cell(activity.herramientas)} | "
                f"{_escape_md_cell(activity.mensual)} | "
                f"{_escape_md_cell(activity.semestral)} | "
                f"{_escape_md_cell(activity.anual)} |"
            )
    else:
        lines.append("|  |  |  |  |  |  |")

    return "\n".join(lines)


def build_chunks(data: BitacoraData) -> list[ChunkPayload]:
    chunks: list[ChunkPayload] = []

    descripcion_chunk_lines = [
        _chunk_context_line(data, "descripcion"),
        "",
        f"# Bitácora de Mantenimiento: {data.maquina}",
        "",
        f"**Sección:** {data.seccion}",
        f"**Ubicación:** {data.ubicacion}",
        f"**Responsable:** {data.responsable}",
        f"**Actualización:** {data.actualizacion}",
        "",
        "---",
        "",
        "## Descripción técnica",
        "",
        f"**Especificaciones de trabajo:** {data.especificaciones}",
        f"**Materiales de trabajo:** {data.materiales}",
        f"**Características técnicas:** {data.caracteristicas}",
        f"**Uso:** {data.uso}",
    ]
    chunks.append(ChunkPayload(chunk_type="descripcion", text="\n".join(descripcion_chunk_lines)))

    for subsection, activities in data.internal_sections.items():
        subsection_table = _render_internal_subsection_table(subsection, activities)
        chunk_text = "\n".join([
            _chunk_context_line(data, "mantenimiento_interno"),
            "",
            f"# Bitácora de Mantenimiento: {data.maquina}",
            "",
            "## Plan de mantenimiento interno",
            "",
            subsection_table,
        ])
        chunks.append(
            ChunkPayload(
                chunk_type="mantenimiento_interno",
                text=chunk_text,
                subsection=subsection,
            )
        )

    if data.external_rows:
        external_lines = [
            _chunk_context_line(data, "mantenimiento_externo"),
            "",
            f"# Bitácora de Mantenimiento: {data.maquina}",
            "",
            "## Mantenimiento externo",
            "",
            "| Fecha | Descripción | Refacciones |",
            "|-------|-------------|-------------|",
        ]
        for row in data.external_rows:
            external_lines.append(
                "| "
                f"{_escape_md_cell(row.fecha)} | "
                f"{_escape_md_cell(row.descripcion)} | "
                f"{_escape_md_cell(row.refacciones)} |"
            )
        chunks.append(
            ChunkPayload(chunk_type="mantenimiento_externo", text="\n".join(external_lines))
        )

    return chunks


def _chunk_id(data: BitacoraData, chunk: ChunkPayload, index: int) -> str:
    seed = "|".join([
        data.archivo_origen,
        data.maquina,
        chunk.chunk_type,
        chunk.subsection,
        str(index),
    ])
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()


def _upload_chunks(
    vector_store: Chroma,
    data: BitacoraData,
    chunks: list[ChunkPayload],
) -> tuple[int, int]:
    metadata_base = {
        "fuente": "bitacora_mantenimiento",
        "categoria": "mantenimiento",
        "maquina": data.maquina,
        "seccion": data.seccion,
        "ubicacion": data.ubicacion,
        "responsable": data.responsable,
        "archivo_origen": data.archivo_origen,
        "actualizacion": data.actualizacion,
    }

    vector_store.delete(where={"archivo_origen": data.archivo_origen})

    uploaded = 0
    failed = 0

    for idx, chunk in enumerate(chunks):
        metadata = {**metadata_base, "chunk_type": chunk.chunk_type}
        chunk_id = _chunk_id(data, chunk, idx)

        success = False
        for attempt in range(3):
            try:
                vector_store.add_texts(
                    texts=[chunk.text],
                    metadatas=[metadata],
                    ids=[chunk_id],
                )
                success = True
                break
            except Exception as exc:
                if attempt < 2:
                    log.warning(
                        "Fallo al subir chunk %s (%s). Reintento %s/2 en 2s. Error: %s",
                        chunk_id,
                        chunk.chunk_type,
                        attempt + 1,
                        exc,
                    )
                    time.sleep(2)
                else:
                    log.error(
                        "Fallo definitivo al subir chunk %s (%s). Error: %s",
                        chunk_id,
                        chunk.chunk_type,
                        exc,
                    )

        if success:
            uploaded += 1
        else:
            failed += 1

    return uploaded, failed


def process_file(file_path: Path, output_dir: Path, vector_store: Chroma | None, dry_run: bool) -> tuple[bool, int, int, int]:
    section_name = file_path.parent.name
    data = parse_bitacora_xlsx(file_path, section_name)

    markdown = render_markdown(data)
    output_filename = _build_output_filename(section_name, file_path.stem)
    output_path = output_dir / output_filename
    output_path.write_text(markdown, encoding="utf-8")

    chunks = build_chunks(data)
    generated_chunks = len(chunks)

    if dry_run:
        log.info("Dry-run | %s | markdown=%s | chunks=%s", file_path.name, output_filename, generated_chunks)
        return True, generated_chunks, 0, 0

    if vector_store is None:
        raise RuntimeError("Vector store no inicializado.")

    uploaded, failed = _upload_chunks(vector_store, data, chunks)
    file_ok = failed == 0
    return file_ok, generated_chunks, uploaded, failed


def _collect_xlsx_files(input_dir: Path) -> list[Path]:
    files = list(input_dir.rglob("*.xlsx"))
    files.extend(input_dir.rglob("*.XLSX"))
    unique_files = sorted(set(files))
    return unique_files


def main(input_dir: str, output_dir: str, dry_run: bool) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists() or not input_path.is_dir():
        raise ValueError(f"La carpeta de entrada no existe o no es directorio: {input_dir}")

    xlsx_files = _collect_xlsx_files(input_path)
    if not xlsx_files:
        log.warning("No se encontraron archivos .xlsx en %s", input_path)

    vector_store: Chroma | None = None
    if not dry_run:
        vector_store = _build_vector_store()

    archivos_ok = 0
    archivos_error = 0
    chunks_generados = 0
    chunks_subidos = 0
    chunks_fallidos = 0

    for file_path in xlsx_files:
        try:
            file_ok, generated, uploaded, failed = process_file(
                file_path=file_path,
                output_dir=output_path,
                vector_store=vector_store,
                dry_run=dry_run,
            )
            chunks_generados += generated
            chunks_subidos += uploaded
            chunks_fallidos += failed
            if file_ok:
                archivos_ok += 1
            else:
                archivos_error += 1
        except Exception as exc:
            archivos_error += 1
            log.error("Error procesando %s: %s", file_path, exc)

    log.info("=== Resumen de ingesta bitacoras ===")
    log.info("Archivos OK: %s", archivos_ok)
    log.info("Archivos con error: %s", archivos_error)
    log.info("Chunks generados: %s", chunks_generados)
    log.info("Chunks subidos: %s", chunks_subidos)
    log.info("Chunks fallidos: %s", chunks_fallidos)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingesta bitacoras XLSX -> Markdown -> Chroma Cloud")
    parser.add_argument("--input-dir", required=True, help="Carpeta raiz que contiene subcarpetas con .xlsx")
    parser.add_argument("--output-dir", required=True, help="Carpeta destino para Markdown normalizado")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Genera Markdown y chunks sin subir a ChromaDB",
    )

    args = parser.parse_args()
    main(input_dir=args.input_dir, output_dir=args.output_dir, dry_run=args.dry_run)
