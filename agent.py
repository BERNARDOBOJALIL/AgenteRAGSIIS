import logging
import os
import re
import json
import unicodedata
from difflib import SequenceMatcher
from typing import Annotated, Any, TypedDict

import chromadb
from dotenv import load_dotenv
from firebase_sources import build_firebase_runtime_context, fetch_firebase_live_snapshot
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()
logging.basicConfig(level=logging.WARNING)  # silenciar logs de LangChain en chat

COLLECTION_NAME = "asistente_universidad"
EMBEDDING_MODEL = "models/gemini-embedding-001"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_CLOUD_HOST = os.getenv("CHROMA_CLOUD_HOST", "api.trychroma.com")
CHROMA_CLOUD_PORT = int(os.getenv("CHROMA_CLOUD_PORT", "443"))
AGENT_DEBUG_TRACE = os.getenv("AGENT_DEBUG_TRACE", "").strip().lower() in {"1", "true", "yes", "on"}
SALON_CODE_PATTERN = re.compile(r"\bj[\s_-]?\d{1,3}(?:-[a-z])?\b", flags=re.IGNORECASE)
SALON_CODE_LOOSE_PATTERN = re.compile(r"\bj[\s_-]?(\d{1,3})(?:[\s_-]?([a-z]))?\b", flags=re.IGNORECASE)
HORARIO_CLASS_NAME_KEYS = (
    "clase",
    "clases",
    "materia",
    "materias",
    "asignatura",
    "asignaturas",
    "curso",
    "cursos",
    "taller",
    "actividad",
    "nombre_clase",
    "nombreclase",
    "nombre_materia",
    "nombremateria",
)
HORARIO_GROUP_KEYS = ("grupo", "grupos", "seccion", "secciones")
HORARIO_TEACHER_KEYS = ("profesor", "docente", "maestro")
SALON_FUZZY_MIN_RATIO = 0.58
SALON_RELAXED_FUZZY_MIN_RATIO = 0.42

agent_log = logging.getLogger(__name__)

SYSTEM_PROMPT = """Eres un asistente virtual amable y preciso de la universidad.

Reglas obligatorias:
- Responde siempre en español, claro y conciso.
- Usa únicamente la información del contexto proporcionado.
- Si no hay datos suficientes, responde exactamente: "No tengo esa información disponible."
- Nunca inventes datos, horarios, nombres, requisitos o personas.
- Nunca proporciones información personal o sensible de alumnos.
- Nunca menciones fuentes técnicas o internas (Firebase, Chroma, JSON, API, logs, permisos o fichas de conocimiento).
- Para preguntas de salones, prioriza equipamiento y ubicación aproximada (piso, zona y referencias).
- Incluye horarios detallados solo cuando el usuario lo pida explícitamente.

Uso de tools:
- Para preguntas sobre equipos, máquinas, instrumentos o herramientas del IDIT (características técnicas,
  especificaciones, ubicación, responsable, mantenimiento, uso permitido), usa buscar_equipos_idit.
  Si el usuario menciona una sección específica (ej. FABLAB, ELECTRONICA), pasa ese dato al parámetro seccion.
  Si pregunta por un tipo específico de información (ej. plan de mantenimiento), pasa el valor al parámetro chunk_type.
- Para preguntas sobre información general e institucional del IDIT (qué es el IDIT, servicios disponibles,
  tecnologías, áreas, programas académicos, descripción de laboratorios), usa buscar_informacion_idit.
- Nunca respondas preguntas sobre equipos o información institucional sin consultar primero las tools correspondientes.
- El agente decide qué tool usar según la intención de la pregunta y el contexto disponible, no por listas rígidas de palabras.
- Para consultas de salones, usa buscar_salones_idit y considera tanto coincidencias exactas como aproximadas.
- Cuando buscar_salones_idit regrese similarity y match_confidence, prioriza los resultados con mayor similarity.
- Si no hay coincidencia exacta pero sí candidatos de confidence media o baja, ofrece el mejor candidato y pide confirmación.
- Integra siempre la información de las tools con el contexto operativo disponible para dar la mejor respuesta posible.
"""


def _normalize_lookup_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def _normalize_compact_text(value: Any) -> str:
    """Normaliza y compacta texto para comparar variantes con/ sin separadores."""
    normalized = _normalize_lookup_text(value)
    return re.sub(r"[^a-z0-9]+", "", normalized)


def _emit_structured_debug(event: str, **payload: Any) -> None:
    """Emite logs estructurados cuando el modo debug del agente esta activo."""
    if not AGENT_DEBUG_TRACE:
        return

    record = {
        "event": event,
        **payload,
    }
    try:
        printable = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        printable = str(record)

    print(f"[LANGGRAPH_DEBUG] {printable}")


def _canonicalize_salon_code(value: Any) -> str:
    """Normaliza variaciones como J003/j003/j-003 a formato canonico J-003."""
    text = str(value or "").strip()
    if not text:
        return ""

    match = SALON_CODE_LOOSE_PATTERN.search(text)
    if not match:
        return ""

    try:
        number = int(match.group(1))
    except (TypeError, ValueError):
        return ""

    suffix = str(match.group(2) or "").strip().upper()
    canonical = f"J-{number:03d}"
    if suffix:
        canonical = f"{canonical}-{suffix}"
    return canonical


def _normalize_salon_codes_in_text(value: Any) -> str:
    """Reescribe codigos de salon en texto libre al formato canonico."""
    text = str(value or "").strip()
    if not text:
        return ""

    def _replace(match: re.Match[str]) -> str:
        try:
            number = int(match.group(1))
        except (TypeError, ValueError):
            return match.group(0)

        suffix = str(match.group(2) or "").strip().upper()
        canonical = f"J-{number:03d}"
        if suffix:
            canonical = f"{canonical}-{suffix}"
        return canonical

    return SALON_CODE_LOOSE_PATTERN.sub(_replace, text)


def _build_salon_query_variants(query: str) -> list[str]:
    """Genera variantes ligeras de consulta sin expandir por listas de sinonimos."""
    base = str(query or "").strip()
    if not base:
        return []

    normalized_full = _normalize_salon_codes_in_text(base)
    canonical_code = _canonicalize_salon_code(base)

    variants: list[str] = []
    seen: set[str] = set()

    def _add(value: str) -> None:
        text = str(value or "").strip()
        if not text:
            return
        key = _normalize_lookup_text(text).replace(" ", "")
        if key in seen:
            return
        seen.add(key)
        variants.append(text)

    _add(base)
    _add(normalized_full)
    _add(_normalize_lookup_text(normalized_full))

    if canonical_code:
        _add(canonical_code)
        _add(canonical_code.replace("-", ""))
        _add(f"salon {canonical_code}")

    return variants


def _tokenize_for_tool_search(question: str) -> set[str]:
    stopwords = {
        "donde",
        "esta",
        "esta",
        "que",
        "cual",
        "cuales",
        "los",
        "las",
        "del",
        "de",
        "para",
        "con",
        "una",
        "unos",
        "unas",
        "salon",
        "salones",
        "horario",
        "horarios",
        "disponible",
        "disponibilidad",
    }
    normalized = _normalize_lookup_text(question)
    tokens = [tok for tok in re.split(r"[^a-z0-9]+", normalized) if len(tok) >= 3]
    return {tok for tok in tokens if tok not in stopwords}


def _contains_term_with_boundaries(text: str, term: str) -> bool:
    if not text or not term:
        return False
    if " " in term:
        return term in text
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text))


def _normalized_word_tokens(text: str) -> set[str]:
    return {tok for tok in re.split(r"[^a-z0-9]+", text or "") if tok}


def _contains_approx_term(tokens: set[str], term: str, min_ratio: float = 0.83) -> bool:
    if not term or len(term) < 4 or not tokens:
        return False

    term_compact = _normalize_compact_text(term)
    if len(term_compact) < 4:
        return False

    for token in tokens:
        token_compact = _normalize_compact_text(token)
        if len(token_compact) < 4:
            continue
        if abs(len(token_compact) - len(term_compact)) > 4:
            continue

        if token_compact == term_compact:
            return True
        if token_compact in term_compact or term_compact in token_compact:
            return True
        if SequenceMatcher(None, token_compact, term_compact).ratio() >= min_ratio:
            return True

    return False


def _expand_salon_query_terms(base_terms: set[str]) -> set[str]:
    """Expande terminos solo con reglas morfologicas ligeras."""
    expanded: set[str] = set()

    for raw_term in base_terms:
        term = _normalize_lookup_text(raw_term)
        if not term:
            continue

        expanded.add(term)
        compact = _normalize_compact_text(term)
        if compact:
            expanded.add(compact)

        if term.endswith("es") and len(term) > 4:
            expanded.add(term[:-2])
        elif term.endswith("s") and len(term) > 4:
            expanded.add(term[:-1])

    return {term for term in expanded if len(term) >= 3}


def _salon_term_weight(term: str) -> int:
    if len(term) >= 11:
        return 5
    if len(term) >= 8:
        return 4
    if len(term) >= 5:
        return 3
    return 2


def _score_salon_item_detailed(
    item: dict[str, Any],
    normalized_query: str,
    terms: set[str],
) -> tuple[int, dict[str, Any]]:
    data = item.get("data") if isinstance(item.get("data"), dict) else {}
    nomenclatura = str(data.get("nomenclatura") or item.get("doc_id") or "").strip()
    nombre = str(data.get("nombre") or "").strip()

    nomen_norm = _normalize_lookup_text(nomenclatura)
    nombre_norm = _normalize_lookup_text(nombre)
    tipo_norm = _normalize_lookup_text(data.get("tipo"))
    nomen_compact = _normalize_compact_text(nomen_norm)
    nombre_compact = _normalize_compact_text(nombre_norm)
    tipo_compact = _normalize_compact_text(tipo_norm)
    normalized_query_compact = _normalize_compact_text(normalized_query)

    blob_parts = [
        nomen_norm,
        nombre_norm,
        tipo_norm,
        _normalize_lookup_text(data.get("piso")),
        _normalize_lookup_text(data.get("tipoHorario")),
        _normalize_lookup_text(_format_clases_resumen(data.get("horario"), max_items=8)),
        " ".join(_normalize_lookup_text(eq) for eq in data.get("equipamiento", []) if isinstance(eq, str)),
        " ".join(_normalize_lookup_text(name) for name in _extract_responsable_names(data.get("responsables"), max_items=8)),
    ]
    blob = " ".join(part for part in blob_parts if part)
    blob_compact = _normalize_compact_text(blob)
    nomen_tokens = _normalized_word_tokens(nomen_norm)
    nombre_tokens = _normalized_word_tokens(nombre_norm)
    tipo_tokens = _normalized_word_tokens(tipo_norm)
    blob_tokens = _normalized_word_tokens(blob)
    if nomen_compact:
        nomen_tokens.add(nomen_compact)
    if nombre_compact:
        nombre_tokens.add(nombre_compact)
    if tipo_compact:
        tipo_tokens.add(tipo_compact)
    if blob_compact:
        blob_tokens.add(blob_compact)

    base_terms = {term for term in terms if term}
    expanded_terms = _expand_salon_query_terms(base_terms)

    score = 0
    score_components: dict[str, int] = {}
    matched_terms: set[str] = set()
    field_hits = {"nomenclatura": 0, "nombre": 0, "tipo": 0, "blob": 0}

    def add_component(component: str, value: int) -> None:
        nonlocal score
        if value == 0:
            return
        score += value
        score_components[component] = score_components.get(component, 0) + value

    if normalized_query:
        if normalized_query in nomen_norm:
            add_component("exact_query_nomenclatura", 34)
        elif normalized_query_compact and normalized_query_compact in nomen_compact:
            add_component("exact_query_nomenclatura_compact", 28)
        if normalized_query in nombre_norm:
            add_component("exact_query_nombre", 22)
        elif normalized_query_compact and normalized_query_compact in nombre_compact:
            add_component("exact_query_nombre_compact", 20)
        if normalized_query in tipo_norm:
            add_component("exact_query_tipo", 16)
        elif normalized_query_compact and normalized_query_compact in tipo_compact:
            add_component("exact_query_tipo_compact", 14)
        elif normalized_query in blob:
            add_component("exact_query_blob", 7)
        elif normalized_query_compact and normalized_query_compact in blob_compact:
            add_component("exact_query_blob_compact", 6)

    code_match = SALON_CODE_PATTERN.search(normalized_query)
    if code_match:
        asked_code = _normalize_salon_code(code_match.group(0))
        item_code = _normalize_salon_code(nomenclatura)
        if asked_code and asked_code == item_code:
            add_component("exact_code_match", 120)
        elif asked_code and asked_code in item_code:
            add_component("partial_code_match", 28)

    for term in sorted(expanded_terms):
        weight = _salon_term_weight(term)
        term_compact = _normalize_compact_text(term)
        if _contains_term_with_boundaries(nomen_norm, term):
            add_component(f"term_nomenclatura:{term}", weight * 7)
            matched_terms.add(term)
            field_hits["nomenclatura"] += 1
        elif _contains_term_with_boundaries(nombre_norm, term):
            add_component(f"term_nombre:{term}", weight * 6)
            matched_terms.add(term)
            field_hits["nombre"] += 1
        elif _contains_term_with_boundaries(tipo_norm, term):
            add_component(f"term_tipo:{term}", weight * 5)
            matched_terms.add(term)
            field_hits["tipo"] += 1
        elif _contains_term_with_boundaries(blob, term):
            add_component(f"term_blob:{term}", weight * 2)
            matched_terms.add(term)
            field_hits["blob"] += 1
        elif term_compact and term_compact in nomen_compact:
            add_component(f"term_nomenclatura_compact:{term}", weight * 6)
            matched_terms.add(term)
            field_hits["nomenclatura"] += 1
        elif term_compact and term_compact in nombre_compact:
            add_component(f"term_nombre_compact:{term}", weight * 5)
            matched_terms.add(term)
            field_hits["nombre"] += 1
        elif term_compact and term_compact in tipo_compact:
            add_component(f"term_tipo_compact:{term}", weight * 4)
            matched_terms.add(term)
            field_hits["tipo"] += 1
        elif term_compact and term_compact in blob_compact:
            add_component(f"term_blob_compact:{term}", weight * 2)
            matched_terms.add(term)
            field_hits["blob"] += 1
        elif _contains_approx_term(nomen_tokens, term):
            add_component(f"term_nomenclatura_aprox:{term}", weight * 4)
            matched_terms.add(term)
            field_hits["nomenclatura"] += 1
        elif _contains_approx_term(nombre_tokens, term):
            add_component(f"term_nombre_aprox:{term}", weight * 4)
            matched_terms.add(term)
            field_hits["nombre"] += 1
        elif _contains_approx_term(tipo_tokens, term):
            add_component(f"term_tipo_aprox:{term}", weight * 3)
            matched_terms.add(term)
            field_hits["tipo"] += 1
        elif _contains_approx_term(blob_tokens, term):
            add_component(f"term_blob_aprox:{term}", weight)
            matched_terms.add(term)
            field_hits["blob"] += 1

    coverage_ratio = 0.0
    if expanded_terms:
        coverage_ratio = len(matched_terms) / len(expanded_terms)
        add_component("term_coverage_bonus", int(round(coverage_ratio * 16)))

    if len(matched_terms) >= 3:
        add_component("multi_term_bonus", 8)

    if score < 0:
        score = 0

    top_components = [
        {"component": name, "value": value}
        for name, value in sorted(score_components.items(), key=lambda pair: abs(pair[1]), reverse=True)[:8]
    ]

    best_term_similarity = 0.0
    token_pool = nomen_tokens | nombre_tokens | tipo_tokens | blob_tokens
    for term in expanded_terms:
        term_compact = _normalize_compact_text(term)
        if len(term_compact) < 3:
            continue
        for token in token_pool:
            token_compact = _normalize_compact_text(token)
            if len(token_compact) < 3:
                continue
            if token_compact == term_compact:
                best_term_similarity = max(best_term_similarity, 1.0)
                continue
            ratio = SequenceMatcher(None, token_compact, term_compact).ratio()
            if ratio > best_term_similarity:
                best_term_similarity = ratio

    score_similarity = min(1.0, max(0.0, score / 180.0))
    coverage_strength = coverage_ratio * min(1.0, len(expanded_terms) / 3.0) if expanded_terms else 0.0
    term_alignment = min(0.9, best_term_similarity * 0.9)
    similarity = round(min(1.0, max(score_similarity, coverage_strength, term_alignment)), 3)

    details: dict[str, Any] = {
        "base_terms": sorted(base_terms),
        "expanded_terms": sorted(expanded_terms),
        "matched_terms": sorted(matched_terms),
        "matched_phrases": [],
        "coverage_ratio": round(coverage_ratio, 3),
        "similarity": similarity,
        "specific_terms_matched": len(matched_terms),
        "field_hits": field_hits,
        "top_score_components": top_components,
    }

    return score, details


def _extract_responsable_names(values: Any, max_items: int = 4) -> list[str]:
    if not isinstance(values, list):
        return []

    nombres: list[str] = []
    vistos: set[str] = set()
    for item in values:
        if isinstance(item, dict):
            nombre = str(item.get("nombre") or item.get("name") or "").strip()
        else:
            nombre = str(item or "").strip()

        if not nombre:
            continue
        key = _normalize_lookup_text(nombre)
        if key in vistos:
            continue
        vistos.add(key)
        nombres.append(nombre)
        if len(nombres) >= max_items:
            break

    return nombres


def _extract_horario_field(item: dict[str, Any], keys: tuple[str, ...]) -> str:
    normalized_keys = {_normalize_lookup_text(key).replace(" ", "") for key in keys}

    for raw_key, raw_value in item.items():
        key_norm = _normalize_lookup_text(raw_key).replace(" ", "")
        if key_norm not in normalized_keys:
            continue
        value = str(raw_value or "").strip()
        if value:
            return value
    return ""


def _class_descriptor_from_horario_block(block: dict[str, Any]) -> str:
    if not isinstance(block, dict):
        return ""

    class_name = _extract_horario_field(block, HORARIO_CLASS_NAME_KEYS)
    class_group = _extract_horario_field(block, HORARIO_GROUP_KEYS)

    if class_name and class_group:
        return f"{class_name} ({class_group})"
    if class_name:
        return class_name
    if class_group:
        return class_group
    return ""


def _format_clases_resumen(horario: Any, max_items: int = 5) -> str:
    bloques = _compact_horario_for_tool(horario, max_items=max_items * 2)
    if not bloques:
        return "No disponible"

    clases: list[str] = []
    seen: set[str] = set()

    for bloque in bloques:
        descriptor = _class_descriptor_from_horario_block(bloque)
        if not descriptor:
            continue
        key = _normalize_lookup_text(descriptor)
        if key in seen:
            continue
        seen.add(key)
        clases.append(descriptor)
        if len(clases) >= max_items:
            break

    return "; ".join(clases) if clases else "No disponible"


def _compact_horario_for_tool(horario: Any, max_items: int = 8) -> list[dict[str, str]]:
    compact: list[dict[str, str]] = []

    if isinstance(horario, list):
        for item in horario[:max_items]:
            if not isinstance(item, dict):
                continue
            dia = str(item.get("dia") or "").strip()
            inicio = str(item.get("inicio") or "").strip()
            fin = str(item.get("fin") or "").strip()
            if not (dia or inicio or fin):
                continue

            payload = {"dia": dia, "inicio": inicio, "fin": fin}
            class_name = _extract_horario_field(item, HORARIO_CLASS_NAME_KEYS)
            class_group = _extract_horario_field(item, HORARIO_GROUP_KEYS)
            class_teacher = _extract_horario_field(item, HORARIO_TEACHER_KEYS)
            if class_name:
                payload["clase"] = class_name
            if class_group:
                payload["grupo"] = class_group
            if class_teacher:
                payload["profesor"] = class_teacher

            compact.append(payload)
        return compact

    if isinstance(horario, dict):
        for day_key, blocks in horario.items():
            day_label = str(day_key or "").strip().replace("_", " ").capitalize()
            if not day_label:
                continue

            if isinstance(blocks, list):
                for block in blocks:
                    if not isinstance(block, dict):
                        continue
                    inicio = str(block.get("inicio") or "").strip()
                    fin = str(block.get("fin") or "").strip()
                    if not (inicio or fin):
                        continue

                    payload = {"dia": day_label, "inicio": inicio, "fin": fin}
                    class_name = _extract_horario_field(block, HORARIO_CLASS_NAME_KEYS)
                    class_group = _extract_horario_field(block, HORARIO_GROUP_KEYS)
                    class_teacher = _extract_horario_field(block, HORARIO_TEACHER_KEYS)
                    if class_name:
                        payload["clase"] = class_name
                    if class_group:
                        payload["grupo"] = class_group
                    if class_teacher:
                        payload["profesor"] = class_teacher

                    compact.append(payload)
                    if len(compact) >= max_items:
                        return compact
            elif isinstance(blocks, dict):
                inicio = str(blocks.get("inicio") or "").strip()
                fin = str(blocks.get("fin") or "").strip()
                if inicio or fin:

                    payload = {"dia": day_label, "inicio": inicio, "fin": fin}
                    class_name = _extract_horario_field(blocks, HORARIO_CLASS_NAME_KEYS)
                    class_group = _extract_horario_field(blocks, HORARIO_GROUP_KEYS)
                    class_teacher = _extract_horario_field(blocks, HORARIO_TEACHER_KEYS)
                    if class_name:
                        payload["clase"] = class_name
                    if class_group:
                        payload["grupo"] = class_group
                    if class_teacher:
                        payload["profesor"] = class_teacher

                    compact.append(payload)
                    if len(compact) >= max_items:
                        return compact

        return compact

    return []


def _format_horario_resumen(horario: Any, max_items: int = 6) -> str:
    bloques = _compact_horario_for_tool(horario, max_items=max_items)
    if not bloques:
        return "No disponible"

    partes: list[str] = []
    for bloque in bloques:
        dia = bloque.get("dia") or ""
        inicio = bloque.get("inicio") or ""
        fin = bloque.get("fin") or ""
        clase_txt = _class_descriptor_from_horario_block(bloque)
        if dia and inicio and fin:
            base = f"{dia} {inicio}-{fin}"
        elif dia and inicio:
            base = f"{dia} desde {inicio}"
        elif dia and fin:
            base = f"{dia} hasta {fin}"
        elif dia:
            base = dia
        else:
            base = ""

        if base and clase_txt:
            partes.append(f"{base} ({clase_txt})")
        elif base:
            partes.append(base)
        elif clase_txt:
            partes.append(clase_txt)

    return "; ".join(partes) if partes else "No disponible"


def _normalize_floor_label_simple(value: Any) -> str:
    raw = str(value or "").strip()
    norm = _normalize_lookup_text(raw)
    if norm in {"pb", "planta baja", "baja", "0", "nivel 0"}:
        return "planta baja"
    if norm in {"pa", "planta alta", "alta", "1", "nivel 1"}:
        return "planta alta"
    return raw


def _extract_j_code_number(value: Any) -> int | None:
    text = str(value or "").strip()
    match = re.search(r"\bj[\s_-]?(\d{1,3})\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _zone_hint_for_code(code: int | None) -> str:
    if code is None:
        return ""
    if code <= 7:
        return 'Bloque derecho'
    if code >7 and  code < 12:
        return 'A la izquierda de la entrada principal'
    if code >= 12 and code <= 19:
        return 'Bloque izquierdo'
    if code >= 20 and code <= 24:
        return 'Zona central por la explanada'
    if code == 25:
        return 'En la explanda'
    return 'A la derecha de la entrada principal'

def _infer_salon_location_for_tool(item: dict[str, Any], data: dict[str, Any]) -> str:
    explicit = str(data.get("ubicacion_aproximada") or "").strip()
    if explicit:
        return explicit

    piso = _normalize_floor_label_simple(data.get("piso"))
    code = _extract_j_code_number(data.get("nomenclatura") or item.get("doc_id"))
    zone = _zone_hint_for_code(code)

    parts = [part for part in [piso, zone] if part]
    return ", ".join(parts) if parts else "ubicacion aproximada no disponible"


def _score_salon_item(item: dict[str, Any], normalized_query: str, terms: set[str]) -> int:
    score, _ = _score_salon_item_detailed(item, normalized_query, terms)
    return score


def _score_user_item(item: dict[str, Any], normalized_query: str, terms: set[str]) -> int:
    usuario = item.get("usuario") if isinstance(item.get("usuario"), dict) else {}
    nombre = str(usuario.get("nombre") or usuario.get("name") or "").strip()
    rol = str(usuario.get("rol") or "").strip()
    tipo = str(usuario.get("tipo") or "").strip()
    correo = str(usuario.get("correo") or usuario.get("email") or "").strip()
    salon_principal = item.get("salon_principal") if isinstance(item.get("salon_principal"), dict) else {}

    blob_parts = [
        _normalize_lookup_text(nombre),
        _normalize_lookup_text(rol),
        _normalize_lookup_text(tipo),
        _normalize_lookup_text(correo),
        _normalize_lookup_text(salon_principal.get("nomenclatura")),
        _normalize_lookup_text(salon_principal.get("nombre")),
    ]
    blob = " ".join(part for part in blob_parts if part)

    score = 0
    if normalized_query and normalized_query in blob:
        score += 8

    for term in terms:
        if term in blob:
            score += 2
        elif term.endswith("es") and len(term) > 3 and term[:-2] in blob:
            score += 1
        elif term.endswith("s") and len(term) > 3 and term[:-1] in blob:
            score += 1

    if score == 0 and not terms and blob:
        score = 1

    return score


def _compact_nested_for_tool(value: Any, max_items: int = 8, max_depth: int = 2) -> Any:
    if max_depth < 0:
        return "..."

    if isinstance(value, dict):
        compact_dict: dict[str, Any] = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= max_items:
                compact_dict["truncated"] = True
                break
            compact_dict[str(key)] = _compact_nested_for_tool(item, max_items=max_items, max_depth=max_depth - 1)
        return compact_dict

    if isinstance(value, list):
        compact_list: list[Any] = []
        for idx, item in enumerate(value):
            if idx >= max_items:
                compact_list.append({"truncated": True})
                break
            compact_list.append(_compact_nested_for_tool(item, max_items=max_items, max_depth=max_depth - 1))
        return compact_list

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def _count_nested_items(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        return len(value)
    return 1


def _safe_json_with_budget(payload: Any, max_chars: int) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(encoded) <= max_chars:
        return encoded

    if isinstance(payload, dict):
        trimmed = dict(payload)
        for key in ("matches", "salones", "usuarios"):
            value = trimmed.get(key)
            if isinstance(value, list) and len(value) > 2:
                trimmed[key] = value[:2]
                trimmed["truncated"] = True
        encoded = json.dumps(trimmed, ensure_ascii=False, separators=(",", ":"))
        if len(encoded) <= max_chars:
            return encoded

    return json.dumps({"truncated": True}, ensure_ascii=False, separators=(",", ":"))


def _resolve_match_similarity(score: int, score_debug: dict[str, Any] | None) -> float:
    details = score_debug if isinstance(score_debug, dict) else {}

    explicit_similarity = details.get("similarity")
    if isinstance(explicit_similarity, (int, float)):
        return round(max(0.0, min(1.0, float(explicit_similarity))), 3)

    coverage = details.get("coverage_ratio")
    coverage_ratio = float(coverage) if isinstance(coverage, (int, float)) else 0.0
    score_ratio = max(0.0, min(1.0, float(score) / 180.0))
    return round(min(1.0, (score_ratio * 0.75) + (coverage_ratio * 0.25)), 3)


def _label_match_confidence(similarity: float) -> str:
    if similarity >= 0.8:
        return "alta"
    if similarity >= 0.55:
        return "media"
    if similarity >= 0.35:
        return "baja"
    return "muy_baja"


def _serialize_salon_for_tool(item: dict[str, Any], score: int, include_schedule: bool) -> dict[str, Any]:
    data = item.get("data") if isinstance(item.get("data"), dict) else {}
    horario = data.get("horario")
    equipamiento = [str(eq).strip() for eq in data.get("equipamiento", []) if str(eq).strip()][:8]
    responsables = _extract_responsable_names(data.get("responsables"), max_items=4)

    payload: dict[str, Any] = {
        "score": score,
        "doc_id": str(item.get("doc_id") or "").strip(),
        "nomenclatura": str(data.get("nomenclatura") or item.get("doc_id") or "").strip(),
        "nombre": str(data.get("nombre") or "").strip(),
        "tipo": str(data.get("tipo") or "").strip(),
        "piso": _normalize_floor_label_simple(data.get("piso")),
        "ubicacion_aproximada": _infer_salon_location_for_tool(item, data),
        "equipamiento": equipamiento,
        "responsables": responsables,
        "tipoHorario": str(data.get("tipoHorario") or "").strip(),
        "horario_resumen": _format_horario_resumen(horario),
        "clases_resumen": _format_clases_resumen(horario),
    }

    if include_schedule:
        compact_horario = _compact_horario_for_tool(horario, max_items=10)
        if compact_horario:
            payload["horario"] = compact_horario

    return {k: v for k, v in payload.items() if v not in (None, "", [], {})}


def _serialize_user_for_tool(item: dict[str, Any], score: int, include_schedule: bool) -> dict[str, Any]:
    usuario = item.get("usuario") if isinstance(item.get("usuario"), dict) else {}
    salon_principal = item.get("salon_principal") if isinstance(item.get("salon_principal"), dict) else {}
    salones_relacionados = item.get("salones_relacionados") if isinstance(item.get("salones_relacionados"), list) else []
    horario = item.get("horario")

    payload: dict[str, Any] = {
        "score": score,
        "doc_id": str(item.get("doc_id") or "").strip(),
        "nombre": str(usuario.get("nombre") or usuario.get("name") or "").strip(),
        "rol": str(usuario.get("rol") or "").strip(),
        "tipo": str(usuario.get("tipo") or "").strip(),
        "correo": str(usuario.get("correo") or usuario.get("email") or "").strip(),
        "salon_principal": {
            "nomenclatura": str(salon_principal.get("nomenclatura") or "").strip(),
            "nombre": str(salon_principal.get("nombre") or "").strip(),
            "piso": str(salon_principal.get("piso") or "").strip(),
        }
        if salon_principal
        else {},
        "salones_relacionados_total": len(salones_relacionados),
        "horario_resumen": _format_horario_resumen(horario),
    }

    if include_schedule:
        compact_horario = _compact_horario_for_tool(horario, max_items=10)
        if compact_horario:
            payload["horario"] = compact_horario

    return {k: v for k, v in payload.items() if v not in (None, "", [], {})}


def _serialize_personal_agenda_for_tool(item: dict[str, Any], score: int, include_details: bool) -> dict[str, Any]:
    usuario = item.get("usuario") if isinstance(item.get("usuario"), dict) else {}
    salon_principal = item.get("salon_principal") if isinstance(item.get("salon_principal"), dict) else {}
    salones_relacionados = item.get("salones_relacionados") if isinstance(item.get("salones_relacionados"), list) else []

    horario = item.get("horario")
    calendario = item.get("calendario")
    citas = item.get("citas")
    notificaciones = item.get("notificaciones")

    payload: dict[str, Any] = {
        "score": score,
        "doc_id": str(item.get("doc_id") or "").strip(),
        "nombre": str(usuario.get("nombre") or usuario.get("name") or "").strip(),
        "rol": str(usuario.get("rol") or "").strip(),
        "tipo": str(usuario.get("tipo") or "").strip(),
        "correo": str(usuario.get("correo") or usuario.get("email") or "").strip(),
        "salon_principal": {
            "nomenclatura": str(salon_principal.get("nomenclatura") or "").strip(),
            "nombre": str(salon_principal.get("nombre") or "").strip(),
            "piso": str(salon_principal.get("piso") or "").strip(),
        }
        if salon_principal
        else {},
        "salones_relacionados": [
            {
                "nomenclatura": str(salon.get("nomenclatura") or salon.get("doc_id") or "").strip(),
                "nombre": str(salon.get("nombre") or "").strip(),
                "piso": str(salon.get("piso") or "").strip(),
            }
            for salon in salones_relacionados[:4]
            if isinstance(salon, dict)
        ],
        "horario_resumen": _format_horario_resumen(horario),
        "calendario_registrado": bool(calendario),
        "citas_registradas": bool(citas),
        "notificaciones_registradas": bool(notificaciones),
        "citas_total": _count_nested_items(citas),
        "notificaciones_total": _count_nested_items(notificaciones),
    }

    if include_details:
        compact_horario = _compact_horario_for_tool(horario, max_items=10)
        if compact_horario:
            payload["horario"] = compact_horario
        if calendario:
            payload["calendario_detalle"] = _compact_nested_for_tool(calendario, max_items=8, max_depth=2)
        if citas:
            payload["citas_detalle"] = _compact_nested_for_tool(citas, max_items=8, max_depth=2)
        if notificaciones:
            payload["notificaciones_detalle"] = _compact_nested_for_tool(notificaciones, max_items=8, max_depth=2)

    return {k: v for k, v in payload.items() if v not in (None, "", [], {})}


def _score_salones_for_query_variant(
    salones: list[dict[str, Any]],
    query_variant: str,
) -> list[tuple[int, dict[str, Any], dict[str, Any]]]:
    """Primera etapa: scoring semantico en memoria sobre snapshot Firebase."""
    normalized_query = _normalize_lookup_text(_normalize_salon_codes_in_text(query_variant))
    terms = _tokenize_for_tool_search(_normalize_salon_codes_in_text(query_variant))

    scored: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for item in salones:
        if not isinstance(item, dict):
            continue
        score, score_debug = _score_salon_item_detailed(item, normalized_query, terms)
        if score <= 0:
            continue
        scored.append((score, item, score_debug))

    scored.sort(
        key=lambda pair: (
            pair[0],
            float(pair[2].get("coverage_ratio") or 0.0),
            int(pair[2].get("specific_terms_matched") or 0),
        ),
        reverse=True,
    )
    return scored


def _fallback_match_salones_by_fields(
    salones: list[dict[str, Any]],
    query_variant: str,
) -> tuple[list[tuple[int, dict[str, Any], dict[str, Any]]], str]:
    """Fallback robusto cuando el scoring principal devuelve 0 resultados."""
    normalized_variant = _normalize_lookup_text(_normalize_salon_codes_in_text(query_variant))
    normalized_variant_compact = _normalize_compact_text(normalized_variant)
    canonical_code = _canonicalize_salon_code(query_variant)
    normalized_code = _normalize_salon_code(canonical_code or query_variant)

    if not normalized_variant and not normalized_variant_compact and not normalized_code:
        return [], "empty_variant"

    field_matches: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for item in salones:
        if not isinstance(item, dict):
            continue

        data = item.get("data") if isinstance(item.get("data"), dict) else {}
        nomen = str(data.get("nomenclatura") or item.get("doc_id") or "").strip()
        nombre = str(data.get("nombre") or "").strip()
        id_conjunto = str(data.get("idConjunto") or "").strip()

        nomen_norm = _normalize_lookup_text(nomen)
        nombre_norm = _normalize_lookup_text(nombre)
        id_conjunto_norm = _normalize_lookup_text(id_conjunto)
        nomen_compact = _normalize_compact_text(nomen_norm)
        nombre_compact = _normalize_compact_text(nombre_norm)
        id_conjunto_compact = _normalize_compact_text(id_conjunto_norm)
        nomen_code_norm = _normalize_salon_code(nomen)

        score = 0
        reasons: list[str] = []

        if normalized_code and normalized_code == nomen_code_norm:
            score += 220
            reasons.append("exact_nomenclatura_code")

        if normalized_variant and normalized_variant == nomen_norm:
            score += 120
            reasons.append("exact_nomenclatura_text")
        elif normalized_variant_compact and normalized_variant_compact == nomen_compact:
            score += 115
            reasons.append("exact_nomenclatura_compact")
        elif normalized_variant and normalized_variant in nomen_norm:
            score += 65
            reasons.append("partial_nomenclatura_text")
        elif normalized_variant_compact and normalized_variant_compact in nomen_compact:
            score += 62
            reasons.append("partial_nomenclatura_compact")

        if normalized_variant and normalized_variant == nombre_norm:
            score += 95
            reasons.append("exact_nombre")
        elif normalized_variant_compact and normalized_variant_compact == nombre_compact:
            score += 90
            reasons.append("exact_nombre_compact")
        elif normalized_variant and normalized_variant in nombre_norm:
            score += 55
            reasons.append("partial_nombre")
        elif normalized_variant_compact and normalized_variant_compact in nombre_compact:
            score += 50
            reasons.append("partial_nombre_compact")

        if normalized_variant and normalized_variant == id_conjunto_norm:
            score += 70
            reasons.append("exact_idConjunto")
        elif normalized_variant_compact and normalized_variant_compact == id_conjunto_compact:
            score += 65
            reasons.append("exact_idConjunto_compact")

        if score <= 0:
            continue

        field_matches.append(
            (
                score,
                item,
                {
                    "fallback_mode": "field_match",
                    "reasons": reasons,
                    "similarity": round(min(0.99, score / 220.0), 4),
                },
            )
        )

    if field_matches:
        field_matches.sort(key=lambda pair: pair[0], reverse=True)
        return field_matches, "field_match"

    fuzzy_matches: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    relaxed_fuzzy_matches: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    if not normalized_variant and not normalized_variant_compact:
        return [], "no_fuzzy_input"

    for item in salones:
        if not isinstance(item, dict):
            continue

        data = item.get("data") if isinstance(item.get("data"), dict) else {}
        fields: dict[str, str] = {
            "nomenclatura": _normalize_lookup_text(data.get("nomenclatura") or item.get("doc_id") or ""),
            "nombre": _normalize_lookup_text(data.get("nombre") or ""),
            "idConjunto": _normalize_lookup_text(data.get("idConjunto") or ""),
        }
        fields_compact: dict[str, str] = {key: _normalize_compact_text(value) for key, value in fields.items()}

        best_field = ""
        best_ratio = 0.0
        for field_name, field_value in fields.items():
            field_compact = fields_compact.get(field_name, "")
            if not field_value and not field_compact:
                continue
            ratio_norm = (
                SequenceMatcher(None, normalized_variant, field_value).ratio()
                if normalized_variant and field_value
                else 0.0
            )
            ratio_compact = (
                SequenceMatcher(None, normalized_variant_compact, field_compact).ratio()
                if normalized_variant_compact and field_compact
                else 0.0
            )
            ratio = max(ratio_norm, ratio_compact)
            if ratio > best_ratio:
                best_ratio = ratio
                best_field = field_name

        score = int(round(best_ratio * 90))
        payload = (
            score,
            item,
            {
                "fallback_mode": "fuzzy_match",
                "best_field": best_field,
                "similarity": round(best_ratio, 4),
                "threshold": SALON_FUZZY_MIN_RATIO,
            },
        )

        if best_ratio >= SALON_FUZZY_MIN_RATIO:
            fuzzy_matches.append(payload)
        elif best_ratio >= SALON_RELAXED_FUZZY_MIN_RATIO:
            relaxed_fuzzy_matches.append(
                (
                    score,
                    item,
                    {
                        "fallback_mode": "fuzzy_relaxed",
                        "best_field": best_field,
                        "similarity": round(best_ratio, 4),
                        "threshold": SALON_RELAXED_FUZZY_MIN_RATIO,
                    },
                )
            )

    fuzzy_matches.sort(key=lambda pair: pair[0], reverse=True)
    if fuzzy_matches:
        return fuzzy_matches, "fuzzy_match"

    relaxed_fuzzy_matches.sort(key=lambda pair: pair[0], reverse=True)
    if relaxed_fuzzy_matches:
        return relaxed_fuzzy_matches[:12], "fuzzy_relaxed"

    return [], "no_match"


@tool("buscar_salones_idit")
def buscar_salones_idit(query: str, max_matches: int = 5, include_schedule: bool = True) -> str:
    """Busca salones y servicios del IDIT en datos operativos en vivo.

    Usa esta tool cuando la consulta del usuario trate de ubicacion de salones,
    clases y horarios, disponibilidad, responsables o equipamiento de espacios del IDIT
    (por ejemplo FABLAB, laboratorios o codigos tipo J-001-A, J001, j001, J-001,j-).
    Si no ecnuentras el codigo o resultado exacto regresa el que tenga mayor coincidencia.

    Args:
        query: Consulta libre del usuario en espanol. Puede incluir nombre de servicio,
            codigo de salon, clase/materia, responsable, horario o palabras de ubicacion.
        max_matches: Numero maximo de coincidencias a devolver. Se acota internamente
            para evitar payloads excesivos.
        include_schedule: Si es True, incluye bloques de horario compactados por salon.
            Si es False, devuelve solo resumen de horario.

    Returns:
        str: JSON serializado con estructura:
            - query: texto consultado.
            - total_matches: total de coincidencias devueltas.
            - matches: lista de salones ordenada por relevancia (score desc) con campos
                            como nomenclatura, nombre, tipo, ubicacion_aproximada, horario_resumen,
                            clases_resumen, responsables, equipamiento y, opcionalmente, horario.
    """
    snapshot = fetch_firebase_live_snapshot()
    salones = snapshot.get("salones", []) if isinstance(snapshot, dict) else []

    limit = max(1, min(max_matches, 12))
    normalized_query = _normalize_salon_codes_in_text(query)
    query_variants = _build_salon_query_variants(query)
    if not query_variants:
        query_variants = [str(query or "").strip()]

    attempted_queries: list[str] = []
    attempts_debug: list[dict[str, Any]] = []

    # Acumulamos por documento para conservar el mejor score entre variantes.
    best_by_doc: dict[str, tuple[int, dict[str, Any], dict[str, Any], str]] = {}

    for idx, variant in enumerate(query_variants, start=1):
        attempted_queries.append(variant)
        firestore_query = {
            "collection": "salones",
            "attempt_index": idx,
            "query": variant,
            "normalized_query": _normalize_salon_codes_in_text(variant),
        }

        _emit_structured_debug(
            "tool.buscar_salones_idit.firestore_query",
            firestore_query=firestore_query,
            include_schedule=include_schedule,
        )

        scored = _score_salones_for_query_variant(salones, variant)
        search_mode = "score"
        if not scored:
            scored, search_mode = _fallback_match_salones_by_fields(salones, variant)

        attempts_debug.append(
            {
                "query": variant,
                "search_mode": search_mode,
                "matches_found": len(scored),
                "firestore_query": firestore_query,
            }
        )

        for score, item, score_debug in scored[: max(limit * 2, 12)]:
            data = item.get("data") if isinstance(item.get("data"), dict) else {}
            doc_id = str(item.get("doc_id") or data.get("nomenclatura") or "").strip()
            if not doc_id:
                continue

            enriched_debug = dict(score_debug or {})
            enriched_debug["matched_by_query"] = variant
            enriched_debug["search_mode"] = search_mode

            previous = best_by_doc.get(doc_id)
            if previous is None or score > previous[0]:
                best_by_doc[doc_id] = (score, item, enriched_debug, variant)

    selected = sorted(
        best_by_doc.values(),
        key=lambda entry: (
            entry[0],
            float(entry[2].get("coverage_ratio") or 0.0),
            int(entry[2].get("specific_terms_matched") or 0),
        ),
        reverse=True,
    )[:limit]

    if not selected and salones:
        rescue_scored, rescue_mode = _fallback_match_salones_by_fields(salones, query)
        for score, item, score_debug in rescue_scored[: max(limit * 2, 12)]:
            data = item.get("data") if isinstance(item.get("data"), dict) else {}
            doc_id = str(item.get("doc_id") or data.get("nomenclatura") or "").strip()
            if not doc_id:
                continue
            enriched_debug = dict(score_debug or {})
            enriched_debug["matched_by_query"] = query
            enriched_debug["search_mode"] = rescue_mode
            best_by_doc[doc_id] = (score, item, enriched_debug, query)

        selected = sorted(
            best_by_doc.values(),
            key=lambda entry: (
                entry[0],
                float(entry[2].get("coverage_ratio") or 0.0),
                int(entry[2].get("specific_terms_matched") or 0),
            ),
            reverse=True,
        )[:limit]

    matches_payload: list[dict[str, Any]] = []
    for score, item, score_debug, matched_query in selected:
        serialized = _serialize_salon_for_tool(item, score, include_schedule=include_schedule)
        similarity = _resolve_match_similarity(score, score_debug)
        serialized["similarity"] = similarity
        serialized["match_confidence"] = _label_match_confidence(similarity)
        serialized["search_mode"] = str(score_debug.get("search_mode") or "score")
        serialized["matched_query"] = matched_query
        if AGENT_DEBUG_TRACE:
            serialized["score_debug"] = {
                key: value
                for key, value in score_debug.items()
                if key in {"search_mode", "matched_by_query", "reasons", "best_field", "similarity", "top_score_components"}
            }
        matches_payload.append(serialized)

    payload = {
        "query": query,
        "normalized_query": normalized_query,
        "attempted_queries": attempted_queries,
        "total_attempts": len(attempted_queries),
        "total_matches": len(matches_payload),
        "matches": matches_payload,
        "debug_info": {
            "source": "firebase_live_snapshot_memory_scan",
            "firestore_queries": [attempt.get("firestore_query") for attempt in attempts_debug],
            "attempts": attempts_debug,
        },
    }

    return json.dumps(payload, ensure_ascii=False)


@tool("buscar_personal_idit")
def buscar_personal_idit(query: str, max_matches: int = 5, include_schedule: bool = True) -> str:
    """Busca personal academico del IDIT y su contexto operativo basico.

    Usa esta tool cuando el usuario pregunte por profesores/personal (quien es,
    correo, rol, tipo, en que salon opera o su disponibilidad general).

    Args:
        query: Consulta libre del usuario en espanol sobre personas del IDIT.
        max_matches: Numero maximo de coincidencias a devolver, ordenadas por score.
        include_schedule: Si es True, agrega horario compactado por persona.
            Si es False, entrega solo horario_resumen.

    Returns:
        str: JSON serializado con estructura:
            - query: texto consultado.
            - total_matches: total de coincidencias devueltas.
            - matches: lista de personal con campos como nombre, rol, tipo, correo,
              salon_principal, salones_relacionados_total, horario_resumen y,
              opcionalmente, horario.
    """
    snapshot = fetch_firebase_live_snapshot()
    usuarios = snapshot.get("usuarios", []) if isinstance(snapshot, dict) else []

    normalized_query = _normalize_lookup_text(query)
    terms = _tokenize_for_tool_search(query)

    scored: list[tuple[int, dict[str, Any]]] = []
    for item in usuarios:
        if not isinstance(item, dict):
            continue
        score = _score_user_item(item, normalized_query, terms)
        if score > 0:
            scored.append((score, item))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    limit = max(1, min(max_matches, 12))
    selected = scored[:limit]

    payload = {
        "query": query,
        "normalized_query": _normalize_salon_codes_in_text(query),
        "attempted_queries": [query],
        "total_attempts": 1,
        "total_matches": len(selected),
        "matches": [
            _serialize_user_for_tool(item, score, include_schedule=include_schedule)
            for score, item in selected
        ],
        "debug_info": {
            "source": "firebase_live_snapshot_memory_scan",
            "firestore_queries": [
                {
                    "collection": "usuarios",
                    "query": query,
                }
            ],
        },
    }

    return json.dumps(payload, ensure_ascii=False)


@tool("obtener_agenda_personal_idit")
def obtener_agenda_personal_idit(query: str, max_matches: int = 3, include_details: bool = True) -> str:
    """Obtiene agenda operativa completa de personal academico del IDIT.

    Usa esta tool cuando la consulta requiera detalle de agenda: horario,
    calendario, citas, notificaciones y salones asociados de una persona.

    Args:
        query: Consulta libre del usuario en espanol orientada a agenda personal
            (por ejemplo "agenda del profesor X", "calendario", "citas").
        max_matches: Numero maximo de personas a devolver (top por relevancia).
        include_details: Si es True, incluye detalle compactado de horario,
            calendario, citas y notificaciones. Si es False, devuelve solo
            indicadores y totales resumidos.

    Returns:
        str: JSON serializado con estructura:
            - query: texto consultado.
            - total_matches: total de coincidencias devueltas.
            - matches: lista de personas con identidad basica (nombre, rol, tipo,
              correo), relacion con salones, horario_resumen y estado de agenda
              (calendario/citas/notificaciones), con detalle opcional segun
              include_details.
    """
    snapshot = fetch_firebase_live_snapshot()
    usuarios = snapshot.get("usuarios", []) if isinstance(snapshot, dict) else []

    normalized_query = _normalize_lookup_text(query)
    terms = _tokenize_for_tool_search(query)

    scored: list[tuple[int, dict[str, Any]]] = []
    for item in usuarios:
        if not isinstance(item, dict):
            continue
        score = _score_user_item(item, normalized_query, terms)
        if score > 0:
            scored.append((score, item))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    limit = max(1, min(max_matches, 10))
    selected = scored[:limit]

    payload = {
        "query": query,
        "normalized_query": _normalize_salon_codes_in_text(query),
        "attempted_queries": [query],
        "total_attempts": 1,
        "total_matches": len(selected),
        "matches": [
            _serialize_personal_agenda_for_tool(item, score, include_details=include_details)
            for score, item in selected
        ],
        "debug_info": {
            "source": "firebase_live_snapshot_memory_scan",
            "firestore_queries": [
                {
                    "collection": "usuarios/horarios",
                    "query": query,
                }
            ],
        },
    }

    return json.dumps(payload, ensure_ascii=False)


def _normalize_salon_code(value: Any) -> str:
    canonical = _canonicalize_salon_code(value)
    base = canonical if canonical else str(value or "")
    return _normalize_lookup_text(base).replace(" ", "").replace("_", "").replace("-", "")


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    chroma_context: str
    firebase_context: str
    question: str
    original_question: str
    debug_trace: dict[str, Any]


def _truncate_history_text(text: str, max_chars: int = 350) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[: max_chars - 3]}..."


def _truncate_context_block(text: str, max_chars: int) -> str:
    """Recorta contexto largo para evitar rebasar el limite de tokens del modelo."""
    payload = str(text or "")
    if len(payload) <= max_chars:
        return payload

    if max_chars <= 120:
        return payload[:max_chars]

    marker = "\n...[contexto truncado por limite de tokens]...\n"
    keep_each = max(20, (max_chars - len(marker)) // 2)
    head = payload[:keep_each]
    tail = payload[-keep_each:]
    clipped = f"{head}{marker}{tail}"
    return clipped[:max_chars]


def _history_to_messages(historial: list[dict]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for turno in historial[-4:]:
        pregunta_hist = _truncate_history_text(str(turno.get("pregunta") or ""), max_chars=350)
        respuesta_hist = _truncate_history_text(str(turno.get("respuesta") or ""), max_chars=350)
        if pregunta_hist:
            messages.append(HumanMessage(content=pregunta_hist))
        if respuesta_hist:
            messages.append(AIMessage(content=respuesta_hist))
    return messages


def _preview_json(value: Any, *, max_chars: int = 280) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        text = str(value)
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[: max_chars - 3]}..."


def _extract_tool_message_preview(content: Any, *, max_chars: int = 320) -> str:
    if isinstance(content, str):
        return _truncate_history_text(content, max_chars=max_chars)
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str) and part.strip():
                parts.append(part.strip())
            elif isinstance(part, dict):
                text_part = str(part.get("text") or "").strip()
                if text_part:
                    parts.append(text_part)
        return _truncate_history_text(" ".join(parts), max_chars=max_chars)
    return _truncate_history_text(str(content), max_chars=max_chars)


def _print_pretty_debug_trace(*, question: str, final_state: dict[str, Any], final_text: str) -> None:
    debug_trace = final_state.get("debug_trace") if isinstance(final_state, dict) else {}
    if not isinstance(debug_trace, dict):
        debug_trace = {}

    totals = debug_trace.get("firebase_totals") if isinstance(debug_trace.get("firebase_totals"), dict) else {}
    firebase_error = str(debug_trace.get("firebase_error") or "").strip()
    firebase_context_chars = int(debug_trace.get("firebase_context_chars") or 0)
    firebase_context_chars_prompt = int(debug_trace.get("firebase_context_chars_prompt") or firebase_context_chars)

    print("\n" + "=" * 72)
    print("TRACE LANGGRAPH (DEBUG)")
    print("=" * 72)
    print(f"Pregunta: {question}")
    print(
        "Contexto Firebase: "
        f"salones={totals.get('salones', 0)} | "
        f"usuarios_objetivo={totals.get('usuarios_objetivo', 0)} | "
        f"usuarios_all={totals.get('usuarios_all', 0)} | "
        f"chars_prompt={firebase_context_chars_prompt} | chars_total={firebase_context_chars}"
    )
    if firebase_error:
        print(f"Firebase warning: {firebase_error}")

    messages = final_state.get("messages") if isinstance(final_state, dict) else []
    if not isinstance(messages, list):
        messages = []

    print("\nPasos de tools:")
    step = 0
    for message in messages:
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            for tool_call in message.tool_calls:
                step += 1
                tool_name = str(tool_call.get("name") or "tool_sin_nombre")
                args_preview = _preview_json(tool_call.get("args") or {})
                print(f"  {step}. CALL  {tool_name}")
                print(f"     args: {args_preview}")
        elif isinstance(message, ToolMessage):
            tool_name = str(getattr(message, "name", "") or "tool_resultado")
            preview = _extract_tool_message_preview(getattr(message, "content", ""))
            print(f"     RESULT {tool_name}: {preview}")

    if step == 0:
        print("  (sin invocaciones de tools)")

    graph_events = debug_trace.get("events") if isinstance(debug_trace.get("events"), list) else []
    if graph_events:
        print("\nEventos de grafo:")
        for idx, event in enumerate(graph_events[-12:], start=1):
            if not isinstance(event, dict):
                continue
            event_name = str(event.get("event") or "evento")
            event_payload = {
                key: value
                for key, value in event.items()
                if key != "event"
            }
            print(f"  {idx}. {event_name}: {_preview_json(event_payload, max_chars=220)}")

    print("\nRespuesta final:")
    print(_truncate_history_text(final_text, max_chars=900))
    print("=" * 72 + "\n")


def _append_graph_event(debug_trace: dict[str, Any], event: str, **payload: Any) -> dict[str, Any]:
    """Acumula eventos estructurados del flujo LangGraph en el estado."""
    existing_events = debug_trace.get("events") if isinstance(debug_trace.get("events"), list) else []
    events = [item for item in existing_events if isinstance(item, dict)]
    events.append({"event": event, **payload})

    # Mantener buffer acotado para no inflar el estado.
    if len(events) > 40:
        events = events[-40:]

    updated = dict(debug_trace)
    updated["events"] = events
    _emit_structured_debug(f"graph.{event}", **payload)
    return updated


def build_graph(vector_store: Chroma, frontend_context: str | None = None):
    llm = ChatGroq(model=GROQ_MODEL, temperature=0.1)
    
    # Definir tools de ChromaDB con closure sobre vector_store
    @tool("buscar_equipos_idit")
    def buscar_equipos_idit(query: str, seccion: str = "", chunk_type: str = "") -> str:
        """Busca equipos y máquinas del IDIT en las bitácoras de mantenimiento.

        Usa esta tool cuando el usuario pregunte sobre equipos, máquinas, instrumentos
        o herramientas del IDIT: qué hace un equipo, dónde está, quién lo opera,
        sus características técnicas, especificaciones, materiales, plan de mantenimiento
        o uso permitido.

        Args:
            query: Consulta libre en español sobre el equipo o máquina.
            seccion: Filtro opcional por sección del IDIT (ej. "FABLAB", "ELECTRONICA").
                     Dejar vacío para buscar en todas las secciones.
            chunk_type: Filtro opcional por tipo de chunk:
                        "descripcion" para info técnica general,
                        "mantenimiento_interno" para plan de mantenimiento,
                        "mantenimiento_externo" para historial externo.
                        Dejar vacío para buscar en todos los tipos.

        Returns:
            str: JSON con lista de equipos encontrados y su información estructurada.
        """
        where_filter = {"fuente": {"$eq": "bitacora_mantenimiento"}}
        if seccion.strip():
            where_filter = {
                "$and": [
                    where_filter,
                    {"seccion": {"$eq": seccion.strip().upper()}},
                ]
            }
        if chunk_type.strip():
            where_filter = {
                "$and": [
                    where_filter,
                    {"chunk_type": {"$eq": chunk_type.strip()}},
                ]
            }

        docs = vector_store.similarity_search(query, k=6, filter=where_filter)

        matches = []
        seen_machines = set()
        for doc in docs:
            meta = doc.metadata
            machine_key = f"{meta.get('maquina','')}_{meta.get('chunk_type','')}"
            if machine_key in seen_machines:
                continue
            seen_machines.add(machine_key)
            matches.append({
                "maquina": meta.get("maquina", ""),
                "seccion": meta.get("seccion", ""),
                "ubicacion": meta.get("ubicacion", ""),
                "responsable": meta.get("responsable", ""),
                "actualizacion": meta.get("actualizacion", ""),
                "chunk_type": meta.get("chunk_type", ""),
                "contenido": doc.page_content[:1200],
            })

        payload = {
            "query": query,
            "attempted_queries": [query],
            "total_attempts": 1,
            "total_matches": len(matches),
            "matches": matches,
            "debug_info": {
                "source": "chroma_similarity_search",
                "collection": COLLECTION_NAME,
            },
        }
        return json.dumps(payload, ensure_ascii=False)

    @tool("buscar_informacion_idit")
    def buscar_informacion_idit(query: str, categoria: str = "") -> str:
        """Busca información general e institucional del IDIT.

        Usa esta tool cuando el usuario pregunte sobre qué es el IDIT, sus servicios,
        tecnologías disponibles, áreas de trabajo, programas académicos, descripción
        general de laboratorios, información de contacto, o cualquier información
        de tipo institucional que no sea sobre un equipo o máquina específica ni
        sobre salones, horarios o personal.

        Args:
            query: Consulta libre en español sobre información del IDIT.
            categoria: Filtro opcional por categoría del documento.
                       Dejar vacío para buscar en todos los documentos de conocimiento.

        Returns:
            str: JSON con fragmentos de información relevante encontrados.
        """
        where_filter = {"fuente": {"$eq": "web_scraping"}}
        if categoria.strip():
            where_filter = {
                "$and": [
                    where_filter,
                    {"categoria": {"$eq": categoria.strip()}},
                ]
            }

        docs = vector_store.similarity_search(query, k=5, filter=where_filter)

        matches = []
        for doc in docs:
            meta = doc.metadata
            matches.append({
                "titulo": meta.get("titulo", meta.get("url", "")),
                "categoria": meta.get("categoria", ""),
                "url": meta.get("url", ""),
                "fecha": meta.get("fecha_normalizacion_utc", ""),
                "contenido": doc.page_content[:1000],
            })

        payload = {
            "query": query,
            "attempted_queries": [query],
            "total_attempts": 1,
            "total_matches": len(matches),
            "matches": matches,
            "debug_info": {
                "source": "chroma_similarity_search",
                "collection": COLLECTION_NAME,
            },
        }
        return json.dumps(payload, ensure_ascii=False)
    
    tools = [
        buscar_salones_idit,
        buscar_personal_idit,
        obtener_agenda_personal_idit,
        buscar_equipos_idit,
        buscar_informacion_idit,
    ]
    model_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    def retrieve_context(state: AgentState) -> dict[str, Any]:
        question = str(state.get("question") or "").strip()
        original_question = str(state.get("original_question") or question).strip()

        if not question and original_question:
            question = _normalize_salon_codes_in_text(original_question)

        firebase_runtime = build_firebase_runtime_context(
            question,
            frontend_context=frontend_context,
        )
        firebase_context = str(firebase_runtime.get("context_text") or "Sin datos operativos")
        raw_context_limit = os.getenv("AGENT_FIREBASE_CONTEXT_MAX_CHARS", "1600")
        try:
            firebase_context_max_chars = max(600, int(str(raw_context_limit).strip() or "1600"))
        except ValueError:
            firebase_context_max_chars = 1600
        firebase_context_for_prompt = _truncate_context_block(firebase_context, firebase_context_max_chars)
        debug_trace = dict(state.get("debug_trace") or {})
        debug_trace["firebase_totals"] = (
            firebase_runtime.get("totals") if isinstance(firebase_runtime.get("totals"), dict) else {}
        )
        debug_trace["firebase_error"] = str(firebase_runtime.get("error") or "")
        debug_trace["firebase_context_chars"] = len(firebase_context)
        debug_trace["firebase_context_chars_prompt"] = len(firebase_context_for_prompt)
        debug_trace = _append_graph_event(
            debug_trace,
            "retrieve_context",
            question_original=original_question,
            question_normalized=question,
            firebase_totals=debug_trace.get("firebase_totals") or {},
            firebase_prompt_chars=len(firebase_context_for_prompt),
        )

        context_block = (
            "[DATOS OPERATIVOS EN VIVO - FIREBASE]\n"
            f"{firebase_context_for_prompt}\n\n"
            "[PREGUNTA ORIGINAL]\n"
            f"{original_question}\n\n"
            "[PREGUNTA NORMALIZADA]\n"
            f"{question}\n\n"
            "[PREGUNTA]\n"
            f"{question}"
        )

        return {
            "chroma_context": "",
            "firebase_context": firebase_context_for_prompt,
            "messages": [HumanMessage(content=context_block)],
            "question": question,
            "original_question": original_question,
            "debug_trace": debug_trace,
        }

    def agent(state: AgentState) -> dict[str, Any]:
        response = model_with_tools.invoke(state["messages"])
        tool_calls = []
        if isinstance(response, AIMessage) and getattr(response, "tool_calls", None):
            tool_calls = [str(call.get("name") or "") for call in response.tool_calls]

        debug_trace = _append_graph_event(
            dict(state.get("debug_trace") or {}),
            "agent_response",
            has_tool_calls=bool(tool_calls),
            tool_calls=tool_calls,
            message_type=type(response).__name__,
        )

        return {
            "messages": [response],
            "debug_trace": debug_trace,
        }

    def should_continue(state: AgentState) -> str:
        messages = state.get("messages") or []
        if not messages:
            _emit_structured_debug("graph.route", decision="end", reason="empty_messages")
            return "end"

        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
            _emit_structured_debug("graph.route", decision="tools", reason="model_tool_calls")
            return "tools"

        _emit_structured_debug("graph.route", decision="end", reason="no_tool_calls")
        return "end"

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("agent", agent)
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("retrieve_context")
    workflow.add_edge("retrieve_context", "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


def _sanitize_user_facing_response(text: str) -> str:
    """Elimina phrasing tecnico para mantener respuestas naturales al usuario."""
    if not text:
        return ""

    cleaned = str(text)
    substitutions: list[tuple[str, str]] = [
        (r"seg[uú]n\s+(el\s+)?contexto\s+de\s+firebase", "con la informacion disponible"),
        (r"de\s+acuerdo\s+con\s+firebase", "con la informacion disponible"),
        (r"seg[uú]n\s+(el\s+)?json", "con la informacion disponible"),
        (r"fired?base_operativo", ""),
        (r"datos_operativos_universidad", ""),
        (r"cruce_front_mapa_salones_json", ""),
        (r"referencias_textuales_sugeridas", ""),
        (r"contexto\s+firebase", "informacion disponible"),
        (r"contexto\s+chroma", "informacion disponible"),
        (r"contexto\s+siis\s+frontend", "informacion disponible"),
    ]

    for pattern, replacement in substitutions:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"(?:\s*\n\s*){3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()

    return cleaned or "No tengo esa información disponible."


def inicializar_vector_store() -> Chroma:
    """Carga el vector store desde Chroma Cloud."""
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


def generar_respuesta_rag(
    vector_store: Chroma,
    pregunta: str,
    historial: list[dict],
    frontend_context: str | None = None,
) -> str:
    """Responde con RAG usando un flujo de estado en LangGraph."""
    pregunta_original = str(pregunta or "").strip()
    pregunta_normalizada = _normalize_salon_codes_in_text(pregunta_original)

    graph = build_graph(vector_store, frontend_context=frontend_context)
    initial_state: AgentState = {
        "messages": [SystemMessage(content=SYSTEM_PROMPT), *_history_to_messages(historial)],
        "chroma_context": "",
        "firebase_context": "",
        "question": pregunta_normalizada,
        "original_question": pregunta_original,
        "debug_trace": {
            "question_original": pregunta_original,
            "question_normalized": pregunta_normalizada,
        },
    }

    if AGENT_DEBUG_TRACE:
        final_state = graph.invoke(initial_state)
    else:
        final_state = graph.invoke(initial_state)

    messages = final_state.get("messages") if isinstance(final_state, dict) else []
    final_text = ""

    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, AIMessage):
                continue

            content = getattr(message, "content", "")
            if isinstance(content, str):
                final_text = content.strip()
            elif isinstance(content, list):
                parts: list[str] = []
                for part in content:
                    if isinstance(part, str) and part.strip():
                        parts.append(part.strip())
                    elif isinstance(part, dict):
                        text_part = str(part.get("text") or "").strip()
                        if text_part:
                            parts.append(text_part)
                final_text = " ".join(parts).strip()
            else:
                final_text = str(content).strip()

            if final_text:
                break

    if not final_text:
        return "No tengo esa información disponible."

    if AGENT_DEBUG_TRACE and isinstance(final_state, dict):
        _print_pretty_debug_trace(
            question=pregunta_original,
            final_state=final_state,
            final_text=final_text,
        )

    return _sanitize_user_facing_response(final_text)


def chat_loop(vector_store: Chroma):
    """Bucle de conversación en terminal."""
    print("\n" + "═" * 55)
    print("  🎓 Asistente Universitario  ")
    print("  Escribe 'salir' o 'exit' para terminar")
    print("═" * 55 + "\n")

    historial = []

    while True:
        try:
            pregunta = input("Tú: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n¡Hasta luego!")
            break

        if not pregunta:
            continue

        if pregunta.lower() in {"salir", "exit", "quit", "q"}:
            print("¡Hasta luego!")
            break

        try:
            respuesta = generar_respuesta_rag(vector_store, pregunta, historial)
        except Exception as e:
            respuesta = f"Ocurrió un error al procesar tu pregunta: {e}"

        print(f"\nAsistente: {respuesta}\n")
        historial.append({"pregunta": pregunta, "respuesta": respuesta})


def main():
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: Falta GOOGLE_API_KEY en el archivo .env")
        return
    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: Falta GROQ_API_KEY en el archivo .env")
        return
    if not CHROMA_API_KEY:
        print("ERROR: Falta CHROMA_API_KEY en el archivo .env")
        return
    if not CHROMA_TENANT:
        print("ERROR: Falta CHROMA_TENANT en el archivo .env")
        return
    if not CHROMA_DATABASE:
        print("ERROR: Falta CHROMA_DATABASE en el archivo .env")
        return

    print("Iniciando asistente universitario...")
    try:
        vector_store = inicializar_vector_store()
        total = vector_store._collection.count()
        print(f"Base de datos lista: {total} fragmentos indexados")
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        return

    logging.info(f"Modelo de chat configurado (Groq): {GROQ_MODEL}")
    chat_loop(vector_store)


if __name__ == "__main__":
    main()
