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
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

try:
    from langchain_core.callbacks import ConsoleCallbackHandler
except ImportError:
    from langchain_core.tracers import ConsoleCallbackHandler

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
TOOL_MAX_MATCHES = int(os.getenv("AGENT_TOOL_MAX_MATCHES", "6"))
TOOL_MAX_ROUNDS = int(os.getenv("AGENT_TOOL_MAX_ROUNDS", "3"))
AGENT_DEBUG_TRACE = os.getenv("AGENT_DEBUG_TRACE", "").strip().lower() in {"1", "true", "yes", "on"}
SALON_CODE_PATTERN = re.compile(r"\bj[\s_-]?\d{1,3}(?:-[a-z])?\b", flags=re.IGNORECASE)
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
SALON_SHORT_QUERY_TOKENS = {"ia", "ai", "ar", "ra", "vr", "ml", "xr", "3d", "ti", "ux", "ui"}
SALON_TERM_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "ia": ("inteligencia", "artificial"),
    "ai": ("inteligencia", "artificial"),
    "ar": ("realidad", "aumentada"),
    "ra": ("realidad", "aumentada"),
    "vr": ("realidad", "virtual"),
    "ml": ("machine", "learning", "aprendizaje"),
    "fablab": ("fabricacion", "digital"),
    "labs": ("laboratorio",),
    "lab": ("laboratorio",),
}
SALON_PHRASE_HINTS: dict[str, tuple[str, ...]] = {
    "ia": ("inteligencia artificial",),
    "ai": ("inteligencia artificial",),
    "ar": ("realidad aumentada",),
    "ra": ("realidad aumentada",),
    "vr": ("realidad virtual",),
    "ml": ("machine learning", "aprendizaje automatico"),
}
SALON_GENERIC_TERMS = {
    "salon",
    "salones",
    "laboratorio",
    "laboratorios",
    "aula",
    "aulas",
    "proyecto",
    "proyectos",
    "espacio",
    "espacios",
}
SALON_HIGH_SIGNAL_TERMS = {
    "ia",
    "ai",
    "ar",
    "ra",
    "vr",
    "ml",
    "xr",
    "3d",
    "inteligencia",
    "artificial",
    "biometrica",
    "neuromarketing",
    "robotica",
    "aumentada",
    "virtual",
    "fabrica",
    "fabricacion",
    "digital",
}

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
"""


def _normalize_lookup_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


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
    if not term or len(term) < 6 or not tokens:
        return False

    term_prefix = term[:6]
    for token in tokens:
        if len(token) < 6:
            continue
        if abs(len(token) - len(term)) > 4:
            continue
        if token.startswith(term_prefix) or term.startswith(token[:6]):
            return True
        if SequenceMatcher(None, token, term).ratio() >= min_ratio:
            return True

    return False


def _expand_salon_query_terms(base_terms: set[str]) -> set[str]:
    expanded: set[str] = set()

    for raw_term in base_terms:
        term = _normalize_lookup_text(raw_term)
        if not term:
            continue

        expanded.add(term)

        if term.endswith("es") and len(term) > 4:
            expanded.add(term[:-2])
        elif term.endswith("s") and len(term) > 4:
            expanded.add(term[:-1])

        for extra in SALON_TERM_EXPANSIONS.get(term, ()): 
            expanded.add(extra)

    return expanded


def _salon_phrase_hints_from_terms(base_terms: set[str]) -> set[str]:
    hints: set[str] = set()
    for raw_term in base_terms:
        term = _normalize_lookup_text(raw_term)
        for phrase in SALON_PHRASE_HINTS.get(term, ()): 
            hints.add(phrase)
    return hints


def _salon_term_weight(term: str) -> int:
    if term in SALON_HIGH_SIGNAL_TERMS:
        return 6
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
    nomen_tokens = _normalized_word_tokens(nomen_norm)
    nombre_tokens = _normalized_word_tokens(nombre_norm)
    tipo_tokens = _normalized_word_tokens(tipo_norm)
    blob_tokens = _normalized_word_tokens(blob)

    short_terms = {
        tok
        for tok in re.split(r"[^a-z0-9]+", normalized_query)
        if len(tok) == 2 and tok in SALON_SHORT_QUERY_TOKENS
    }
    base_terms = {term for term in terms if term} | short_terms
    expanded_terms = _expand_salon_query_terms(base_terms)
    phrase_hints = _salon_phrase_hints_from_terms(base_terms)

    score = 0
    score_components: dict[str, int] = {}
    matched_terms: set[str] = set()
    matched_phrases: set[str] = set()
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
        if normalized_query in nombre_norm:
            add_component("exact_query_nombre", 22)
        if normalized_query in tipo_norm:
            add_component("exact_query_tipo", 16)
        elif normalized_query in blob:
            add_component("exact_query_blob", 7)

    code_match = SALON_CODE_PATTERN.search(normalized_query)
    if code_match:
        asked_code = _normalize_salon_code(code_match.group(0))
        item_code = _normalize_salon_code(nomenclatura)
        if asked_code and asked_code == item_code:
            add_component("exact_code_match", 120)
        elif asked_code and asked_code in item_code:
            add_component("partial_code_match", 28)

    for phrase in sorted(phrase_hints):
        if phrase in nombre_norm:
            add_component(f"phrase_nombre:{phrase}", 24)
            matched_phrases.add(phrase)
            field_hits["nombre"] += 1
        elif phrase in tipo_norm:
            add_component(f"phrase_tipo:{phrase}", 18)
            matched_phrases.add(phrase)
            field_hits["tipo"] += 1
        elif phrase in blob:
            add_component(f"phrase_blob:{phrase}", 9)
            matched_phrases.add(phrase)
            field_hits["blob"] += 1

    for term in sorted(expanded_terms):
        weight = _salon_term_weight(term)
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

    if matched_phrases and len(matched_terms) >= 2:
        add_component("phrase_alignment_bonus", 10)

    specific_terms = [term for term in matched_terms if term not in SALON_GENERIC_TERMS]
    if matched_terms and not specific_terms:
        add_component("generic_match_penalty", -4)

    if score < 0:
        score = 0

    top_components = [
        {"component": name, "value": value}
        for name, value in sorted(score_components.items(), key=lambda pair: abs(pair[1]), reverse=True)[:8]
    ]

    details: dict[str, Any] = {
        "base_terms": sorted(base_terms),
        "expanded_terms": sorted(expanded_terms),
        "matched_terms": sorted(matched_terms),
        "matched_phrases": sorted(matched_phrases),
        "coverage_ratio": round(coverage_ratio, 3),
        "specific_terms_matched": len(specific_terms),
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


@tool("buscar_salones_idit")
def buscar_salones_idit(query: str, max_matches: int = 5, include_schedule: bool = True) -> str:
    """Busca salones y servicios del IDIT en datos operativos en vivo.

    Usa esta tool cuando la consulta del usuario trate de ubicacion de salones,
    clases y horarios, disponibilidad, responsables o equipamiento de espacios del IDIT
    (por ejemplo FABLAB, laboratorios o codigos tipo J-001-A).

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

    normalized_query = _normalize_lookup_text(query)
    terms = _tokenize_for_tool_search(query)

    scored: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for item in salones:
        if not isinstance(item, dict):
            continue
        score, score_debug = _score_salon_item_detailed(item, normalized_query, terms)
        if score > 0:
            scored.append((score, item, score_debug))

    scored.sort(
        key=lambda pair: (
            pair[0],
            float(pair[2].get("coverage_ratio") or 0.0),
            int(pair[2].get("specific_terms_matched") or 0),
        ),
        reverse=True,
    )

    limit = max(1, min(max_matches, 12))
    if not scored:
        selected = []
    else:
        selected = scored[:limit]

    payload = {
        "query": query,
        "total_matches": len(selected),
        "matches": [
            _serialize_salon_for_tool(item, score, include_schedule=include_schedule)
            for score, item, _ in selected
        ],
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
        "total_matches": len(selected),
        "matches": [
            _serialize_user_for_tool(item, score, include_schedule=include_schedule)
            for score, item in selected
        ],
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
        "total_matches": len(selected),
        "matches": [
            _serialize_personal_agenda_for_tool(item, score, include_details=include_details)
            for score, item in selected
        ],
    }

    return json.dumps(payload, ensure_ascii=False)


def _normalize_salon_code(value: Any) -> str:
    return _normalize_lookup_text(value).replace(" ", "").replace("_", "")


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    chroma_context: str
    firebase_context: str
    question: str


def _truncate_history_text(text: str, max_chars: int = 350) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[: max_chars - 3]}..."


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
            "total_matches": len(matches),
            "matches": matches,
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
            "total_matches": len(matches),
            "matches": matches,
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

        firebase_runtime = build_firebase_runtime_context(
            question,
            frontend_context=frontend_context,
        )
        firebase_context = str(firebase_runtime.get("context_text") or "Sin datos operativos")

        context_block = (
            "[DATOS OPERATIVOS EN VIVO - FIREBASE]\n"
            f"{firebase_context}\n\n"
            "[PREGUNTA]\n"
            f"{question}"
        )

        return {
            "chroma_context": "",
            "firebase_context": firebase_context,
            "messages": [HumanMessage(content=context_block)],
        }

    def agent(state: AgentState) -> dict[str, Any]:
        response = model_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        messages = state.get("messages") or []
        if not messages:
            return "end"

        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
            return "tools"

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
    graph = build_graph(vector_store, frontend_context=frontend_context)
    initial_state: AgentState = {
        "messages": _history_to_messages(historial),
        "chroma_context": "",
        "firebase_context": "",
        "question": pregunta,
    }

    if AGENT_DEBUG_TRACE:
        config = RunnableConfig(callbacks=[ConsoleCallbackHandler()])
        final_state = graph.invoke(initial_state, config=config, debug=True)
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
