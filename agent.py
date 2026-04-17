import logging
import os
import re
import json
import unicodedata
from typing import Any

import chromadb
from dotenv import load_dotenv
from firebase_sources import build_firebase_runtime_context, fetch_firebase_live_snapshot
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq

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

SYSTEM_PROMPT = """Eres un asistente virtual amable y preciso de la universidad.
Tu trabajo es responder preguntas de los alumnos usando UNICAMENTE la informacion
disponible en el contexto proporcionado.

Reglas obligatorias:
- Responde siempre en espanol, claro y conciso.
- Si no hay datos suficientes, responde exactamente: "No tengo esa información disponible."
- Nunca inventes datos, horarios, nombres o requisitos.
- Nunca proporciones informacion personal o sensible de usuarios tipo alumno/estudiante.
- Si te piden informacion de un alumno, responde exactamente: "No tengo esa información disponible."
- Nunca menciones fuentes tecnicas, infraestructura o errores internos (Firebase, Chroma, API, JSON, permisos, logs, etc.).
- Nunca uses frases como "segun el contexto de Firebase", "segun el JSON" o equivalentes tecnicos.
- Para preguntas de salones, prioriza equipamiento y ubicacion aproximada (piso, zona y referencias cercanas).
- Si no hay equipamiento explicito pero existe equipamiento_inferido_conservador, puedes usarlo con lenguaje prudente ("es probable que", "podria contar con") y aclarar que es una estimacion.
- Solo comparte horario/calendario detallado cuando el usuario lo pida de forma explicita.
- Si piden como llegar, da indicaciones aproximadas y humanas; evita una ruta exacta paso a paso.
- Responde siempre en español.
- Se amable y usa un tono cercano pero profesional.
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

    score = 0
    if normalized_query and normalized_query in nombre_norm:
        score += 16
    if normalized_query and normalized_query in nomen_norm:
        score += 18
    if normalized_query and normalized_query in blob:
        score += 6

    for term in terms:
        if term in nombre_norm:
            score += 4
        elif term in nomen_norm:
            score += 5
        elif term in blob:
            score += 1

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

    scored: list[tuple[int, dict[str, Any]]] = []
    for item in salones:
        if not isinstance(item, dict):
            continue
        score = _score_salon_item(item, normalized_query, terms)
        if score > 0:
            scored.append((score, item))

    scored.sort(key=lambda pair: pair[0], reverse=True)

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
            for score, item in selected
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


def _question_needs_salon_disambiguation(question: str) -> bool:
    normalized = _normalize_lookup_text(question)
    markers = {
        "donde",
        "ubic",
        "horario",
        "disponible",
        "salon",
        "servicio",
        "fablab",
        "laboratorio",
        "clase",
        "clases",
        "materia",
        "materias",
        "grupo",
        "grupos",
    }
    return any(marker in normalized for marker in markers)


def _question_requests_salon_classes(question: str) -> bool:
    normalized = _normalize_lookup_text(question)
    markers = {
        "clase",
        "clases",
        "materia",
        "materias",
        "asignatura",
        "asignaturas",
        "curso",
        "cursos",
        "grupo",
        "grupos",
        "que se imparte",
        "que se da",
        "que dan",
    }
    return any(marker in normalized for marker in markers)


def _normalize_salon_code(value: Any) -> str:
    return _normalize_lookup_text(value).replace(" ", "").replace("_", "")


def _build_exact_salon_response(question: str, salones_payload: dict[str, Any]) -> str:
    code_match = SALON_CODE_PATTERN.search(question or "")
    if not code_match:
        return ""

    asked_code = _normalize_salon_code(code_match.group(0))
    matches = salones_payload.get("matches") if isinstance(salones_payload, dict) else None
    if not isinstance(matches, list) or not matches:
        return ""

    selected = None
    for item in matches:
        if not isinstance(item, dict):
            continue
        item_code = _normalize_salon_code(item.get("nomenclatura") or item.get("doc_id") or "")
        if item_code == asked_code:
            selected = item
            break

    if not isinstance(selected, dict):
        return ""

    codigo = str(selected.get("nomenclatura") or selected.get("doc_id") or "Sin código").strip()
    nombre = str(selected.get("nombre") or "Sin nombre").strip()
    piso = str(selected.get("piso") or "No disponible").strip()
    ubicacion = str(
        selected.get("ubicacion_aproximada")
        or selected.get("ubicacion_descriptiva")
        or "No disponible"
    ).strip()
    horario_resumen = str(selected.get("horario_resumen") or "No disponible").strip()

    piso_norm = _normalize_lookup_text(piso)
    ubicacion_norm = _normalize_lookup_text(ubicacion)
    if piso and piso != "No disponible" and piso_norm and piso_norm in ubicacion_norm:
        ubicacion_txt = ubicacion
    elif piso and piso != "No disponible" and ubicacion and ubicacion != "No disponible":
        ubicacion_txt = f"{piso}, {ubicacion}"
    elif piso and piso != "No disponible":
        ubicacion_txt = piso
    else:
        ubicacion_txt = ubicacion

    responsables = selected.get("responsables") if isinstance(selected.get("responsables"), list) else []
    responsables_txt = ", ".join(str(x).strip() for x in responsables[:2] if str(x).strip()) or "No disponible"

    clases_requested = _question_requests_salon_classes(question)
    clases_resumen = str(selected.get("clases_resumen") or "No disponible").strip()
    clases_line = ""
    if clases_requested:
        if clases_resumen and clases_resumen != "No disponible":
            clases_line = f" Clases registradas: {clases_resumen}."
        else:
            clases_line = " No hay clases registradas para ese salon en los datos actuales."

    return (
        f"{codigo} ({nombre}) se ubica en {ubicacion_txt}. "
        f"Horario: {horario_resumen}. "
        f"Responsable: {responsables_txt}."
        f"{clases_line}"
    )


def _build_multi_match_salon_response(question: str, salones_payload: dict[str, Any]) -> str:
    if not _question_needs_salon_disambiguation(question):
        return ""

    if SALON_CODE_PATTERN.search(question or ""):
        return ""

    matches = salones_payload.get("matches") if isinstance(salones_payload, dict) else None
    if not isinstance(matches, list) or len(matches) < 2:
        return ""

    first_name = _normalize_lookup_text(matches[0].get("nombre")) if isinstance(matches[0], dict) else ""
    same_name_matches = [
        item
        for item in matches
        if isinstance(item, dict) and first_name and _normalize_lookup_text(item.get("nombre")) == first_name
    ]

    selected = same_name_matches[:3] if len(same_name_matches) >= 2 else [m for m in matches[:3] if isinstance(m, dict)]
    if len(selected) < 2:
        return ""

    lines = [
        "Encontré varias coincidencias y las enumero para evitar ambigüedad:",
    ]
    include_clases = _question_requests_salon_classes(question)

    for idx, item in enumerate(selected, start=1):
        codigo = str(item.get("nomenclatura") or item.get("doc_id") or "Sin código").strip()
        nombre = str(item.get("nombre") or "Sin nombre").strip()
        piso = str(item.get("piso") or "No disponible").strip()
        ubicacion = str(item.get("ubicacion_aproximada") or "No disponible").strip()
        horario_resumen = str(item.get("horario_resumen") or "No disponible").strip()
        clases_resumen = str(item.get("clases_resumen") or "No disponible").strip()

        piso_norm = _normalize_lookup_text(piso)
        ubicacion_norm = _normalize_lookup_text(ubicacion)
        if piso and piso != "No disponible" and piso_norm and piso_norm in ubicacion_norm:
            ubicacion_txt = ubicacion
        elif piso and piso != "No disponible" and ubicacion and ubicacion != "No disponible":
            ubicacion_txt = f"{piso}, {ubicacion}"
        elif piso and piso != "No disponible":
            ubicacion_txt = piso
        else:
            ubicacion_txt = ubicacion

        equipamiento = item.get("equipamiento") if isinstance(item.get("equipamiento"), list) else []
        responsables = item.get("responsables") if isinstance(item.get("responsables"), list) else []
        equipamiento_txt = ", ".join(str(x).strip() for x in equipamiento[:4] if str(x).strip()) or "No disponible"
        responsables_txt = ", ".join(str(x).strip() for x in responsables[:2] if str(x).strip()) or "No disponible"

        clase_txt = f" Clases: {clases_resumen}." if include_clases else ""

        lines.append(
            f"{idx}. {codigo} - {nombre}. Ubicación: {ubicacion_txt}. Horario: {horario_resumen}. "
            f"Responsable: {responsables_txt}. Equipamiento: {equipamiento_txt}.{clase_txt}"
        )

    lines.append("Si me indicas el código exacto (por ejemplo, J-001 o J-001-A), te doy la respuesta puntual.")
    return "\n".join(lines)


def _question_requests_personal_info(question: str) -> bool:
    normalized = _normalize_lookup_text(question)
    markers = {
        "profesor",
        "profesora",
        "docente",
        "academico",
        "academica",
        "usuario",
        "personal",
        "agenda",
        "calendario",
        "citas",
        "notificaciones",
        "correo",
        "horario",
        "horarios",
        "opera",
        "responsable",
    }
    return any(marker in normalized for marker in markers)


def _build_personal_agenda_response(question: str, agenda_payload: dict[str, Any]) -> str:
    matches = agenda_payload.get("matches") if isinstance(agenda_payload, dict) else None
    if not isinstance(matches, list) or not matches:
        return ""

    if not _question_requests_personal_info(question):
        return ""

    selected = [item for item in matches[:3] if isinstance(item, dict)]
    if not selected:
        return ""

    if len(selected) == 1:
        item = selected[0]
        nombre = str(item.get("nombre") or "No disponible").strip()
        rol = str(item.get("rol") or "No disponible").strip()
        tipo = str(item.get("tipo") or "No disponible").strip()
        correo = str(item.get("correo") or "No disponible").strip()
        horario_resumen = str(item.get("horario_resumen") or "No disponible").strip()

        salon = item.get("salon_principal") if isinstance(item.get("salon_principal"), dict) else {}
        salones_rel = item.get("salones_relacionados") if isinstance(item.get("salones_relacionados"), list) else []
        salon_cod = str(salon.get("nomenclatura") or "").strip()
        salon_nom = str(salon.get("nombre") or "").strip()
        salon_piso = str(salon.get("piso") or "").strip()

        calendario_txt = "registrado" if bool(item.get("calendario_registrado")) else "sin registro"
        citas_total = int(item.get("citas_total") or 0)
        notifs_total = int(item.get("notificaciones_total") or 0)

        if salon_cod or salon_nom:
            salon_line = f"Opera principalmente en {salon_cod or salon_nom}"
            if salon_nom and salon_cod and salon_nom != salon_cod:
                salon_line += f" ({salon_nom})"
            if salon_piso:
                salon_line += f", {salon_piso}"
            salon_line += "."
        elif salones_rel:
            codigos = [
                str(s.get("nomenclatura") or s.get("nombre") or "").strip()
                for s in salones_rel[:3]
                if isinstance(s, dict)
            ]
            codigos = [c for c in codigos if c]
            if codigos:
                salon_line = f"Salones relacionados: {', '.join(codigos)}."
            else:
                salon_line = "No hay un salón principal claramente asociado."
        else:
            salon_line = "No hay un salón principal claramente asociado."

        return (
            f"Información de {nombre}: rol {rol}, tipo {tipo}. "
            f"Correo: {correo}. {salon_line} "
            f"Horario: {horario_resumen}. "
            f"Calendario: {calendario_txt}. "
            f"Citas registradas: {citas_total}. "
            f"Notificaciones registradas: {notifs_total}."
        )

    lines = ["Encontré varias coincidencias de personal y las enumero:"]
    for idx, item in enumerate(selected, start=1):
        nombre = str(item.get("nombre") or "No disponible").strip()
        tipo = str(item.get("tipo") or "No disponible").strip()
        horario_resumen = str(item.get("horario_resumen") or "No disponible").strip()
        salon = item.get("salon_principal") if isinstance(item.get("salon_principal"), dict) else {}
        salon_cod = str(salon.get("nomenclatura") or "Sin salón principal").strip()

        lines.append(
            f"{idx}. {nombre} ({tipo}). Salón principal: {salon_cod}. "
            f"Horario: {horario_resumen}."
        )

    lines.append("Si me indicas el nombre exacto, te doy el detalle completo de esa persona.")
    return "\n".join(lines)


def _run_operational_tools(question: str) -> dict[str, str]:
    salones_payload: dict[str, Any] = {"matches": []}
    usuarios_payload: dict[str, Any] = {"matches": []}
    agenda_payload: dict[str, Any] = {"matches": []}

    try:
        salones_raw = buscar_salones_idit.invoke(
            {
                "query": question,
                "max_matches": TOOL_MAX_MATCHES,
                "include_schedule": True,
            }
        )
        salones_payload = json.loads(salones_raw) if isinstance(salones_raw, str) else {}
    except Exception:
        salones_payload = {"matches": []}

    try:
        usuarios_raw = buscar_personal_idit.invoke(
            {
                "query": question,
                "max_matches": min(TOOL_MAX_MATCHES, 4),
                "include_schedule": True,
            }
        )
        usuarios_payload = json.loads(usuarios_raw) if isinstance(usuarios_raw, str) else {}
    except Exception:
        usuarios_payload = {"matches": []}

    try:
        agenda_raw = obtener_agenda_personal_idit.invoke(
            {
                "query": question,
                "max_matches": min(TOOL_MAX_MATCHES, 3),
                "include_details": True,
            }
        )
        agenda_payload = json.loads(agenda_raw) if isinstance(agenda_raw, str) else {}
    except Exception:
        agenda_payload = {"matches": []}

    direct_response = _build_personal_agenda_response(question, agenda_payload)
    if not direct_response:
        direct_response = _build_exact_salon_response(question, salones_payload)
    if not direct_response:
        direct_response = _build_multi_match_salon_response(question, salones_payload)

    tool_context_text = (
        "TOOLS_OPERATIVAS_IDIT\n"
        "salones_tool_json=\n"
        f"{_safe_json_with_budget(salones_payload, max_chars=4200)}\n"
        "personal_tool_json=\n"
        f"{_safe_json_with_budget(usuarios_payload, max_chars=2200)}\n"
        "agenda_personal_tool_json=\n"
        f"{_safe_json_with_budget(agenda_payload, max_chars=3000)}"
    )

    return {
        "direct_response": direct_response,
        "context_text": tool_context_text,
    }


def _registered_tools() -> list[Any]:
    return [buscar_salones_idit, buscar_personal_idit, obtener_agenda_personal_idit]


def _tool_map() -> dict[str, Any]:
    return {
        "buscar_salones_idit": buscar_salones_idit,
        "buscar_personal_idit": buscar_personal_idit,
        "obtener_agenda_personal_idit": obtener_agenda_personal_idit,
    }


def _safe_parse_tool_json(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _build_direct_response_from_payloads(question: str, payloads: dict[str, dict[str, Any]]) -> str:
    agenda_payload = payloads.get("obtener_agenda_personal_idit", {})
    agenda_response = _build_personal_agenda_response(question, agenda_payload)
    if agenda_response:
        return agenda_response

    salones_payload = payloads.get("buscar_salones_idit", {})
    exact_response = _build_exact_salon_response(question, salones_payload)
    if exact_response:
        return exact_response

    multi_response = _build_multi_match_salon_response(question, salones_payload)
    if multi_response:
        return multi_response

    return ""


def _build_tools_context_from_payloads(payloads: dict[str, dict[str, Any]]) -> str:
    salones_payload = payloads.get("buscar_salones_idit", {"matches": []})
    personal_payload = payloads.get("buscar_personal_idit", {"matches": []})
    agenda_payload = payloads.get("obtener_agenda_personal_idit", {"matches": []})

    return (
        "TOOLS_OPERATIVAS_IDIT\n"
        "salones_tool_json=\n"
        f"{_safe_json_with_budget(salones_payload, max_chars=4200)}\n"
        "personal_tool_json=\n"
        f"{_safe_json_with_budget(personal_payload, max_chars=2200)}\n"
        "agenda_personal_tool_json=\n"
        f"{_safe_json_with_budget(agenda_payload, max_chars=3000)}"
    )


def _is_operational_question(question: str) -> bool:
    return _question_needs_salon_disambiguation(question) or _question_requests_personal_info(question)


def _execute_tool_calls_react(
    model_with_tools: Any,
    messages: list[Any],
    max_rounds: int,
) -> tuple[Any, dict[str, dict[str, Any]], list[Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    current_messages = list(messages)
    tool_lookup = _tool_map()

    response = model_with_tools.invoke(current_messages)
    rounds = 0

    while rounds < max_rounds and getattr(response, "tool_calls", None):
        current_messages.append(response)

        for tool_call in response.tool_calls:
            tool_name = str(tool_call.get("name") or "")
            tool_args = tool_call.get("args") or {}
            tool_id = str(tool_call.get("id") or "")

            if tool_name not in tool_lookup:
                result_text = f"Tool {tool_name} no encontrada"
            else:
                try:
                    result = tool_lookup[tool_name].invoke(tool_args)
                    result_text = str(result)
                except Exception as exc:
                    result_text = f"Error ejecutando {tool_name}: {exc}"

            parsed_payload = _safe_parse_tool_json(result_text)
            if parsed_payload:
                payloads[tool_name] = parsed_payload

            if tool_id:
                current_messages.append(ToolMessage(content=result_text, tool_call_id=tool_id))

        response = model_with_tools.invoke(current_messages)
        rounds += 1

    return response, payloads, current_messages


def _compact_frontend_context(frontend_context: str | None, *, max_chars: int = 1500) -> str:
    """Compacta contexto opcional del frontend para no inflar tokens."""
    if not frontend_context:
        return ""

    compact = " ".join(str(frontend_context).split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[: max_chars - 3]}..."


def _clip_text_block(text: str | None, *, max_chars: int) -> str:
    if not text:
        return ""
    compact = " ".join(str(text).split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[: max_chars - 3]}..."


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
    """Responde con RAG hibrido usando ReAct con tools (salones/personal/agenda)."""
    resultados = vector_store.similarity_search(pregunta, k=3)

    contexto_chroma = []
    for doc in resultados:
        fuente = doc.metadata.get("fuente", "desconocida")
        categoria = doc.metadata.get("categoria", "general")
        contexto_chroma.append(f"[Fuente: {fuente} | Categoría: {categoria}]\n{doc.page_content}")

    firebase_context = build_firebase_runtime_context(
        pregunta,
        frontend_context=frontend_context,
    )

    contexto_firebase = firebase_context.get("context_text", "")
    frontend_context_txt = _compact_frontend_context(frontend_context)
    solicitud_horario = bool(firebase_context.get("schedule_requested", False))
    solicitud_horario_detallado = bool(firebase_context.get("schedule_details_requested", False))

    contexto_chroma_raw = "\n\n---\n\n".join(contexto_chroma) if contexto_chroma else "Sin contexto Chroma"
    contexto_chroma_txt = _clip_text_block(contexto_chroma_raw, max_chars=2200) or "Sin contexto Chroma"
    contexto_firebase_txt = _clip_text_block(contexto_firebase, max_chars=3200) or "Sin datos operativos"
    contexto_frontend_raw = frontend_context_txt if frontend_context_txt else "Sin contexto frontend"
    contexto_frontend_txt = _clip_text_block(contexto_frontend_raw, max_chars=900) or "Sin contexto frontend"

    react_system_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "Modo agente con tools:\n"
        "- Para consultas de salones, horarios, ubicaciones, personal academico, calendario, citas o notificaciones, usa tools antes de responder.\n"
        "- Tools disponibles: buscar_salones_idit, buscar_personal_idit, obtener_agenda_personal_idit.\n"
        "- Si hay varias coincidencias de un mismo salon/servicio, enumera las opciones para evitar ambiguedad.\n"
        "- Si no hay datos suficientes, responde exactamente: 'No tengo esa información disponible.'\n"
        "- Entrega solo el mensaje final para usuario; no muestres razonamiento interno ni pasos de tool-calling."
    )

    pregunta_contextual = (
        "Contexto de apoyo para responder:\n"
        f"solicitud_horario={str(solicitud_horario).lower()}\n"
        f"solicitud_horario_detallado={str(solicitud_horario_detallado).lower()}\n\n"
        f"Contexto Chroma:\n{contexto_chroma_txt}\n\n"
        f"Datos operativos (salones/personal):\n{contexto_firebase_txt}\n\n"
        f"Contexto SIIS frontend:\n{contexto_frontend_txt}\n\n"
        f"Pregunta del usuario: {pregunta}"
    )

    llm = ChatGroq(model=GROQ_MODEL, temperature=0.1)
    model_with_tools = llm.bind_tools(_registered_tools())

    messages: list[Any] = [SystemMessage(content=react_system_prompt)]
    for turno in historial[-3:]:
        pregunta_hist = str(turno.get("pregunta") or "").strip()
        respuesta_hist = str(turno.get("respuesta") or "").strip()
        if pregunta_hist:
            messages.append(HumanMessage(content=_clip_text_block(pregunta_hist, max_chars=280) or pregunta_hist))
        if respuesta_hist:
            messages.append(AIMessage(content=_clip_text_block(respuesta_hist, max_chars=500) or respuesta_hist))
    messages.append(HumanMessage(content=pregunta_contextual))

    respuesta, tool_payloads, _ = _execute_tool_calls_react(
        model_with_tools=model_with_tools,
        messages=messages,
        max_rounds=max(1, TOOL_MAX_ROUNDS),
    )

    direct_tool_response = _build_direct_response_from_payloads(pregunta, tool_payloads)
    if direct_tool_response:
        return _sanitize_user_facing_response(direct_tool_response)

    contexto_tools = _build_tools_context_from_payloads(tool_payloads) if tool_payloads else ""

    if _is_operational_question(pregunta) and not tool_payloads:
        fallback_tools = _run_operational_tools(pregunta)
        fallback_direct = str(fallback_tools.get("direct_response") or "").strip()
        if fallback_direct:
            return _sanitize_user_facing_response(fallback_direct)
        contexto_tools = str(fallback_tools.get("context_text") or "").strip()

    contenido = getattr(respuesta, "content", "")
    if isinstance(contenido, str) and contenido.strip():
        return _sanitize_user_facing_response(contenido)

    if contexto_tools:
        rescue_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            "Genera una respuesta final clara y concisa para el usuario con el contexto siguiente. "
            "No expliques herramientas internas.\n\n"
            f"{contexto_tools}\n\n"
            f"Pregunta: {pregunta}"
        )
        rescue_response = llm.invoke(rescue_prompt)
        rescue_content = getattr(rescue_response, "content", "")
        if isinstance(rescue_content, str) and rescue_content.strip():
            return _sanitize_user_facing_response(rescue_content)

    return "No tengo esa información disponible."


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
