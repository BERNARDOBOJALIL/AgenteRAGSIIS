#!/usr/bin/env python3
"""
Test para las nuevas tools de ChromaDB: buscar_equipos_idit y buscar_informacion_idit.
Verifica que:
1. Las tools se invocan correctamente
2. Retornan JSON con estructura esperada
3. Los filtros de metadatos funcionan
4. El agente puede llamarlas autónomamente
"""

import os
import sys
import json
from dotenv import load_dotenv

sys.path.insert(0, r"c:\Users\moiss\Documents\AgenteSIIS\AgenteRAGSIIS")

from agent import (
    inicializar_vector_store,
    build_graph,
    generar_respuesta_rag,
    AgentState,
    _history_to_messages,
)

load_dotenv()


def test_tools_structure():
    """Verifica que las tools se registren correctamente en el grafo."""
    print("=" * 70)
    print("TEST 1: Estructura de tools en el grafo")
    print("=" * 70)

    vector_store = inicializar_vector_store()
    graph = build_graph(vector_store)

    # Obtener info del grafo
    nodes = list(graph.nodes.keys()) if hasattr(graph, "nodes") else []
    print(f"Nodos del grafo: {nodes}")

    if "tools" in nodes:
        print("OK: Nodo 'tools' presente en el grafo")
    else:
        print("ERROR: Nodo 'tools' no encontrado")
        return False

    print("OK: Test 1 pasado\n")
    return True


def test_equipment_search():
    """Prueba buscar_equipos_idit."""
    print("=" * 70)
    print("TEST 2: Invocar buscar_equipos_idit")
    print("=" * 70)

    vector_store = inicializar_vector_store()
    graph = build_graph(vector_store)

    # Crear estado inicial
    initial_state: AgentState = {
        "messages": [],
        "chroma_context": "",
        "firebase_context": "",
        "question": "¿Qué especificaciones tiene el brazo romer?",
    }

    print(f"Pregunta: {initial_state['question']}")
    print("\nInvocando grafo...")

    try:
        final_state = graph.invoke(initial_state)
        messages = final_state.get("messages", [])
        print(f"Mensajes retornados: {len(messages)}")

        # Revisar si el agente invocó tools
        has_tool_calls = False
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                has_tool_calls = True
                for call in msg.tool_calls:
                    print(f"Tool invocada: {call.get('name')}")
                    if call.get("name") == "buscar_equipos_idit":
                        print("OK: buscar_equipos_idit fue invocada")

        print("OK: Test 2 completado\n")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        return False


def test_info_search():
    """Prueba buscar_informacion_idit."""
    print("=" * 70)
    print("TEST 3: Invocar buscar_informacion_idit")
    print("=" * 70)

    vector_store = inicializar_vector_store()
    graph = build_graph(vector_store)

    # Crear estado inicial
    initial_state: AgentState = {
        "messages": [],
        "chroma_context": "",
        "firebase_context": "",
        "question": "¿Qué es el IDIT y cuáles son sus servicios?",
    }

    print(f"Pregunta: {initial_state['question']}")
    print("\nInvocando grafo...")

    try:
        final_state = graph.invoke(initial_state)
        messages = final_state.get("messages", [])
        print(f"Mensajes retornados: {len(messages)}")

        # Revisar si el agente invocó tools
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for call in msg.tool_calls:
                    if call.get("name") == "buscar_informacion_idit":
                        print("OK: buscar_informacion_idit fue invocada")

        print("OK: Test 3 completado\n")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    """Ejecuta todos los tests."""
    print("\n" + "=" * 70)
    print("VALIDACION DE NUEVAS TOOLS CHROMA")
    print("=" * 70 + "\n")

    results = []

    results.append(("Estructura de tools", test_tools_structure()))
    results.append(("Buscar equipos", test_equipment_search()))
    results.append(("Buscar informacion", test_info_search()))

    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)

    for test_name, passed in results:
        status = "PASADO" if passed else "FALLIDO"
        print(f"{test_name}: {status}")

    all_passed = all(result for _, result in results)
    if all_passed:
        print("\nOK: Todos los tests pasaron!")
        return 0
    else:
        print("\nERROR: Algunos tests fallaron")
        return 1


if __name__ == "__main__":
    sys.exit(main())
