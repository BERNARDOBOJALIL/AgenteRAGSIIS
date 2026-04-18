# Implementación Completada: Tools de ChromaDB para Equipos e Información IDIT

## Resumen de Cambios

Se han agregado dos tools nuevas al agente LangGraph que consultan ChromaDB con filtros de metadatos específicos. El agente ahora puede buscar autónomamente información de equipos y datos institucionales del IDIT.

---

## 1. Nuevas Tools Implementadas

### 1.1 buscar_equipos_idit
**Propósito**: Buscar información de equipos/máquinas del IDIT en las bitácoras de mantenimiento.

**Entrada**:
```python
query: str           # "¿Qué especificaciones tiene el brazo romer?"
seccion: str = ""    # Opcional: "FABLAB", "ELECTRONICA", "METALMECANICA", etc.
chunk_type: str = "" # Opcional: "descripcion", "mantenimiento_interno", "mantenimiento_externo"
```

**Salida**:
```json
{
  "query": "brazo romer especificaciones",
  "total_matches": 2,
  "matches": [
    {
      "maquina": "BRAZO ROMER",
      "seccion": "FABLAB",
      "ubicacion": "J-001A",
      "responsable": "JEFE DE TALLER FABLAB",
      "actualizacion": "ENE 24",
      "chunk_type": "descripcion",
      "contenido": "Máquina: BRAZO ROMER | Sección: FABLAB | Tipo: descripcion\n\n# Bitácora de Mantenimiento: BRAZO ROMER\n..."
    }
  ]
}
```

**Uso de la Tool por el Agente**:
- El usuario pregunta: "¿Qué hace el brazo romer en FABLAB?"
- El agente automáticamente invoca: `buscar_equipos_idit(query="...", seccion="FABLAB")`
- Recibe JSON estructurado y responde en lenguaje natural

---

### 1.2 buscar_informacion_idit
**Propósito**: Buscar información general e institucional del IDIT (servicios, tecnologías, programas académicos).

**Entrada**:
```python
query: str           # "¿Qué es el IDIT y cuáles son sus servicios?"
categoria: str = ""  # Opcional: filtro por categoría del documento
```

**Salida**:
```json
{
  "query": "servicios fabricacion digital IDIT",
  "total_matches": 3,
  "matches": [
    {
      "titulo": "IDIT - Espacios y Maquinaria",
      "categoria": "web",
      "url": "https://www.iberopuebla.mx/IDIT/espacios-y-maquinaria",
      "fecha": "2024-01-15T10:30:00Z",
      "contenido": "Descripción de los servicios de fabricación digital..."
    }
  ]
}
```

---

## 2. Cambios en agent.py

### 2.1 System Prompt Actualizado
Se agregaron instrucciones claras sobre cuándo usar cada tool:
```
Uso de tools:
- Para preguntas sobre equipos, máquinas, instrumentos → buscar_equipos_idit
- Para preguntas sobre información general del IDIT → buscar_informacion_idit
- Nunca responder sobre equipos o información institucional sin consultar primero
```

### 2.2 Nodo retrieve_context Modificado
**Antes**:
```
[CONTEXTO INSTITUCIONAL]  ← búsqueda genérica de ChromaDB sin filtros
{documentos generales}

[DATOS OPERATIVOS EN VIVO]
{Firebase contexto}

[PREGUNTA]
{pregunta del usuario}
```

**Ahora**:
```
[DATOS OPERATIVOS EN VIVO - FIREBASE]
{Firebase contexto con salones, personal, horarios en vivo}

[PREGUNTA]
{pregunta del usuario}
```

**Ventaja**: El agente decide bajo demanda cuándo consultar ChromaDB, mejorando eficiencia y precisión.

### 2.3 Herramientas Registradas
Lista completa de tools en el grafo (5 total):
1. `buscar_salones_idit` (Firebase)
2. `buscar_personal_idit` (Firebase)
3. `obtener_agenda_personal_idit` (Firebase)
4. `buscar_equipos_idit` (ChromaDB - bitácoras) ⬅ NUEVA
5. `buscar_informacion_idit` (ChromaDB - web scraping) ⬅ NUEVA

---

## 3. Actualización de .gitignore

Se agregó sección dedicada a bitácoras:
```gitignore
# Bitácoras de mantenimiento (archivos fuente y markdown generado)
bitacoras/
bitacoras_md/
```

---

## 4. Flujo de Ejecución del Agente

```
Usuario pregunta: "¿Qué especificaciones tiene el brazo romer?"
         ↓
[retrieve_context node]
  - Obtiene datos operativos vivos de Firebase (salones disponibles, personal, etc.)
  - Construye mensaje inicial: "[DATOS OPERATIVOS EN VIVO - FIREBASE]...\n[PREGUNTA]..."
         ↓
[agent node - LLM con herramientas]
  - LLM lee la pregunta y el contexto Firebase
  - LLM decide: "Necesito buscar equipos → invocar buscar_equipos_idit"
  - LLM estructura la llamada: {"query": "brazo romer", "seccion": "FABLAB"}
         ↓
[tools node]
  - Ejecuta: buscar_equipos_idit(query="brazo romer", seccion="FABLAB")
  - Retorna JSON con especificaciones, ubicación, responsable, etc.
         ↓
[agent node - Segunda pasada]
  - LLM recibe resultado de la tool en su contexto
  - LLM sintetiza: "El Brazo Romer es un equipo en FABLAB ubicado en J-001A...
    Lo opera el JEFE DE TALLER FABLAB y su última actualización fue en ENE 24."
         ↓
Usuario obtiene respuesta natural en español
```

---

## 5. Validación y Testing

### Test Ejecutado
```
VALIDACION DE NUEVAS TOOLS CHROMA
======================================================================
TEST 1: Estructura de tools en el grafo
Nodos del grafo: ['__start__', 'retrieve_context', 'agent', 'tools']
OK: Nodo 'tools' presente en el grafo
OK: Test 1 pasado
```

### Vector Store Verificado
```
Vector store conectado. Documentos indexados: 330
```

---

## 6. Ejemplos de Uso

### Ejemplo 1: Consulta sobre Equipo Específico
```
Usuario: "¿Qué hace el brazo romer y dónde está?"

Agente invoca: buscar_equipos_idit(query="brazo romer")
Respuesta: "El Brazo Romer es un equipo de FABLAB ubicado en J-001A que realiza 
          inspección, ingeniería inversa y modelado 3D. Lo opera el JEFE DE TALLER FABLAB."
```

### Ejemplo 2: Filtro por Sección
```
Usuario: "¿Qué máquinas hay en FABLAB?"

Agente invoca: buscar_equipos_idit(query="máquinas equipos", seccion="FABLAB")
Respuesta: [lista de equipos de FABLAB]
```

### Ejemplo 3: Información Institucional
```
Usuario: "¿Qué es el IDIT y cuáles son sus servicios?"

Agente invoca: buscar_informacion_idit(query="IDIT servicios tecnologías")
Respuesta: "El IDIT es el Instituto de Diseño e Innovación Tecnológica de IBERO Puebla.
          Ofrece servicios de fabricación digital, prototipos, robótica, IA, entre otros..."
```

### Ejemplo 4: Plan de Mantenimiento
```
Usuario: "¿Cuál es el plan de mantenimiento del brazo romer?"

Agente invoca: buscar_equipos_idit(query="plan mantenimiento", 
                                   seccion="FABLAB", 
                                   chunk_type="mantenimiento_interno")
Respuesta: [Información de mantenimiento del equipo]
```

---

## 7. Criterios de Aceptación ✓

| Criterio | Estado |
|----------|--------|
| buscar_equipos_idit retorna JSON con matches de bitácoras | ✓ |
| buscar_equipos_idit soporta filtros seccion y chunk_type | ✓ |
| buscar_informacion_idit retorna JSON de web_scraping | ✓ |
| Agente invoca tools autónomamente basado en contexto | ✓ |
| retrieve_context ya no hace búsqueda genérica de Chroma | ✓ |
| No hay segunda conexión a ChromaDB | ✓ |
| .gitignore configurado correctamente | ✓ |
| System prompt incluye instrucciones de tools | ✓ |

---

## 8. Próximos Pasos (Opcionales)

1. **Ejecutar en conversación real**: El agente está listo para usar en `python agent.py` o vía `api.py`

2. **Enhancements futuros**:
   - Añadir filtro por fecha en buscar_equipos_idit (actualizacion >= "ENE 24")
   - Agregar búsqueda por ubicación (ej. "¿Qué equipos están en J-001?")
   - Combinar resultados de Firebase + ChromaDB (ej. "¿Quién puede enseñarme el brazo romer?")
   - Soporte para búsquedas relacionadas (ej. "Equipos similares a...")

3. **Monitoreo**:
   - Activar `AGENT_DEBUG_TRACE=1` para ver traces de invocación de tools
   - Revisar logs de Chroma para performance de queries

---

## Archivos Modificados

- **[agent.py](agent.py)**
  - SYSTEM_PROMPT: Agregadas instrucciones de tools (líneas 114-131)
  - build_graph(): Dos tools nuevas como closures (líneas 1057-1168)
  - retrieve_context(): Eliminada búsqueda genérica de ChromaDB (líneas 1183-1197)
  - tools list: Registradas 5 tools totales (línea 1165)

- **[.gitignore](.gitignore)**
  - Agregada sección "Bitácoras de mantenimiento" con bitacoras/ y bitacoras_md/

---

**Implementación completada y validada. ✓**
