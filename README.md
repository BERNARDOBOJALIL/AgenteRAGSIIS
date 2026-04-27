# Agente RAG SIIS IDIT

Proyecto RAG en Python para responder consultas del IDIT con datos unificados desde:

- Web scraping
- PDFs
- Firestore de salones IDIT (todo)
- Firestore de usuarios (todas las colecciones + vista objetivo academico)

Web/PDF/DOCX se transforman a Markdown para ChromaDB.
Firestore se consulta en vivo en runtime del agente (sin convertir a Markdown).

## Estructura

```text
AgenteRAGSIIS/
├── agent.py
├── api.py
├── ingest.py
├── scraper.py
├── firebase_sources.py
├── markdown_unifier.py
├── normalize_chroma_markdown.py
├── crud_chroma.py
├── requirements.txt
└── .env
```

## Instalacion

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Variables de entorno

### Requeridas para embeddings y chat

```env
GOOGLE_API_KEY=...
GROQ_API_KEY=...
```

### Chroma Cloud (si usas runtime cloud en agent.py/api.py)

```env
CHROMA_API_KEY=...
CHROMA_TENANT=...
CHROMA_DATABASE=...
CHROMA_CLOUD_HOST=api.trychroma.com
CHROMA_CLOUD_PORT=443
```

### Firestore (ya presentes en tu .env actual)

```env
VITE_FIREBASE_API_KEY=...                 # usuarios
VITE_FIREBASE_PROJECT_ID=...              # usuarios

VITE_FIREBASE_SALONES_API_KEY=...         # salones IDIT
VITE_FIREBASE_SALONES_PROJECT_ID=...      # salones IDIT
```

### Opcionales para ajustar extraccion Firestore

```env
FIREBASE_USUARIOS_COLLECTION=usuarios
FIREBASE_HORARIOS_COLLECTION=horarios
FIREBASE_CALENDARIOS_COLLECTION=calendarios
FIREBASE_CITAS_COLLECTION=citas
FIREBASE_NOTIFICACIONES_COLLECTION=notificaciones
FIREBASE_INCLUDE_SUBCOLLECTIONS=true
FIREBASE_MAX_TRAVERSAL_DEPTH=2
FIREBASE_SALONES_MAX_TRAVERSAL_DEPTH=none
FIREBASE_USERS_INCLUDE_SUBCOLLECTIONS=true
FIREBASE_USERS_MAX_TRAVERSAL_DEPTH=none

# Si REST no permite descubrir colecciones raiz, define allowlists:
FIREBASE_SALONES_COLLECTION_ALLOWLIST=salones
FIREBASE_USERS_COLLECTION_ALLOWLIST=usuarios,horarios,calendarios,citas,notificaciones

FIREBASE_PROFESSOR_FIELD_MARKERS=rol,role,tipo_usuario,teacher,is_professor
FIREBASE_PROFESSOR_VALUE_MARKERS=profesor,docente,teacher,faculty
FIREBASE_TARGET_ROLES=academico,académico
FIREBASE_TARGET_TYPES=profesor,administrativo,aminisrativo
FIREBASE_EXCLUDED_STUDENT_MARKERS=alumno,alumna,estudiante,student
FIREBASE_RUNTIME_CACHE_TTL_SECONDS=45
```

### Acceso total recomendado (Firebase Admin SDK)

Para acceder a todas las colecciones/subcolecciones de ambos proyectos sin limites de descubrimiento REST,
configura service account por proyecto. En Render usa variables de entorno con el JSON completo o en base64; no dependas de rutas locales.

```env
# Proyecto salones
FIREBASE_SALONES_SERVICE_ACCOUNT_JSON={...json...}
# O en base64:
# FIREBASE_SALONES_SERVICE_ACCOUNT_JSON_BASE64=...

# Proyecto usuarios
FIREBASE_USERS_SERVICE_ACCOUNT_JSON={...json...}
# O en base64:
# FIREBASE_USERS_SERVICE_ACCOUNT_JSON_BASE64=...

# Opcional en desarrollo local:
# FIREBASE_SALONES_SERVICE_ACCOUNT_FILE=C:/ruta/sa-salones.json
# FIREBASE_USERS_SERVICE_ACCOUNT_FILE=C:/ruta/sa-usuarios.json
# Defaults locales en este repo:
# - siis-d3571-336618a9b5d1.json (salones)
# - siis-9593c-611399213026.json (usuarios)
```

Tambien se acepta fallback global:

```env
GOOGLE_APPLICATION_CREDENTIALS=C:/ruta/service-account.json
FIREBASE_SERVICE_ACCOUNT_JSON={...json...}
FIREBASE_SERVICE_ACCOUNT_JSON_BASE64=...
```

Nota: la fuente de salones se recorre completa por defecto (todas las colecciones y documentos). Solo usa `FIREBASE_SALONES_MAX_TRAVERSAL_DEPTH` si quieres limitar profundidad.

Nota: en runtime, el agente consulta Firebase en vivo para:
- Salones: toda la informacion disponible.
- Usuarios: `rol=academico` y `tipo` profesor/administrativo (incluye variantes tipograficas) y excluye alumnos/estudiantes.
- Usuarios (full): limita registros a IDs de usuarios objetivo para colecciones `usuarios`, `horarios`, `calendarios`, `citas` y `notificaciones`.
- Join por `doc_id` con colecciones de horarios, calendarios, citas y notificaciones.
- Cruza personal academico con salones (por claves de salon y responsables) para responder en que salon operan.

## Formato Markdown unificado

Todos los documentos indexados quedan en el mismo esquema:

- Tipo de consulta
- Titulo
- Resumen
- Informacion principal
- Personal relacionado
- Equipamiento
- Ubicacion
- Servicios
- Datos estructurados
- Fuente

Cuando falta un dato, el sistema coloca exactamente:

```text
No poseemos esa infromacion.
```

## Tipos de consulta normalizados

El normalizador clasifica cada documento en una de estas categorias:

1. Informacion general de IDIT
2. Informacion sobre personal, equipamiento y ubicaciones dentro del IDIT
3. Informacion sobre servicios del IDIT

## Ingesta principal

### Ejecutar todo (web + PDF + DOCX)

```bash
python ingest.py
```

### Limpiar e indexar desde cero

```bash
python ingest.py --reset
```

### Solo web/PDF

```bash
python ingest.py --skip-firebase
```

### Incluir Firestore en ingesta estatica (opcional)

No es necesario para el agente en modo dinamico, pero queda disponible:

```bash
python ingest.py --include-firebase-static
```

### Definir carpeta de PDFs

```bash
python ingest.py --pdf-dir ./pdfs
```

### Convertir e ingerir DOCX

Por defecto se buscan DOCX en la raiz del proyecto de forma recursiva.

```bash
python ingest.py --docx-dir ./
```

Para hacer solo DOCX:

```bash
python ingest.py --skip-web --skip-pdf --skip-firebase --docx-dir ./
```

### Subir directo a Chroma Cloud

```bash
python ingest.py --cloud
```

### Exportar documentos Markdown generados

Por defecto se exportan en:

```text
./exports/markdown_ingesta
```

Puedes cambiarlo con:

```bash
python ingest.py --export-md-dir ./exports/md_custom
```

## Script para normalizar Chroma ya existente

Si ya tienes datos previos de scraping en Chroma y quieres reacomodarlos al formato unificado:

```bash
python normalize_chroma_markdown.py
```

Por defecto:

- Normaliza solo fuentes que parecen web scraping
- Exporta Markdown en `./exports/chroma_normalizado`
- Trabaja contra Chroma local (`./chroma_db`)

Opciones comunes:

```bash
# Simular sin escribir cambios
python normalize_chroma_markdown.py --dry-run

# Normalizar todas las fuentes
python normalize_chroma_markdown.py --all-sources

# Reprocesar incluso los ya normalizados
python normalize_chroma_markdown.py --force

# Usar Chroma Cloud
python normalize_chroma_markdown.py --cloud
```

## Cargar y convertir PDFs a Markdown

La conversion PDF -> Markdown se hace automaticamente dentro de `ingest.py` usando `scraper.py` + `markdown_unifier.py`.

Solo coloca tus PDFs en la carpeta indicada y ejecuta ingesta.

## Ejecutar chat en terminal

```bash
python agent.py
```

## Ejecutar API

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health`
- `POST /chat`
- `DELETE /chat/{session_id}`


