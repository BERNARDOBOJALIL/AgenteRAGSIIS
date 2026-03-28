# 🎓 Agente RAG Universitario

Asistente inteligente para tu universidad construido con **LangChain + Chroma Cloud + Groq + Google Gemini**.

---

## Estructura del proyecto

```
agente_universidad/
├── ingest.py          # Alimenta la base de datos (ejecutar primero)
├── agent.py           # Chat con el agente en terminal
├── api.py             # API HTTP para frontend (FastAPI)
├── scraper.py         # Lógica de web scraping y carga de documentos
├── crud_chroma.py     # Operaciones CRUD sobre ChromaDB
├── pdfs/              # (Opcional) PDFs para ingesta local
├── chroma_db/         # (Opcional) DB local temporal para procesos de ingesta
├── requirements.txt
└── .env
```

---

## Instalación

```bash
# 1. Crear entorno virtual
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Copiar y configurar variables de entorno
cp .env.example .env
# Edita .env con tus API keys
```

---

## Configuración de API Keys

Edita el archivo `.env`:

```env
GOOGLE_API_KEY=tu_key_de_google_ai_studio
GROQ_API_KEY=tu_key_de_groq
CHROMA_API_KEY=tu_key_de_chroma_cloud
CHROMA_TENANT=tu_tenant_de_chroma
CHROMA_DATABASE=RAG-BERNY
CHROMA_CLOUD_HOST=api.trychroma.com    # opcional
CHROMA_CLOUD_PORT=443                  # opcional
LANGCHAIN_API_KEY=tu_key_de_langsmith   # opcional
LANGSMITH_API_KEY=tu_key_de_langsmith   # opcional
```

Obtén tu `GOOGLE_API_KEY` gratis en: https://aistudio.google.com/app/apikey

---

## Cómo alimentar la base de datos

### 1. Desde PDFs

Coloca tus archivos PDF en la carpeta `pdfs/`:
```
pdfs/
├── reglamento_escolar.pdf
├── oferta_academica.pdf
└── servicios_universitarios.pdf
```

### 2. Desde web scraping

Edita `ingest.py` y agrega tus URLs:

```python
URLS_ESTATICAS = [
    "https://www.miuniversidad.edu.mx/becas",
    "https://www.miuniversidad.edu.mx/biblioteca",
]
```

### 3. Datos manuales

Agrega información directa en `ingest.py`:

```python
DATOS_MANUALES = [
    {
        "contenido": "La biblioteca abre de 8:00 a 21:00 de lunes a viernes.",
        "categoria": "servicios",
        "fuente": "manual",
    },
]
```

### 4. Ejecutar la ingestión

```bash
# Primera vez o para agregar documentos nuevos
python ingest.py

# Para borrar todo y empezar desde cero
python ingest.py --reset
```

---

## Iniciar el chat en terminal

```bash
python agent.py
```

```
═══════════════════════════════════════════════════════
  🎓 Asistente Universitario
  Escribe 'salir' o 'exit' para terminar
═══════════════════════════════════════════════════════

Tú: ¿Cuáles son los requisitos para obtener una beca?
Asistente: Para obtener una beca debes...
```

## Iniciar API para frontend (FastAPI)

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Endpoints principales:
- `GET /health` → estado y cantidad de documentos en colección.
- `POST /chat` → envía prompt y recibe respuesta.
- `DELETE /chat/{session_id}` → limpia historial de esa sesión.

Ejemplo request a `POST /chat`:

```json
{
  "prompt": "¿Qué es el IDIT?",
  "session_id": "usuario-web-1"
}
```

Ejemplo response:

```json
{
  "response": "El IDIT es...",
  "session_id": "usuario-web-1"
}
```

---

## Categorías de metadatos recomendadas

| `categoria`      | Tipo de contenido                          |
|------------------|--------------------------------------------|
| `reglamento`     | Artículos del reglamento escolar           |
| `servicios`      | Biblioteca, cafetería, fotocopiadora       |
| `laboratorios`   | Horarios y normas de laboratorios          |
| `becas`          | Requisitos y convocatorias                 |
| `inscripciones`  | Procedimientos y fechas                    |
| `profesores`     | Horarios de asesorías y contactos          |
| `deportes`       | Instalaciones y horarios                   |
| `tramites`       | Constancias, historial académico           |

---

## Sitios con JavaScript

Si el sitio de tu universidad usa React/Angular, instala Playwright:

```bash
pip install playwright
playwright install chromium
```

Y agrega las URLs en `URLS_DINAMICAS` dentro de `ingest.py`.

---

## Flujo completo

```
PDFs / Web / Datos manuales
         ↓
     scraper.py
    (limpieza + fragmentación)
         ↓
   GoogleGenerativeAIEmbeddings
    (texto → vectores)
         ↓
      ChromaDB
   (chroma_db/)
         ↓
    agent.py → chat
```

---

## Chroma Cloud

El runtime actual de `agent.py` y `crud_chroma.py` consulta directamente la colección en Chroma Cloud.

Para copiar datos locales a Cloud, usa la CLI oficial de Chroma:

```bash
chroma login
chroma copy --all --from-local --path ./chroma_db --to-cloud --db RAG-BERNY
```

