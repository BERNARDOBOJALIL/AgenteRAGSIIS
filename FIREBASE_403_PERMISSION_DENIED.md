# Error 403: Missing or Insufficient Permissions en Producción

## 🔴 El Problema

```
WARNING:firebase_sources:No se pudo leer la coleccion 'salones' (admin): 403 Missing or insufficient permissions.
```

Este error ocurre cuando:
1. El código intenta usar **FirestoreAdminClient** (basado en service account JSON)
2. Pero el service account **no tiene permisos suficientes** para leer la colección `salones`

## 🔧 Lo Que He Arreglado

Ahora el código implementa **fallback automático a REST API**:

```
Intento 1: Admin Client (service account) 
  ↓ [403 Permission Denied]
Intento 2: REST API (API Key) ✅
  ↓ [Exitoso - retorna datos]
```

## ✅ Cómo Funciona Ahora

En `fetch_salones_raw_records()`:

1. **Primera estrategia:** Intenta con `FirestoreAdminClient` (service account)
2. **Si falla o retorna vacío:** Automáticamente intenta con `FirestoreRESTClient` (API key)
3. **Si REST API funciona:** Retorna los datos de salones
4. **Si todo falla:** Retorna lista vacía (graceful degradation)

## 📋 Qué Necesitas Hacer en Producción

### Opción 1: Dejar el Fallback Automático (Recomendado)
No necesitas hacer nada. El código ahora es resiliente.

**Pero asegúrate de que tienes:**
```bash
✅ VITE_FIREBASE_SALONES_PROJECT_ID=siis-d3571-xxxxx
✅ VITE_FIREBASE_SALONES_API_KEY=AIzaSyD7gb3WmD5GaKy74Z4LxLmj1uoF7ohHQnU
```

Si tienes estas dos variables, REST API funcionará como fallback.

### Opción 2: Deshabilitar Service Account en Producción
Si quieres evitar completamente el intento de admin client en producción:

```bash
# En tu .env de producción:
# Deja en blanco o no incluyas FIREBASE_SALONES_SERVICE_ACCOUNT_FILE
# El código saltará directo a REST API

VITE_FIREBASE_SALONES_PROJECT_ID=siis-d3571-xxxxx
VITE_FIREBASE_SALONES_API_KEY=AIzaSyD7gb3WmD5GaKy74Z4LxLmj1uoF7ohHQnU
# NO incluyas: FIREBASE_SALONES_SERVICE_ACCOUNT_FILE=...
```

### Opción 3: Dar Permisos Correctos al Service Account (Más Complejo)
Si quieres que admin client funcione sin fallback:

1. Ir a **Google Cloud Console** → Tu proyecto Salones
2. **IAM & Admin** → **Service Accounts**
3. Encontrar la service account usada para `siis-d3571-336618a9b5d1.json`
4. Asignar rol: **`roles/datastore.user`** o **`roles/owner`**
5. Guardar cambios (puede tardar 1-2 minutos en propagarse)

## 🔍 Cómo Verificar Que Funciona

### Paso 1: Habilitar Debug Logs
```bash
# En tu .env o Docker env:
AGENT_DEBUG_TRACE=true
```

### Paso 2: Llamar tu API
```bash
curl -X POST https://tu-api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Dime lo que sepas del J003",
    "session_id": "test"
  }'
```

### Paso 3: Revisar los Logs
Busca mensajes como:

```
# Si intenta admin client primero:
Cliente Firestore Admin inicializado exitosamente
# Luego si falla:
Error iterando documentos con cliente actual (posible fallo de permisos): 403 Missing or insufficient permissions

# Luego intenta REST:
Primera estrategia (admin client) retornó 0 salones. Intentando con REST API...
Intentando cargar salones con REST API (fallback)
REST API exitosa: se obtuvieron 42 salones
```

### Paso 4: Verifica la Respuesta
Debería ahora retornar la información completa de J-003:
```json
{
  "response": "El J-003 es el Laboratorio Multidisciplinar de Inteligencia Artificial y Realidad Aumentada..."
}
```

## 🎯 Resumen de Cambios en el Código

### firebase_sources.py

#### 1. `_build_client_from_env()` mejorada:
- Mejor manejo de excepciones cuando admin client falla
- Ahora hace fallback a REST API si admin client no se puede inicializar
- Logs más informativos sobre qué estrategia se usa

#### 2. `fetch_salones_raw_records()` mejorada:
- Intenta primero con admin client (si está disponible)
- Si obtiene datos, los retorna
- Si falla o retorna vacío, automáticamente intenta con REST API
- Gracias al REST API, ya no perderás datos incluso si el service account no tiene permisos

## 📊 Comparativa: Antes vs Después

### Antes (con error 403):
```
Admin Client → 403 Permission Denied → ❌ Retorna []
                                         "No tengo información"
```

### Después (con fallback automático):
```
Admin Client → 403 Permission Denied → REST API → ✅ Retorna 42 salones
                                        (automatic fallback)
                                        "El J-003 es..."
```

## 🐛 Si Sigue Fallando

1. **Verifica que REST API vars existen:**
   ```bash
   curl https://tu-api/firebase-diagnosis | jq '.salones_config'
   # Debe mostrar has_api_key: true
   ```

2. **Verifica que Firestore REST API está habilitada:**
   - Firebase Console → Settings → APIs
   - Busca "Cloud Datastore API" → Habilitada

3. **Revisa permisos del API Key:**
   - Firebase Console → Settings → API Keys
   - Asegura que tiene "Cloud Datastore API" habilitada

4. **Prueba REST API directamente:**
   ```bash
   curl -X POST "https://firestore.googleapis.com/v1/projects/siis-d3571/databases/(default)/documents/salones" \
     -H "Content-Type: application/json" \
     -d '{}' \
     -G --data-urlencode "key=AIzaSyD7gb3WmD5GaKy74Z4LxLmj1uoF7ohHQnU"
   ```

## 💡 Notas Técnicas

- **Admin Client:** Requiere service account JSON, tiene permisos completos, pero puede ser rechazado si roles están mal configurados
- **REST API:** Usa API Key, más simple, pero limitado por quotas
- **Fallback:** El código ahora es resiliente a ambas estrategias

La solución implementada asegura que **incluso si el service account no tiene permisos**, aún puedas servir datos via REST API.

## 🚀 Recomendación Final

Para producción, recomiendo:
1. ✅ Usar **solo REST API** (API Key)
2. ❌ NO usar service account JSON en .env de producción
3. ✅ Confiar en el fallback automático como backup

Esto simplifica la configuración y reduce riesgos de seguridad.
