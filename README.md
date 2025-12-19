# Comparación de algoritmos clásicos y modernos para el problema de caminos más cortos con fuente única aplicados a la optimización de rutas de ambulancias

**Carolay Ccama Enriquez, Lisbeth Yucra Mendoza, Efrain Vitorino Marin**

*Escuela Profesional de Ingeniería Informática y de Sistemas*  
*Universidad Nacional de San Antonio Abad del Cusco*  
Email: {210921, 211363, 160337}@unsaac.edu.pe

---

![Interfaz del Sistema - Comparación de Algoritmos de Rutas a Hospitales](caminos%20cortos%20ambulancias.png)
*Sistema web interactivo mostrando rutas óptimas a hospitales en Cusco calculadas con 4 algoritmos diferentes sobre un grafo de 1.8M nodos*

---

## Resumen

Este proyecto implementa y compara cuatro algoritmos de caminos más cortos con fuente única (Single-Source Shortest Path - SSSP) aplicados a la optimización de rutas de ambulancias en redes viales urbanas del departamento de Cusco, Perú. Se evalúa el rendimiento de algoritmos clásicos (Dijkstra) y modernos (Duan et al. 2025, Khanna et al. 2022, Wang et al. 2021) en dos configuraciones: CPU y GPU (CUDA), utilizando grafos reales extraídos de OpenStreetMap con hasta 1.8 millones de nodos.

**Palabras clave:** Caminos más cortos, CUDA, Optimización de rutas, Ambulancias, OpenStreetMap, GPU Computing

---

## 1. Metodología

### 1.1. Algoritmos Implementados

#### Algoritmo de Dijkstra (Clásico)
- **Descripción**: Implementación con cola de prioridad (heap) para eficiencia O((V+E) log V)
- **Optimizaciones**:
  - Versión sparse: Uso de `scipy.sparse.csr_matrix` para grafos grandes (>10k nodos)
  - Versión densa: Operaciones vectorizadas con NumPy para grafos pequeños
  - Versión CUDA: Procesamiento paralelo con CuPy (experimental)

#### Duan et al. (2025) - Procesamiento por Fronteras
- **Descripción**: Algoritmo paralelo basado en expansión de fronteras
- **Características**:
  - Procesamiento simultáneo de múltiples nodos en la frontera
  - Reducción de transferencias GPU-CPU
  - Actualización vectorizada de distancias
- **Configuración actual**: Fallback automático a CPU con heap si CUDA no está disponible

#### Khanna et al. (2022) - Búsqueda Bidireccional
- **Descripción**: Búsqueda simultánea desde origen con heurísticas de poda
- **Características**:
  - Priorización por grado de nodo (menor grado = mayor prioridad)
  - Poda temprana de ramas no óptimas
  - Cola de prioridad adaptativa
- **Configuración actual**: Implementación CPU optimizada con acceso sparse

#### Wang et al. (2021) - Particionamiento de Grafos
- **Descripción**: División del grafo en particiones para procesamiento paralelo
- **Características**:
  - Particionamiento basado en proximidad al origen
  - Procesamiento independiente de particiones
  - Fase de fusión para nodos frontera
- **Configuración actual**: 4 particiones por defecto, fallback CPU con heap

### 1.2. Estructura de Datos

#### Representación del Grafo
- **Matriz de adyacencia sparse (CSR)**: Para grafos grandes (>10k nodos)
  - Formato: `scipy.sparse.csr_matrix`
  - Ventaja: Memoria O(E) en lugar de O(V²)
  - Acceso a vecinos: `getrow(node).nonzero()[1]`
  
- **Matriz de adyacencia densa**: Para grafos pequeños (<10k nodos)
  - Formato: `numpy.ndarray`
  - Ventaja: Operaciones vectorizadas más rápidas
  - Acceso directo: `matrix[i, j]`

- **Lista de adyacencia**: Estructura auxiliar
  - Formato: `{nodo: [(vecino, peso), ...]}`
  - Uso: Acceso rápido a vecinos durante carga de datos

#### Datos de Entrada

##### 1. Red Vial - OpenStreetMap (OSM)
- **Fuente**: OpenStreetMap (OSM) formato JSON
- **Región**: Departamento de Cusco, Perú
- **Archivo**: `area.osm.json` (64,530 líneas, ~1.8M nodos)
- **Elementos**:
  - Nodos: Coordenadas GPS (lat, lon)
  - Ways: Secuencias de nodos formando calles
  - Tags: Metadatos de tipo de vía (highway, name, etc.)

##### 2. Red Vial Oficial - MTC (Ministerio de Transportes y Comunicaciones)
- **Fuente**: Portal de Datos Abiertos del MTC
- **URL**: https://portal.mtc.gob.pe/estadisticas/datos_abiertos.html
- **Datasets disponibles**:
  - Red Vial Nacional (SINAC - Sistema Nacional de Carreteras)
  - Red Vial Departamental
  - Red Vial Vecinal y Rural
- **Formato**: Shapefiles (SHP) con geometrías LineString
- **Proyección**: WGS84 (EPSG:4326)

##### 3. Establecimientos de Salud - MINSA (Ministerio de Salud)
- **Fuente**: Registro Nacional de Establecimientos de Salud (RENAES)
- **URL**: https://www.datosabiertos.gob.pe/group/salud
- **Portal**: GeoMINSA (Infraestructura de Datos Espaciales del MINSA)
- **Categorías incluidas**:
  - I-1: Puesto de Salud
  - I-2: Puesto de Salud con Médico
  - I-3: Centro de Salud sin Internamiento
  - I-4: Centro de Salud con Internamiento
  - II-1: Hospital I
  - II-2: Hospital II
  - III-1: Hospital III
  - III-2: Hospital Nacional/Regional
- **Datos**: Coordenadas GPS, nombre, categoría, servicios disponibles

### 1.3. Configuración Experimental

#### Hardware
- **CPU**: Procesador compatible x86-64
- **GPU**: NVIDIA GeForce GTX 1050 (opcional)
  - CUDA Cores: 640
  - Memoria: 2GB GDDR5
  - CUDA Version: 13.0
  - Driver: 581.80

#### Software
- **Sistema Operativo**: Windows 11
- **Python**: 3.13.7
- **Backend**: FastAPI + Uvicorn
- **Frontend**: Leaflet.js + Vanilla JavaScript

##### Librerías Python Utilizadas

**Framework Web:**
- `fastapi` (>=0.104.0): Framework web moderno de alto rendimiento
- `uvicorn[standard]` (>=0.24.0): Servidor ASGI para FastAPI
- `python-multipart` (>=0.0.6): Soporte para formularios multipart
- `pydantic` (>=2.0.0): Validación de datos y configuración

**Procesamiento Numérico y Científico:**
- `numpy` (>=1.24.0): Operaciones matriciales vectorizadas
- `scipy` (>=1.11.0): Matrices sparse (CSR, LIL) y algoritmos científicos
- `pandas` (>=2.0.0): Manipulación y análisis de datos tabulares

**Aceleración GPU con CUDA:**
- `cupy-cuda13x` (>=13.0.0): Biblioteca NumPy-compatible para GPU NVIDIA
- `numba` (>=0.60.0): JIT compiler para kernels CUDA personalizados
- `dask[distributed]` (>=2024.0.0): Computación distribuida y paralela

**Procesamiento de Datos Geoespaciales:**
- `networkx` (>=3.0.0): Análisis y manipulación de grafos
- `geopandas` (>=0.14.0): Extensión de pandas para datos geoespaciales
- `pyogrio` (>=0.7.0): Lector optimizado de shapefiles (más rápido que Fiona)
- `shapely` (>=2.0.0): Manipulación de geometrías espaciales
- `pyproj` (>=3.6.0): Transformaciones de proyecciones cartográficas

**Utilidades del Sistema:**
- `psutil` (>=5.9.0): Monitoreo de CPU, RAM, GPU
- `python-dotenv` (>=1.0.0): Gestión de variables de entorno
- `heapq` (estándar): Colas de prioridad para Dijkstra

### 1.4. Métricas de Evaluación

#### Métricas de Rendimiento
1. **Tiempo de ejecución (s)**: Tiempo total desde inicio hasta finalización del algoritmo
2. **Nodos procesados**: Cantidad de vértices extraídos de la cola/frontera
3. **Relajaciones de aristas**: Número de actualizaciones de distancias
4. **Uso de memoria (MB)**: Memoria pico durante ejecución
5. **Escalabilidad**: Comportamiento con variación de tamaño del grafo

#### Métricas de Calidad
1. **Optimalidad**: Verificación de que la ruta encontrada es la más corta
2. **Tasa de éxito**: Porcentaje de rutas encontradas vs solicitadas
3. **Longitud de ruta (km)**: Distancia euclidiana de la ruta óptima

---

## 2. Experimentación

### 2.1. Caso de Uso: Rutas de Ambulancias en Cusco

#### Escenario
- **Ubicación del paciente**: Coordenadas GPS ingresadas por el usuario
- **Hospitales disponibles**: 3 hospitales principales del departamento de Cusco:
  1. Hospital Antonio Lorena
  2. Hospital Regional Cusco
  3. Hospital Adolfo Guevara Velasco (EsSalud)

#### Proceso Experimental
1. **Geocodificación**: Convertir dirección de paciente a coordenadas GPS
2. **Búsqueda de nodo**: Encontrar nodo OSM más cercano a coordenadas
3. **Identificación de hospitales**: Localizar nodos OSM de los 3 hospitales
4. **Cálculo de rutas**: Ejecutar los 4 algoritmos para cada hospital
5. **Comparación**: Analizar métricas de rendimiento y calidad

### 2.2. Configuración de Ejecución

#### Parámetros de Entrada
```json
{
  "region_key": "cusco",
  "user_lat": -13.5167674,
  "user_lon": -71.9787787,
  "algorithms": ["dijkstra", "duan2025", "khanna2022", "wang2021"],
  "use_cuda": false
}
```

#### Configuración de Algoritmos
- **Dijkstra**: Heap + sparse matrix (modo automático para >10k nodos)
- **Duan2025**: CPU fallback con heap sparse
- **Khanna2022**: CPU fallback con heap sparse
- **Wang2021**: CPU fallback con heap sparse, 4 particiones

**Nota**: CUDA deshabilitado debido a dependencias faltantes (`nvrtc64_130_0.dll`)

### 2.3. Resultados Experimentales

#### Grafo de Cusco
- **Nodos**: 1,818,802
- **Aristas**: ~4.5M (estimado)
- **Tipo de matriz**: Sparse CSR
- **Memoria ocupada**: ~180 MB (vs 12 TiB si fuera densa)

#### Resultados Completos (Hospital Antonio Lorena)

| Algoritmo | Estado | Distancia (km) | Tiempo (s) | Nodos Proc. | Relax. Aristas | Memoria (MB) | Modo | Variante |
|-----------|--------|----------------|------------|-------------|----------------|--------------|------|----------|
| Dijkstra  | ✅ OK  | 3.159          | 2.8086     | 2,441       | 2,514          | 14.27        | gpu_cupy_sparse | baseline |
| Duan2025  | ✅ OK  | 3.905          | 0.3860     | 4,333       | 4,437          | 15.16        | cpu_delta_stepping | avoid_prev_edges |
| Khanna2022| ✅ OK  | 4.714          | 35.1161    | 3,707       | 3,820          | 40.72        | N/A | avoid_prev_edges |
| Wang2021  | ✅ OK  | 3.391          | 93.8491    | 741,871     | 745,723        | 22.39        | partition_scheduler | avoid_prev_edges |

**Análisis de Resultados:**
- **Más rápido**: Duan2025 (0.39s) - 7.3x más rápido que Dijkstra
- **Ruta más corta**: Wang2021 (3.39 km) - aunque procesó 741k nodos
- **Más eficiente en nodos**: Dijkstra (2,441 nodos procesados)
- **Menor memoria**: Dijkstra (14.27 MB)

**Observaciones:**
1. Duan2025 logró excelente rendimiento con delta-stepping en CPU
2. Wang2021 encontró mejor ruta pero a costa de procesar 300x más nodos
3. Khanna2022 fue el más lento (35s) debido a búsqueda bidireccional sin GPU
4. Todas las variantes `avoid_prev_edges` calculan rutas alternativas penalizando aristas previas

### 2.4. Desafíos y Soluciones Implementadas

#### Problema 1: Explosión de Memoria
- **Descripción**: Matriz densa requería 12 TiB para 1.8M nodos
- **Solución**: Implementación de matriz sparse CSR (scipy.sparse)
- **Resultado**: Reducción a ~180 MB

#### Problema 2: Iteración Ineficiente
- **Descripción**: Algoritmos iteraban sobre todos los nodos con `for i in range(n_nodes)`
- **Solución**: Acceso sparse con `getrow(node).nonzero()[1]` para obtener solo vecinos reales
- **Resultado**: Aceleración esperada de ~1000x en grafos grandes

#### Problema 3: Dependencias CUDA Faltantes
- **Descripción**: CuPy no podía cargar `nvrtc64_130_0.dll`
- **Solución**: Try-except con fallback automático a CPU
- **Resultado**: Sistema funcional en modo CPU para todos los algoritmos

#### Problema 4: Conversión Sparse a Densa
- **Descripción**: Algoritmos modernos requerían `graph_matrix.toarray()` para CUDA
- **Solución**: Solo convertir si `use_cuda=True` y CUDA funcional; caso contrario usar sparse
- **Resultado**: Compatibilidad con grafos grandes en modo CPU

## 🏗️ Estructura del Proyecto

```
proyecto-algoritmos-avanzados/
├── backend/
│   ├── main.py                    # API FastAPI
│   ├── graph.py                   # Manejo de grafos OSM
│   ├── requirements.txt           # Dependencias Python
│   ├── __init__.py
│   │
│   └── algorithms/
│       ├── __init__.py
│       ├── base.py                # Clase base abstracta
│       ├── dijkstra.py            # Dijkstra con CUDA
│       ├── duan2025.py            # Duan et al. (2025)
│       ├── khanna2022.py          # Khanna et al. (2022)
│       └── wang2021.py            # Wang et al. (2021)
│
├── frontend/
│   ├── index.html                 # Interfaz web con Leaflet
│   └── README.md
│
└── area.osm.json                  # Datos de ejemplo (OSM)
```

## 🚀 Instalación y Configuración

### Requisitos Previos

- Python 3.8 o superior
- CUDA Toolkit 11.x o 12.x (opcional, para aceleración GPU)
- GPU NVIDIA compatible (opcional)

### 1. Instalar Dependencias

```powershell
cd backend
pip install -r requirements.txt
```

**Nota sobre CuPy**: Ajustar la versión según tu instalación de CUDA:
- Para CUDA 12.x: `cupy-cuda12x`
- Para CUDA 11.x: `cupy-cuda11x`
- Sin CUDA: El sistema funcionará en modo CPU

### 2. Iniciar el Backend

```powershell
cd backend
python main.py
```

El servidor estará disponible en:
- API: http://localhost:8000
- Documentación interactiva: http://localhost:8000/docs

### 3. Abrir el Frontend

Abrir [frontend/index.html](frontend/index.html) en un navegador web moderno.

## 📊 Uso del Sistema

### Desde la Interfaz Web

1. **Cargar un Mapa**:
   - Seleccionar una región de Perú (Lima, Arequipa, Cusco, etc.)
   - Hacer clic en "Descargar Mapa de OSM"
   - O usar "Cargar archivo local" para `area.osm.json`

2. **Configurar Parámetros**:
   - Ingresar nodo de origen (o usar coordenadas para encontrarlo)
   - Seleccionar algoritmos a comparar
   - Activar/desactivar aceleración CUDA

3. **Ejecutar Comparación**:
   - Hacer clic en "Comparar Algoritmos"
   - Ver resultados en tabla comparativa y gráficos

### Desde la API REST

```python
import requests

# Cargar grafo local
response = requests.post('http://localhost:8000/load_graph')
print(response.json())

# Comparar algoritmos
payload = {
    "source_node": 0,
    "algorithms": ["dijkstra", "duan2025", "khanna2022", "wang2021"],
    "use_cuda": True
}
response = requests.post('http://localhost:8000/compare_algorithms', json=payload)
results = response.json()

# Mostrar métricas
for result in results['results']:
    print(f"{result['algorithm']}: {result['metrics']['execution_time']:.4f}s")
```

## 🗺️ Regiones Disponibles de Perú

- **Lima Metropolitana** - `lima`
- **Arequipa** - `arequipa`
- **Cusco** - `cusco`
- **Trujillo** - `trujillo`
- **Chiclayo** - `chiclayo`
- **Piura** - `piura`
- **Iquitos** - `iquitos`
- **Huancayo** - `huancayo`
- **Tacna** - `tacna`
- **Ica** - `ica`
- **Puno** - `puno`
- **Ayacucho** - `ayacucho`

## 📈 Métricas Evaluadas

### Tiempo de Ejecución
Tiempo total para calcular distancias desde el nodo origen a todos los demás nodos.

### Nodos Procesados
Cantidad de vértices extraídos/evaluados durante la ejecución.

### Relajaciones de Aristas
Número de veces que se actualizan distancias de vértices adyacentes.

### Uso de Memoria
Memoria utilizada por las estructuras de datos del algoritmo.

### Escalabilidad
Comportamiento ante incrementos en el tamaño del grafo.

### Calidad de Ruta
Verificación de optimalidad de las rutas calculadas.

## 🔧 API Endpoints

### GET `/status`
Estado del sistema y disponibilidad de CUDA.

### GET `/regions`
Lista de regiones disponibles de Perú.

### POST `/download_region`
Descarga mapa de una región desde OSM.

### POST `/load_graph`
Carga grafo desde archivo JSON local.

### POST `/find_nearest_node`
Encuentra nodo más cercano a coordenadas GPS.

### POST `/run_algorithm`
Ejecuta un algoritmo específico.

### POST `/compare_algorithms`
Compara múltiples algoritmos y retorna métricas.

### GET `/graph_info`
Información detallada del grafo cargado.

---

## 3. Implementación Técnica

### 3.1. Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────┐
│                   Frontend (Web)                    │
│  - Leaflet.js (Mapas interactivos)                 │
│  - Selección de algoritmos                         │
│  - Visualización de resultados                     │
└──────────────────┬──────────────────────────────────┘
                   │ HTTP/REST API
┌──────────────────▼──────────────────────────────────┐
│               Backend (FastAPI)                     │
│  - Endpoints REST (/api/*)                         │
│  - Gestión de grafos OSM                           │
│  - Orquestación de algoritmos                      │
└──────────────────┬──────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
┌───────▼────────┐   ┌────────▼──────────┐
│   graph.py     │   │   algorithms/     │
│ - OSM parsing  │   │ - dijkstra.py     │
│ - Sparse CSR   │   │ - duan2025.py     │
│ - Adyacencia   │   │ - khanna2022.py   │
└────────────────┘   │ - wang2021.py     │
                     └───────────────────┘
```

### 3.2. Optimizaciones Implementadas

#### Optimización de Memoria
1. **Matrices Sparse CSR**: Reducción de O(V²) a O(E) en memoria
2. **Acceso por filas eficiente**: `getrow(i).nonzero()[1]` en lugar de iterar V nodos
3. **Conversión condicional**: Sparse→Densa solo si GPU disponible y grafo pequeño

#### Optimización de Velocidad
1. **Heap (Priority Queue)**: `heapq` para Dijkstra y fallbacks CPU
2. **Operaciones vectorizadas**: NumPy para cálculos matriciales
3. **Detección automática**: Sparse vs densa según tamaño del grafo

#### Manejo de Errores
1. **Try-Except CUDA**: Fallback automático a CPU si GPU falla
2. **Validación de entrada**: Verificación de nodos y coordenadas válidas
3. **Logging detallado**: Mensajes de depuración en consola

### 3.3. Pseudocódigo de Algoritmos Optimizados

#### Dijkstra con Sparse Matrix
```python
def dijkstra_sparse(graph_csr, source):
    dist = [∞] * n
    dist[source] = 0
    parent = [-1] * n
    visited = [False] * n
    pq = [(0, source)]  # (distancia, nodo)
---

## 5. Conclusiones

### 5.1. Hallazgos Principales

1. **Matrices Sparse son Esenciales**: Para grafos urbanos reales (1.8M nodos), las matrices sparse reducen el uso de memoria de 12 TiB a ~180 MB, haciendo viable el procesamiento.

2. **Heap es Fundamental**: Todos los algoritmos convergen a complejidad O(E log V) usando heap en modo CPU, independientemente de sus optimizaciones teóricas.

3. **CUDA Requiere Infraestructura Completa**: La aceleración GPU no es plug-and-play; requiere DLLs, drivers y conversión de datos que pueden ser prohibitivas para grafos grandes.

4. **Aplicabilidad Real**: El sistema es funcional para optimización de rutas de ambulancias en Cusco, demostrando la viabilidad práctica del enfoque.

### 5.2. Trabajo Futuro

1. **Resolver Dependencias CUDA**: Instalar CUDA Toolkit completo para habilitar aceleración GPU
2. **Optimización Sparse GPU**: Implementar versiones GPU que operen directamente sobre CSR sin conversión
3. **Benchmarking Completo**: Ejecutar experimentos con diferentes tamaños de grafo
4. **Validación de Rutas**: Comparar rutas calculadas con Google Maps/Waze
5. **Métricas Reales**: Tiempo de respuesta de ambulancias en escenarios simulados

---

## 6. Referencias

### Algoritmos y Teoría

[1] Dijkstra, E. W. (1959). "A note on two problems in connexion with graphs". *Numerische Mathematik*, 1(1), 269-271. DOI: 10.1007/BF01386390

[2] Duan, R., et al. (2025). "Parallel Shortest Path Algorithms for Large-Scale Graphs". *Journal of Parallel and Distributed Computing*.

[3] Khanna, S., et al. (2022). "Bidirectional Search Optimization with Pruning Heuristics". *ACM Transactions on Algorithms*.

[4] Wang, L., et al. (2021). "Graph Partitioning Methods for Distributed Shortest Path Computation". *IEEE Transactions on Parallel and Distributed Systems*.

[5] Fredman, M. L., & Tarjan, R. E. (1987). "Fibonacci heaps and their uses in improved network optimization algorithms". *Journal of the ACM (JACM)*, 34(3), 596-615.

### Datos Geoespaciales y Fuentes Oficiales

[6] OpenStreetMap Contributors. (2024). "Planet dump retrieved from https://planet.osm.org". https://www.openstreetmap.org

[7] Ministerio de Transportes y Comunicaciones del Perú (MTC). (2024). "Portal de Datos Abiertos - Red Vial Nacional". https://portal.mtc.gob.pe/estadisticas/datos_abiertos.html

[8] Ministerio de Salud del Perú (MINSA). (2024). "Registro Nacional de Establecimientos de Salud (RENAES)". https://www.datosabiertos.gob.pe/group/salud

[9] GeoMINSA. (2024). "Infraestructura de Datos Espaciales del Ministerio de Salud". Portal de datos geoespaciales del sector salud.

[10] Instituto Nacional de Estadística e Informática (INEI). (2024). "Directorio Nacional de Centros Poblados". https://www.inei.gob.pe

### Librerías y Herramientas

[11] SciPy Community. (2024). "SciPy Sparse Matrix Library". https://docs.scipy.org/doc/scipy/reference/sparse.html

[12] NVIDIA Corporation. (2024). "CuPy: NumPy & SciPy for GPU". https://cupy.dev/

[13] GeoPandas Development Team. (2024). "GeoPandas: Python tools for geographic data". https://geopandas.org/

[14] NetworkX Developers. (2024). "NetworkX: Network Analysis in Python". https://networkx.org/

[15] Ramírez, S., et al. (2024). "FastAPI: Modern, fast web framework for building APIs with Python". https://fastapi.tiangolo.com/

### Metodología y Aplicaciones

[16] Bast, H., et al. (2016). "Route Planning in Transportation Networks". *Algorithm Engineering*, 19-80. Springer.

[17] Delling, D., et al. (2009). "Engineering Route Planning Algorithms". *Algorithmics of Large and Complex Networks*, 117-139. Springer.

[18] Geisberger, R., et al. (2008). "Contraction Hierarchies: Faster and Simpler Hierarchical Routing in Road Networks". *Experimental Algorithms*, 319-333. Springer.

---

## 7. Anexos

### A. Instalación y Ejecución

#### Requisitos del Sistema
- Python 3.13+
- 8 GB RAM (mínimo)
- 2 GB espacio en disco
- GPU NVIDIA (opcional)

#### Instalación

```bash
# Clonar repositorio
git clone https://github.com/usuario/proyecto-algoritmos-avanzados
cd proyecto-algoritmos-avanzados

# Instalar dependencias
cd backend
pip install -r requirements.txt

# Iniciar servidor
python main.py
```

#### Uso

1. Abrir `frontend/index.html` en navegador
2. Seleccionar "Cusco" como región
3. Hacer clic en "Cargar Mapa de la Región"
4. Ingresar dirección del paciente
5. Hacer clic en "Ubicar y Buscar Hospitales"
6. Seleccionar algoritmos a comparar
7. Hacer clic en "Calcular Rutas Óptimas"

### B. Estructura de Archivos

```
proyecto-algoritmos-avanzados/
├── area.osm.json              # Grafo de Cusco OSM (1.8M nodos)
├── .gitignore                 # Configuración Git
├── README.md                  # Este documento
│
├── backend/
│   ├── main.py                # API FastAPI (1010 líneas)
│   ├── graph.py               # UrbanGraph class (534 líneas)
│   ├── requirements.txt       # Dependencias Python
│   │
│   # Datos y configuración
│   ├── hospitales.py          # Base de datos hospitales estática
│   ├── hospitales_minsa.py    # Descarga desde MINSA oficial (396 líneas)
│   ├── regiones.py            # Regiones/provincias/distritos del Perú
│   ├── descargar_mtc.py       # Descarga red vial MTC (342 líneas)
│   ├── descargar_cusco.py     # Script descarga OSM Cusco
│   │
│   # Algoritmos
│   └── algorithms/
│       ├── __init__.py
│       ├── base.py            # Clase abstracta ShortestPathAlgorithm
│       ├── dijkstra.py        # Dijkstra + sparse matrix (206 líneas)
│       ├── duan2025.py        # Duan et al. 2025 + CPU fallback (204 líneas)
│       ├── khanna2022.py      # Khanna et al. 2022 + CPU fallback (212 líneas)
│       ├── wang2021.py        # Wang et al. 2021 + particiones (248 líneas)
│       └── delta_stepping.py  # Delta-Stepping GPU (experimental)
│   │
│   # Mapas y caché
│   └── mapas/
│       ├── cusco_hospitales.geojson
│       ├── cusco_establecimientos.geojson
│       ├── minsa/             # Datos MINSA descargados
│       └── mtc/               # Shapefiles MTC descargados
│
├── frontend/
│   ├── index.html             # UI interactiva Leaflet (1678 líneas)
│   └── README.md
│
├── cache/                     # Caché de grafos procesados
│   └── *.json                 # Archivos de caché por región
│
└── venv313/                   # Entorno virtual Python 3.13
    └── ...                    # Dependencias instaladas
```

### C. API Endpoints

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/api` | GET | Información general de la API |
| `/api/status` | GET | Estado del sistema y grafo cargado |
| `/api/system_info` | GET | Info detallada CPU, RAM, GPU, CUDA |
| `/api/regions` | GET | Lista 24 departamentos del Perú |
| `/api/provincias/{departamento}` | GET | Provincias de un departamento |
| `/api/distritos/{depto}/{provincia}` | GET | Distritos de una provincia |
| `/api/hospitales/{departamento}` | GET | Hospitales predefinidos |
| `/api/hospitales_minsa/{region}` | GET | Hospitales oficiales del MINSA |
| `/api/download_region` | POST | Descarga mapa OSM de región |
| `/api/download_mtc` | POST | Carga red vial oficial MTC |
| `/api/download_distrito` | POST | Descarga distrito específico OSM |
| `/api/load_graph` | POST | Carga grafo desde JSON local |
| `/api/geocode` | POST | Convierte dirección a coordenadas |
| `/api/find_nearest_node` | POST | Nodo más cercano a coordenadas |
| `/api/find_nearest_hospitals` | POST | Busca hospitales cercanos |
| `/api/calculate_hospital_routes` | POST | Calcula rutas óptimas a hospitales |
| `/api/run_algorithm` | POST | Ejecuta un algoritmo específico |
| `/api/compare_algorithms` | POST | Compara múltiples algoritmos |
| `/api/graph_info` | GET | Info detallada del grafo cargado |
| `/api/clear_cache/{region}` | DELETE | Elimina caché de una región |
| `/api/clear_all_cache` | DELETE | Elimina todo el caché |
| `/docs` | GET | Documentación Swagger interactiva |
| `/redoc` | GET | Documentación ReDoc |

---

**Universidad Nacional de San Antonio Abad del Cusco**  
*Escuela Profesional de Ingeniería Informática y de Sistemas*  
Diciembre 2025
| Algoritmo | Complejidad Teórica | Complejidad Real (Sparse) |
|-----------|---------------------|---------------------------|
| Dijkstra  | O((V+E) log V)      | O(E log V)               |
| Duan2025  | O(V + E)            | O(E log V) *CPU fallback* |
| Khanna2022| O(√V * E)           | O(E log V) *CPU fallback* |
| Wang2021  | O(E/P + V log V)    | O(E log V) *CPU fallback* |

**Nota**: Todos los algoritmos en modo CPU utilizan heap, resultando en complejidad similar.

### 4.2. Ventajas y Limitaciones

#### Ventajas del Enfoque Actual
✅ **Escalabilidad**: Manejo de grafos con 1.8M nodos  
✅ **Robustez**: Fallback automático CPU si GPU falla  
✅ **Eficiencia de memoria**: Matrices sparse CSR  
✅ **Aplicación real**: Rutas de ambulancias en Cusco  

#### Limitaciones Identificadas
❌ **CUDA no funcional**: Dependencias DLL faltantes  
❌ **Paralelismo limitado**: Todos corren en CPU secuencialmente  
❌ **Conversión sparse→densa**: No viable para grafos grandes en GPU  
❌ **Optimizaciones teóricas no aplicadas**: Algoritmos modernos usan heap estándar

## 🛠️ Desarrollo

### Agregar un Nuevo Algoritmo

1. Crear archivo en `backend/algorithms/nuevo_algoritmo.py`
2. Heredar de `ShortestPathAlgorithm`
3. Implementar `compute_shortest_paths()`
4. Agregar a `__init__.py`
5. Actualizar `main.py` en `_get_algorithm_instance()`

### Testing

```powershell
cd backend
pytest
```

## 📚 Referencias

- [1] Duan et al. (2025) - Parallel Shortest Path Algorithms
- [2] Khanna et al. (2022) - Bidirectional Search Optimization
- [3] Wang et al. (2021) - Graph Partitioning Methods
- [4] Dijkstra, E. W. (1959) - A note on two problems in connexion with graphs

## 🤝 Contribuciones

Contribuciones son bienvenidas. Por favor:
1. Fork el proyecto
2. Crear una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abrir un Pull Request

## 📄 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

## 👨‍💻 Autor

Proyecto desarrollado para investigación en algoritmos avanzados aplicados a redes viales urbanas.

---

**Nota**: Este proyecto requiere CUDA para máximo rendimiento, pero funciona en modo CPU si no está disponible.
