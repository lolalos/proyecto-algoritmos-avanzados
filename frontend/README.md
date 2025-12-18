# 🚀 Proyecto: Algoritmos de Caminos Más Cortos con CUDA

Sistema de comparación de algoritmos de caminos más cortos optimizados con aceleración GPU (CUDA) para redes viales urbanas de Perú.

## 📋 Características

- **4 Algoritmos Implementados**:
  - Dijkstra (versión clásica y con cola de prioridad)
  - Duan et al. (2025) - Procesamiento paralelo por fronteras
  - Khanna et al. (2022) - Búsqueda bidireccional con heurísticas
  - Wang et al. (2021) - Particionamiento de grafos

- **Aceleración GPU con CUDA**:
  - Procesamiento paralelo usando CuPy
  - Fallback automático a CPU si CUDA no está disponible
  - Optimización de operaciones matriciales

- **Métricas de Comparación**:
  - ⏱️ Tiempo de ejecución total
  - 🔢 Número de nodos procesados
  - 🔄 Número de relajaciones de aristas
  - 💾 Uso de memoria (MB)
  - 📈 Escalabilidad
  - ✅ Calidad de ruta

- **Soporte para Mapas de Perú**:
  - 12 regiones principales disponibles
  - Descarga directa desde OpenStreetMap
  - Procesamiento de archivos OSM JSON

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

## 🎯 Optimizaciones Implementadas

### Aceleración CUDA
- Operaciones vectorizadas con CuPy
- Procesamiento paralelo de nodos
- Reducción de transferencias GPU-CPU

### Algoritmos Específicos

**Duan2025**:
- Procesamiento por fronteras
- Actualización paralela de distancias
- Reducción de sincronización

**Khanna2022**:
- Búsqueda bidireccional
- Heurísticas de poda
- Priorización por grado de nodo

**Wang2021**:
- Particionamiento de grafo
- Procesamiento independiente de particiones
- Fusión eficiente de resultados

## 📝 Ejemplo de Resultados

```
Comparación de Algoritmos (Grafo de Lima - 5000 nodos)
─────────────────────────────────────────────────────
Algoritmo          | Tiempo(s) | Nodos | Relax. | Memoria(MB)
───────────────────┼───────────┼───────┼────────┼────────────
Dijkstra           | 0.0234    | 5000  | 12450  | 2.34
Duan2025           | 0.0156    | 4892  | 11203  | 2.89
Khanna2022         | 0.0198    | 4756  | 10987  | 2.56
Wang2021           | 0.0172    | 4823  | 11456  | 3.12

Mejor en cada categoría:
⚡ Más rápido: Duan2025 (1.5x speedup)
🔢 Menos nodos: Khanna2022
💾 Menos memoria: Dijkstra
```

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
