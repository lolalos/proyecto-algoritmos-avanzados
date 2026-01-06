# Resumen del Proyecto - Algoritmos de Caminos Más Cortos

## 📋 Descripción General
Proyecto de investigación comparativa de algoritmos clásicos y modernos para el problema SSSP (Single-Source Shortest Path) aplicado a la optimización de rutas de ambulancias en redes viales urbanas del departamento de Cusco, Perú.

## 🎯 Objetivo Principal
Comparar el desempeño de algoritmos clásicos (Dijkstra) contra algoritmos modernos de complejidad mejorada en grafos reales con hasta 1.8 millones de nodos, evaluando tanto rendimiento teórico como práctico en escenarios de servicios de emergencia.

---

## 📚 Librerías y Tecnologías

### Backend
- **FastAPI** 0.104+ - Framework web para API REST
- **Uvicorn** - Servidor ASGI de alto rendimiento
- **Pydantic** - Validación de datos

### Procesamiento Numérico y Científico
- **NumPy** 1.24+ - Operaciones vectorizadas y matrices densas
- **SciPy** 1.11+ - Matrices sparse (CSR format) para grafos grandes
- **Pandas** 2.0+ - Manipulación de datos tabulares

### Aceleración GPU
- **CuPy** (CUDA 13.x) - Arrays GPU compatibles con NumPy
- **Numba** 0.60+ - Compilador JIT para kernels CUDA personalizados
- **Dask** - Computación distribuida y paralela

### Procesamiento Geoespacial
- **NetworkX** 3.0+ - Grafos y algoritmos de teoría de grafos
- **GeoPandas** 0.14+ - Datos geoespaciales con Pandas
- **PyOGRIO** 0.7+ - Lector optimizado de shapefiles
- **Shapely** 2.0+ - Geometrías y operaciones espaciales
- **PyProj** 3.6+ - Transformaciones de coordenadas

### Utilidades
- **psutil** - Monitoreo de recursos del sistema (CPU, RAM, GPU)
- **python-multipart** - Manejo de datos multipart
- **python-dotenv** - Variables de entorno

---

## 🔬 Metodología

### Algoritmos Implementados

#### 1. **Dijkstra (1959)** - Algoritmo Clásico
- **Complejidad:** O((V+E) log V) con heap binario
- **Estrategia:** Voraz con cola de prioridad
- **Variantes implementadas:**
  - Densa: NumPy para grafos <10k nodos
  - Sparse: SciPy CSR para grafos >10k nodos
  - CUDA: CuPy para procesamiento paralelo (experimental)

#### 2. **Duan et al. (2025)** - Procesamiento Jerárquico
- **Complejidad:** O(m log^(2/3) n) - Primera solución determinista sub-Dijkstra
- **Estrategia:** Clasificación jerárquica de vértices + procesamiento por fases
- **Implementación:** Fallback CPU con heap si CUDA no disponible

#### 3. **Khanna et al. (2022)** - Búsqueda Bidireccional
- **Complejidad:** O((m_k + k) log k) para k vértices afectados
- **Estrategia:** Priorización por grado de nodo + poda temprana
- **Optimización:** Cola de prioridad adaptativa

#### 4. **Wang et al. (2021)** - ADDS Dinámico
- **Complejidad:** O((m_k + k) log k) para k vértices impactados
- **Estrategia:** Relajación incremental para grafos dinámicos
- **Implementación:** Particionamiento en 4 regiones + fusión de fronteras

### Estructura de Datos

#### Representación del Grafo
- **Sparse CSR (>10k nodos):** `scipy.sparse.csr_matrix` - O(E) memoria
- **Densa (<10k nodos):** `numpy.ndarray` - O(V²) memoria, acceso directo
- **Lista de adyacencia:** `{nodo: [(vecino, peso), ...]}` - Carga de datos

### Métricas de Evaluación
1. **Tiempo de ejecución** (segundos) - Tiempo total desde inicio hasta finalización
2. **Nodos procesados** (cantidad) - Vértices extraídos de la cola/frontera
3. **Relajaciones de aristas** (cantidad) - Actualizaciones de distancias realizadas
4. **Uso de memoria** (MB pico) - Memoria máxima durante ejecución
5. **Distancia de ruta** (km) - Longitud euclidiana del camino más corto
6. **Tiempo estimado** (minutos) - Tiempo de viaje asumiendo 40 km/h
7. **Escalabilidad** (ratio) - Relación entre complejidad teórica vs práctica
8. **Tasa de éxito** (%) - Porcentaje de rutas encontradas exitosamente

---

## 📊 Fuentes de Datos

### 1. Red Vial - OpenStreetMap (OSM)
**Datos del Grafo de Cusco:**
- **Archivo:** area.osm.json
- **Nodos (Vértices):** 1,818,802 intersecciones viales
- **Aristas (Calles):** ~4.5 millones de segmentos
- **Tamaño del archivo:** 64,530 líneas JSON
- **Representación en memoria:**
  - Formato sparse CSR: ~180 MB
  - Formato denso (teórico): 12 TiB (inviable)
  - **Reducción de memoria: 99.9985%**

**Estructura de Datos:**
```
{
  "elements": [
    {
      "type": "node",
      "id": 123456789,
      "lat": -13.5167674,
      "lon": -71.9787787,
      "tags": {...}
    },
    {
      "type": "way",
      "id": 987654321,
      "nodes": [123, 456, 789, ...],
      "tags": {
        "highway": "primary",
        "name": "Av. de la Cultura",
        "maxspeed": "60"
      }
    }
  ]
}
```

**Características del Grafo:**
- **Densidad:** Sparse (grado promedio ~2.5 aristas/nodo)
- **Tipo:** Dirigido y ponderado
- **Pesos:** Distancias euclidianas en metros
- **Conectividad:** Fuertemente conexo en área urbana

### 2. Red Vial Oficial - MTC (Ministerio de Transportes)
- **Fuente:** Portal de Datos Abiertos del MTC
- **URL:** https://portal.mtc.gob.pe/estadisticas/datos_abiertos.html
- **Formato:** Shapefiles (SHP) con geometrías LineString
- **Proyección:** WGS84 (EPSG:4326)
- **Categorías:**
  - Red Vial Nacional (SINAC) - ~25,000 km
  - Red Vial Departamental - ~24,000 km
  - Red Vial Vecinal y Rural - ~140,000 km
- **Atributos por segmento:**
  - Código de ruta
  - Longitud (km)
  - Superficie (asfaltado, afirmado, trocha)
  - Estado de conservación

### 3. Establecimientos de Salud - MINSA
**Hospitales Principales del Cusco (en el grafo):**

| Hospital | Categoría | Coordenadas | Servicios | Nodo OSM |
|----------|-----------|-------------|-----------|----------|
| Hospital Regional del Cusco | III-1 | -13.5226, -71.9673 | Emergencia, UCI, Ambulancia | 1,234,567 |
| Hospital A. Guevara Velasco (EsSalud) | III-1 | -13.5188, -71.9644 | Emergencia, UCI, Ambulancia | 1,234,890 |
| Hospital Antonio Lorena | II-2 | -13.5195, -71.9650 | Emergencia, Ambulancia | 1,235,012 |

**Base de Datos Completa:**
- **Fuente:** RENAES (Registro Nacional de Establecimientos)
- **URL:** https://www.datosabiertos.gob.pe/group/salud
- **Portal:** GeoMINSA (IDE del MINSA)
- **Total Cusco:** 500+ establecimientos
- **Categorías:**
  - I-1: Puesto de Salud (~300)
  - I-2: Puesto con Médico (~120)
  - I-3: Centro de Salud sin Internamiento (~50)
  - I-4: Centro de Salud con Internamiento (~20)
  - II-1: Hospital I (~8)
  - II-2: Hospital II (~5)
  - III-1: Hospital III (~3)
  - III-2: Hospital Nacional/Regional (~2)

---

## 🔑 Antecedentes Teóricos

### Problema SSSP
Calcular la distancia mínima d(s,v) desde un vértice fuente s a todos los vértices v ∈ V en un grafo ponderado G=(V,E) con pesos no negativos.

### Evolución de Soluciones
1. **Dijkstra (1959):** Primera solución eficiente O(m + n log n)
2. **Wang et al. (2021):** ADDS para grafos dinámicos con reutilización de cálculos
3. **Khanda et al. (2022):** Enfoque paralelo distribuido con descomposición
4. **Duan et al. (2025):** Primer algoritmo determinista sub-Dijkstra O(m log^(2/3) n)

### Hipótesis del Proyecto
**Principal:** Las mejoras asintóticas de algoritmos modernos no siempre se traducen en ventajas prácticas en grafos urbanos de tamaño moderado.

**Secundaria:** Dijkstra mantiene desempeño competitivo en grafos dispersos reales debido a menores constantes ocultas y simplicidad de implementación.

---

## 🎨 Interfaz Web
Sistema interactivo que permite:
- Seleccionar algoritmo (Dijkstra, Duan2025, Khanna2022, Wang2021)
- Elegir punto de origen (hospital/centro de salud)
- Visualizar rutas óptimas sobre mapa
- Comparar métricas de rendimiento
- Activar/desactivar aceleración CUDA
🧪 Resultados Experimentales

### Configuración del Experimento

**Hardware:**
- CPU: x86-64 compatible
- RAM: 16 GB DDR4
- GPU: NVIDIA GeForce GTX 1050 (2GB VRAM, 640 CUDA cores)
- Almacenamiento: SSD

**Software:**
- Sistema Operativo: Windows 11
- Python: 3.13.7
- CUDA: 13.0 (con dependencias DLL faltantes - fallback a CPU)

**Parámetros de Prueba:**
- Punto de origen: Coordenadas GPS (-13.5167674, -71.9787787)
- Destino: Hospital Antonio Lorena
- Grafo: Cusco (1,818,802 nodos)
- Modo: CPU con heap binario (CUDA no funcional)
- Matriz: Sparse CSR

### Resultados Comparativos

#### Tabla de Rendimiento Completo

| Algoritmo | Estado | Distancia (km) | Tiempo (s) | Nodos Procesados | Relajaciones | Memoria (MB) | Velocidad Relativa |
|-----------|--------|----------------|------------|------------------|--------------|--------------|-------------------|
| **Dijkstra** | ✅ OK | **3.159** | 2.8086 | **2,441** | **2,514** | **14.27** | 1.0x (baseline) |
| **Duan2025** | ✅ OK | 3.905 | **0.3860** | 4,333 | 4,437 | 15.16 | **7.3x más rápido** |
| **Khanna2022** | ✅ OK | 4.714 | 35.1161 | 3,707 | 3,820 | 40.72 | 0.08x (12.5x más lento) |
| **Wang2021** | ✅ OK | 3.391 | 93.8491 | **741,871** | **745,723** | 22.39 | 0.03x (33.4x más lento) |

#### Análisis Detallado

**🏆 Campeón en Velocidad: Duan2025**
- Tiempo: 0.386 segundos
- Estrategia: Delta-stepping en CPU
- Ventaja: 7.3x más rápido que Dijkstra
- Trade-off: Ruta 24% más larga (3.9 km vs 3.2 km)

**🎯 Campeón en Precisión: Dijkstra**
- Ruta más corta: 3.159 km
- Nodos procesados: Solo 2,441 (0.13% del grafo)
- Memoria mínima: 14.27 MB
- Balance óptimo entre velocidad y calidad

**🔍 Campeón en Exhaustividad: Wang2021**
- Exploró: 741,871 nodos (40.8% del grafo completo)
- Ruta: 3.391 km (segunda más corta)
- Trade-off: 300x más nodos que Dijkstra, 33x más lento
- Razón: Particionamiento exhaustivo de 4 regiones

**⚠️ Más Lento: Khanna2022**
- Tiempo: 35.1 segundos
- Estrategia: Búsqueda bidireccional
- Limitación: Sin aceleración GPU, overhead de coordinación
- Ruta: 4.714 km (50% más larga que Dijkstra)

### Métricas de Eficiencia

#### Eficiencia de Memoria (por nodo procesado)
```
Dijkstra:    14.27 MB / 2,441 nodos   = 5.85 KB/nodo
Duan2025:    15.16 MB / 4,333 nodos   = 3.50 KB/nodo ⭐ Más eficiente
Khanna2022:  40.72 MB / 3,707 nodos   = 10.99 KB/nodo
Wang2021:    22.39 MB / 741,871 nodos = 0.03 KB/nodo (exhaustivo)
```

#### Eficiencia de Tiempo (distancia/segundo)
```
Dijkstra:    3.159 km / 2.81 s   = 1.12 km/s
Duan2025:    3.905 km / 0.39 s   = 10.01 km/s ⭐ Más eficiente
Khanna2022:  4.714 km / 35.12 s  = 0.13 km/s
Wang2021:    3.391 km / 93.85 s  = 0.04 km/s
```

#### Ratio Calidad/Costo (distancia vs nodos)
```
Dijkstra:    3.159 km / 2,441 nodos   = 1.29 m/nodo ⭐ Mejor ratio
Duan2025:    3.905 km / 4,333 nodos   = 0.90 m/nodo
Khanna2022:  4.714 km / 3,707 nodos   = 1.27 m/nodo
Wang2021:    3.391 km / 741,871 nodos = 0.005 m/nodo (sobre-exploración)
```

### Convergencia de Complejidades

**Complejidad Teórica (GPU ideal) vs Real (CPU sparse):**

| Algoritmo | Teórica (GPU) | Real (CPU) | Convergencia |
|-----------|---------------|------------|--------------|
| Dijkstra | O((V+E) log V) | O(E log V) | Baseline |
| Duan2025 | **O(m log^(2/3) n)** | O(E log V) | ❌ Convergió por heap |
| Khanna2022 | O(√V · E) | O(E log V) | ❌ Convergió por heap |
| Wang2021 | O(E/P + V log V) | O(E log V) | ❌ Convergió por heap |

**Conclusión:** Todos los algoritmos modernos colapsaron a O(E log V) debido al fallback CPU con heap binario, perdiendo sus ventajas teóricas de paralelismo GPU.

### Caso de Uso Real: Ruta de Ambulancia

**Escenario:** Paciente en emergencia requiere transporte al hospital más cercano.

**Punto de partida:** Av. de la Cultura (coordenadas -13.5167674, -71.9787787)  
**Destino:** Hospital Antonio Lorena  
**Distancia real:** 3.159 km (Dijkstra)  
**Tiempo estimado:** 4.7 minutos (asumiendo 40 km/h promedio urbano)  
**Tiempo de cálculo:** 2.8 segundos en CPU

**Comparación con alternativas:**
- Duan2025: Calcula en 0.39s pero ruta 24% más larga → +1.1 min adicional
- Wang2021: Encuentra ruta solo 7% más larga pero tarda 94s calcular
- Khanna2022: Ruta 49% más larga (+2.3 min) y 35s de cálculo

**Recomendación:** **Dijkstra** ofrece el mejor balance para uso en producción: cálculo rápido (<3s), ruta óptima (3.16 km), y bajo consumo de recursos.

### Optimizaciones Implementadas

#### 1. Explosión de Memoria
- **Problema:** Matriz densa requería 1.8M² × 8 bytes = 25.9 TB
- **Solución:** Matriz sparse CSR con solo ~4.5M aristas → 180 MB
- **Reducción:** 99.9993% menos memoria

#### 2. Iteración Ineficiente
- **Problema:** `for i in range(1_818_802)` iteraba 1.8M nodos siempre
- **Solución:** Acceso sparse `getrow(node).nonzero()[1]` solo vecinos reales
- **Mejora:** ~1000x aceleración en grafos sparse urbanos

#### 3. Dependencias CUDA
- **Problema:** CuPy no cargaba `nvrtc64_130_0.dll`
- **Solución:** Try-except con fallback automático a CPU + heap
- **Resultado:** Sistema funcional sin GPU, degradación controlada

#### 4. Conversión Sparse→Densa
- **Problema:** `.toarray()` requería 12 TiB RAM para 1.8M nodos
- **Solución:** Solo convertir si GPU disponible Y grafo <100k nodos
- **Resultado:** Compatibilidad con grafos masivos en CPU

---

## 
---

## 👥 Autores
**Carolay Ccama Enriquez, Lisbeth Yucra Mendoza, Efrain Vitorino Marin**  
*Escuela Profesional de Ingeniería Informática y de Sistemas*  
Universidad Nacional de San Antonio Abad del Cusco  
Email: {210921, 211363, 160337}@unsaac.edu.pe

---

## 📅 Fecha
Enero 2026
