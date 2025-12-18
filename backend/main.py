"""
Backend FastAPI para comparación de algoritmos de caminos más cortos.
Proporciona API REST para ejecutar algoritmos y obtener métricas.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import os
import sys

# Importar módulos del proyecto
from graph import UrbanGraph
from algorithms.dijkstra import DijkstraAlgorithm, DijkstraPriorityQueue
from algorithms.duan2025 import Duan2025Algorithm
from algorithms.khanna2022 import Khanna2022Algorithm
from algorithms.wang2021 import Wang2021Algorithm

# Importar datos de regiones y hospitales
from regiones import get_all_departamentos
from hospitales import get_hospitales_region

# Crear aplicación FastAPI
app = FastAPI(
    title="Algoritmos de Caminos Más Cortos - API",
    description="Comparación de algoritmos con aceleración CUDA para redes viales urbanas",
    version="1.0.0"
)

# Configurar CORS para permitir solicitudes desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Grafo global (cargado en memoria)
urban_graph = UrbanGraph()
graph_loaded = False

# Modelos de datos para API
class AlgorithmRequest(BaseModel):
    algorithm: str  # 'dijkstra', 'duan2025', 'khanna2022', 'wang2021', 'all'
    source_node: int
    use_cuda: bool = True

class RegionDownloadRequest(BaseModel):
    region_key: str
    network_type: str = 'drive'

class CoordinateRequest(BaseModel):
    lat: float
    lon: float

class CompareAlgorithmsRequest(BaseModel):
    source_node: int
    algorithms: List[str] = ['dijkstra', 'duan2025', 'khanna2022', 'wang2021']
    use_cuda: bool = True

class HospitalRouteRequest(BaseModel):
    region_key: str
    user_lat: float
    user_lon: float
    algorithms: List[str] = ['dijkstra', 'duan2025']
    use_cuda: bool = True


# Endpoints de la API

@app.get("/api")
async def root():
    """Endpoint raíz con información de la API."""
    return {
        "message": "API de Comparación de Algoritmos de Caminos Más Cortos",
        "version": "1.0.0",
        "status": "running",
        "graph_loaded": graph_loaded,
        "nodes": urban_graph.num_nodes if graph_loaded else 0,
        "edges": urban_graph.num_edges if graph_loaded else 0
    }

@app.get("/api/status")
async def get_status():
    """Obtiene el estado actual del sistema."""
    return {
        "graph_loaded": graph_loaded,
        "num_nodes": urban_graph.num_nodes if graph_loaded else 0,
        "num_edges": urban_graph.num_edges if graph_loaded else 0,
        "cuda_available": _check_cuda_availability()
    }

@app.get("/api/regions")
async def list_regions():
    """Lista los 24 departamentos del Perú."""
    departamentos = get_all_departamentos()
    
    # Verificar cuáles tienen caché
    cache_dir = os.path.join(os.path.dirname(__file__), 'mapas')
    regions_data = []
    
    for key, info in departamentos.items():
        cache_file = os.path.join(cache_dir, f'{key}_drive.json')
        cached = os.path.exists(cache_file)
        
        regions_data.append({
            "key": key,
            "name": info['name'],
            "capital": info['capital'],
            "cached": cached,
            "cache_file": f'{key}_drive.json' if cached else None
        })
    
    return {"regions": regions_data}

@app.get("/api/provincias/{departamento_key}")
async def get_provincias(departamento_key: str):
    """Retorna provincias de un departamento."""
    provincias = UrbanGraph.get_provincias(departamento_key)
    if not provincias:
        return {"provincias": {}}
    return {"provincias": provincias}

@app.get("/api/distritos/{departamento_key}/{provincia_key}")
async def get_distritos(departamento_key: str, provincia_key: str):
    """Retorna distritos de una provincia."""
    distritos = UrbanGraph.get_distritos(departamento_key, provincia_key)
    if not distritos:
        return {"distritos": {}}
    return {"distritos": distritos}

@app.get("/api/hospitales/{departamento_key}")
async def get_hospitales(departamento_key: str):
    """Retorna hospitales predefinidos de un departamento."""
    hospitales = get_hospitales_region(departamento_key)
    if not hospitales:
        return {"hospitales": [], "message": f"No hay hospitales registrados para {departamento_key}"}
    return {"hospitales": hospitales, "total": len(hospitales)}

@app.post("/api/geocode")
async def geocode_address(address: str, region: str = None, provincia: str = None, distrito: str = None):
    """
    Geocodifica una dirección a coordenadas usando Nominatim.
    Construye query completa con región/provincia/distrito si están disponibles.
    """
    import requests
    
    # Construir query completa
    full_address = address
    if distrito:
        full_address += f", {distrito}"
    if provincia:
        full_address += f", {provincia}"
    if region:
        full_address += f", {region}"
    full_address += ", Peru"
    
    try:
        # Usar Nominatim de OpenStreetMap
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": full_address,
            "format": "json",
            "limit": 1,
            "addressdetails": 1
        }
        headers = {
            "User-Agent": "AlgoritmosAvanzados/1.0"
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()
        
        if data and len(data) > 0:
            result = data[0]
            return {
                "success": True,
                "lat": float(result['lat']),
                "lon": float(result['lon']),
                "display_name": result.get('display_name', address),
                "address": result.get('address', {})
            }
        else:
            return {
                "success": False,
                "message": "No se encontró la dirección. Intenta ser más específico."
            }
            
    except Exception as e:
        return {
            "success": False,
            "message": f"Error al geocodificar: {str(e)}"
        }

@app.post("/api/check_map")
async def check_map(distrito_key: str = None, region_key: str = None):
    """Verifica si existe caché para una ubicación."""
    result = urban_graph.check_map_cache(distrito_key, region_key)
    return result

@app.post("/api/download_region")
async def download_region(request: RegionDownloadRequest):
    """
    Carga mapa de región desde area.osm.json o pickle local.
    Para Cusco, usa directamente area.osm.json
    """
    global graph_loaded
    
    # Para Cusco, cargar directamente desde area.osm.json
    if request.region_key.lower() == 'cusco':
        base_dir = os.path.dirname(os.path.abspath(__file__))
        osm_json_path = os.path.join(os.path.dirname(base_dir), 'area.osm.json')
        
        if os.path.exists(osm_json_path):
            print(f"📦 Cargando Cusco desde {osm_json_path}...")
            success = urban_graph.load_from_osm_json(osm_json_path)
            
            if success:
                graph_loaded = True
                return {
                    "success": True,
                    "message": f"Cusco cargado desde area.osm.json",
                    "num_nodes": urban_graph.num_nodes,
                    "num_edges": urban_graph.num_edges
                }
            else:
                raise HTTPException(status_code=500, detail="Error al cargar area.osm.json")
        else:
            raise HTTPException(status_code=404, detail=f"Archivo area.osm.json no encontrado en {osm_json_path}")
    
    # Para otras regiones, intentar desde pickle
    cache_dir = os.path.join(os.path.dirname(__file__), 'mapas')
    pkl_file = os.path.join(cache_dir, f'{request.region_key}_graph.pkl')
    
    # Si NO existe, generar desde shapefile
    if not os.path.exists(pkl_file):
        print(f"📥 Generando {request.region_key} desde shapefile...")
        
        try:
            import subprocess
            import sys
            
            # Crear script de generación dinámico
            script_path = os.path.join(os.path.dirname(__file__), f'generar_{request.region_key}.py')
            
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(f'''
import geopandas as gpd
import networkx as nx
import pickle
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

print("🚀 Generando mapa de {request.region_key}...")

shapefile_path = Path(__file__).parent / 'mapas' / 'peru-251217-free.shp' / 'gis_osm_roads_free_1.shp'
roads = gpd.read_file(shapefile_path)

# Filtrar por región (ajustar bounds según departamento)
# Por ahora solo Cusco
cusco_bounds = {{'min_lon': -73.5, 'max_lon': -70.5, 'min_lat': -15.0, 'max_lat': -11.5}}
region_roads = roads.cx[cusco_bounds['min_lon']:cusco_bounds['max_lon'], cusco_bounds['min_lat']:cusco_bounds['max_lat']]

valid_types = ['motorway','trunk','primary','secondary','tertiary','unclassified','residential','motorway_link','trunk_link','primary_link','secondary_link','tertiary_link','living_street','service','road']
if 'fclass' in region_roads.columns:
    region_roads = region_roads[region_roads['fclass'].isin(valid_types)]

G = nx.MultiDiGraph()
node_id = 0
coords_to_id = {{}}

for idx, road in region_roads.iterrows():
    try:
        coords = list(road.geometry.coords)
        for i in range(len(coords) - 1):
            start = (round(coords[i][1], 6), round(coords[i][0], 6))
            end = (round(coords[i+1][1], 6), round(coords[i+1][0], 6))
            
            if start not in coords_to_id:
                coords_to_id[start] = node_id
                G.add_node(node_id, lat=start[0], lon=start[1])
                node_id += 1
            if end not in coords_to_id:
                coords_to_id[end] = node_id
                G.add_node(node_id, lat=end[0], lon=end[1])
                node_id += 1
            
            lat1, lon1 = radians(start[0]), radians(start[1])
            lat2, lon2 = radians(end[0]), radians(end[1])
            dlat, dlon = lat2 - lat1, lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            dist = 6371000 * 2 * atan2(sqrt(a), sqrt(1-a))
            
            G.add_edge(coords_to_id[start], coords_to_id[end], length=dist, highway=road.get('fclass','unknown'))
    except:
        continue

output = Path(__file__).parent / 'mapas' / '{request.region_key}_graph.pkl'
with open(output, 'wb') as f:
    pickle.dump({{'graph': G, 'num_nodes': G.number_of_nodes(), 'num_edges': G.number_of_edges()}}, f)

print(f"✅ Generado: {{G.number_of_nodes()}} nodos, {{G.number_of_edges()}} aristas")
''')
            
            # Ejecutar generación
            result = subprocess.run([sys.executable, script_path], 
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                raise Exception(f"Error: {result.stderr}")
            
            print(result.stdout)
            os.remove(script_path)  # Limpiar
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generando mapa: {str(e)}")
    
    # Cargar desde pickle
    success = urban_graph.download_region_from_osm(
        request.region_key,
        request.network_type
    )
    
    if success:
        graph_loaded = True
        return {
            "success": True,
            "message": f"Región {request.region_key} cargada exitosamente",
            "num_nodes": urban_graph.num_nodes,
            "num_edges": urban_graph.num_edges
        }
    else:
        raise HTTPException(status_code=500, detail="Error al cargar región")

@app.post("/api/download_distrito")
async def download_distrito(departamento_key: str, provincia_key: str, distrito_key: str):
    """Descarga un distrito específico desde OpenStreetMap."""
    global graph_loaded
    
    # Obtener query del distrito
    query, distrito_id = UrbanGraph.get_distrito_query(departamento_key, provincia_key, distrito_key)
    if not query:
        raise HTTPException(status_code=404, detail="Distrito no encontrado")
    
    success = urban_graph.download_distrito_from_osm(query, distrito_id)
    
    if success:
        graph_loaded = True
        return {
            "success": True,
            "message": f"Distrito {distrito_key} cargado exitosamente",
            "num_nodes": urban_graph.num_nodes,
            "num_edges": urban_graph.num_edges,
            "cache_key": distrito_id
        }
    else:
        raise HTTPException(status_code=500, detail="Error al descargar distrito")

@app.post("/api/load_graph")
async def load_graph(filepath: str = "../area.osm.json"):
    """Carga un grafo desde un archivo JSON local."""
    global graph_loaded
    
    # Ajustar ruta relativa
    if not os.path.isabs(filepath):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(base_dir, filepath)
    
    success = urban_graph.load_from_osm_json(filepath)
    
    if success:
        graph_loaded = True
        return {
            "success": True,
            "message": "Grafo cargado exitosamente",
            "num_nodes": urban_graph.num_nodes,
            "num_edges": urban_graph.num_edges
        }
    else:
        raise HTTPException(status_code=500, detail="Error al cargar grafo")

@app.delete("/api/clear_cache/{region_key}")
async def clear_cache(region_key: str):
    """Elimina el caché de una región para forzar nueva descarga."""
    cache_dir = os.path.join(os.path.dirname(__file__), 'mapas')
    cache_file = os.path.join(cache_dir, f'{region_key}_drive.json')
    
    if os.path.exists(cache_file):
        try:
            os.remove(cache_file)
            return {
                "success": True,
                "message": f"Caché de {region_key} eliminado. Próxima descarga será completa.",
                "file_deleted": f'{region_key}_drive.json'
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al eliminar caché: {str(e)}")
    else:
        return {
            "success": False,
            "message": f"No existe caché para {region_key}",
            "file_deleted": None
        }

@app.delete("/api/clear_all_cache")
async def clear_all_cache():
    """Elimina todo el caché de mapas."""
    cache_dir = os.path.join(os.path.dirname(__file__), 'mapas')
    
    if not os.path.exists(cache_dir):
        return {"success": True, "message": "No hay caché para eliminar", "files_deleted": 0}
    
    try:
        files_deleted = 0
        for filename in os.listdir(cache_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(cache_dir, filename)
                os.remove(filepath)
                files_deleted += 1
        
        return {
            "success": True,
            "message": f"Se eliminaron {files_deleted} archivos de caché",
            "files_deleted": files_deleted
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al limpiar caché: {str(e)}")

@app.post("/api/find_nearest_node")
async def find_nearest_node(request: CoordinateRequest):
    """Encuentra el nodo más cercano a una coordenada."""
    if not graph_loaded:
        raise HTTPException(status_code=400, detail="Grafo no cargado")
    
    nearest_idx = urban_graph.find_nearest_node(request.lat, request.lon)
    node_info = urban_graph.get_node_info(nearest_idx)
    
    return {
        "nearest_node": nearest_idx,
        "node_info": node_info
    }

@app.post("/api/run_algorithm")
async def run_algorithm(request: AlgorithmRequest):
    """Ejecuta un algoritmo específico."""
    if not graph_loaded:
        raise HTTPException(status_code=400, detail="Grafo no cargado. Use /load_graph primero.")
    
    # Validar nodo fuente
    if request.source_node < 0 or request.source_node >= urban_graph.num_nodes:
        raise HTTPException(status_code=400, detail="Nodo fuente inválido")
    
    # Seleccionar algoritmo
    algorithm = _get_algorithm_instance(request.algorithm, request.use_cuda)
    
    if algorithm is None:
        raise HTTPException(status_code=400, detail=f"Algoritmo '{request.algorithm}' no reconocido")
    
    # Ejecutar algoritmo
    try:
        graph_matrix = urban_graph.get_adjacency_matrix()
        metrics = algorithm.compute_shortest_paths(
            graph_matrix,
            request.source_node,
            urban_graph.node_mapping
        )
        
        # Obtener información del nodo fuente
        source_info = urban_graph.get_node_info(request.source_node)
        
        return {
            "success": True,
            "algorithm": request.algorithm,
            "source_node": request.source_node,
            "source_info": source_info,
            "metrics": metrics.to_dict(),
            "sample_paths": _get_sample_paths(metrics, urban_graph, num_samples=5)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al ejecutar algoritmo: {str(e)}")

@app.post("/api/compare_algorithms")
async def compare_algorithms(request: CompareAlgorithmsRequest):
    """Compara múltiples algoritmos y retorna métricas comparativas."""
    if not graph_loaded:
        raise HTTPException(status_code=400, detail="Grafo no cargado")
    
    # Validar nodo fuente
    if request.source_node < 0 or request.source_node >= urban_graph.num_nodes:
        raise HTTPException(status_code=400, detail="Nodo fuente inválido")
    
    results = []
    graph_matrix = urban_graph.get_adjacency_matrix()
    
    for algo_name in request.algorithms:
        try:
            algorithm = _get_algorithm_instance(algo_name, request.use_cuda)
            
            if algorithm is None:
                continue
            
            metrics = algorithm.compute_shortest_paths(
                graph_matrix,
                request.source_node,
                urban_graph.node_mapping
            )
            
            results.append({
                "algorithm": algo_name,
                "metrics": metrics.to_dict()
            })
        
        except Exception as e:
            results.append({
                "algorithm": algo_name,
                "error": str(e)
            })
    
    # Crear comparación
    comparison = _create_comparison_table(results)
    
    return {
        "success": True,
        "source_node": request.source_node,
        "source_info": urban_graph.get_node_info(request.source_node),
        "results": results,
        "comparison": comparison
    }

@app.get("/api/graph_info")
async def get_graph_info():
    """Obtiene información detallada del grafo cargado."""
    if not graph_loaded:
        raise HTTPException(status_code=400, detail="Grafo no cargado")
    
    # Calcular estadísticas del grafo
    adj_matrix = urban_graph.get_adjacency_matrix()
    
    # Grado de nodos
    node_degrees = np.sum(adj_matrix > 0, axis=1)
    
    return {
        "num_nodes": urban_graph.num_nodes,
        "num_edges": urban_graph.num_edges,
        "avg_degree": float(np.mean(node_degrees)),
        "max_degree": int(np.max(node_degrees)),
        "min_degree": int(np.min(node_degrees)),
        "density": float(urban_graph.num_edges / (urban_graph.num_nodes * (urban_graph.num_nodes - 1))),
        "sample_nodes": [
            urban_graph.get_node_info(i) 
            for i in range(min(10, urban_graph.num_nodes))
        ]
    }

@app.get("/api/hospitales/{region_key}")
async def get_hospitales(region_key: str):
    """Obtiene lista de hospitales de una región."""
    hospitales = UrbanGraph.get_hospitales_by_region(region_key)
    return {"region": region_key, "hospitales": hospitales}

@app.post("/api/find_hospitals")
async def find_nearest_hospitals(request: CoordinateRequest):
    """Encuentra hospitales cercanos a una ubicación."""
    if not graph_loaded:
        raise HTTPException(status_code=400, detail="Grafo no cargado")
    
    # Detectar región (simplificado - puedes mejorar esto)
    # Por ahora asumimos que el grafo cargado corresponde a una región
    region_key = 'cusco'  # Hardcoded por ahora, mejorar después
    
    hospitales = urban_graph.find_nearest_hospitals(
        request.lat, 
        request.lon,
        region_key,
        max_hospitals=5
    )
    
    return {
        "user_location": {"lat": request.lat, "lon": request.lon},
        "nearest_hospitals": hospitales
    }

@app.post("/api/calculate_hospital_routes")
async def calculate_hospital_routes(request: HospitalRouteRequest):
    """
    Calcula rutas óptimas a hospitales usando múltiples algoritmos.
    Automáticamente encuentra nodos, hospitales cercanos y compara algoritmos.
    """
    if not graph_loaded:
        raise HTTPException(status_code=400, detail="Grafo no cargado. Carga un mapa primero.")
    
    print(f"\n🚑 Calculando rutas desde ({request.user_lat}, {request.user_lon})")
    
    # 1. Encontrar nodo más cercano a la ubicación del usuario
    user_node = urban_graph.find_nearest_node(request.user_lat, request.user_lon)
    user_coords = urban_graph.get_node_info(user_node)
    print(f"👤 Usuario en nodo {user_node}: ({user_coords['lat']:.5f}, {user_coords['lon']:.5f})")
    
    # 2. Obtener hospitales cercanos del departamento
    hospitales = urban_graph.find_nearest_hospitals(
        request.user_lat,
        request.user_lon,
        request.region_key,
        max_hospitals=5
    )
    
    if not hospitales:
        raise HTTPException(
            status_code=404, 
            detail=f"No hay hospitales registrados para {request.region_key}"
        )
    
    print(f"🏥 Encontrados {len(hospitales)} hospitales cercanos")
    
    # 3. Calcular rutas con cada algoritmo seleccionado
    all_routes = []
    graph_matrix = urban_graph.get_adjacency_matrix()
    
    for hospital in hospitales[:3]:  # Top 3 hospitales más cercanos
        hospital_node = hospital['node_index']
        hospital_coords = urban_graph.get_node_info(hospital_node)
        
        print(f"\n🏥 {hospital['name']} (Nodo {hospital_node})")
        
        for algo_name in request.algorithms:
            try:
                # Instanciar algoritmo
                algorithm = _get_algorithm_instance(algo_name, request.use_cuda)
                if algorithm is None:
                    print(f"  ⚠️  Algoritmo {algo_name} no disponible")
                    continue
                
                # Ejecutar algoritmo
                print(f"  🔄 Ejecutando {algo_name}...")
                metrics = algorithm.compute_shortest_paths(
                    graph_matrix,
                    user_node,
                    urban_graph.node_mapping
                )
                
                # Obtener ruta al hospital
                if hospital_node in metrics.path_to_nodes:
                    path_indices = metrics.path_to_nodes[hospital_node]
                    distance = metrics.distances_computed.get(hospital_node, 0)
                    
                    # Convertir índices a coordenadas
                    path_coordinates = []
                    for node_idx in path_indices:
                        node_info = urban_graph.get_node_info(node_idx)
                        if node_info:
                            path_coordinates.append([node_info['lat'], node_info['lon']])
                    
                    # Agregar resultado
                    all_routes.append({
                        "hospital_name": hospital['name'],
                        "hospital_tipo": hospital.get('tipo', 'General'),
                        "hospital_nivel": hospital.get('nivel', 'II-1'),
                        "hospital_lat": hospital['lat'],
                        "hospital_lon": hospital['lon'],
                        "hospital_node": hospital_node,
                        "distance_direct_km": hospital['distance_direct_km'],
                        "algorithm": algo_name,
                        "path_distance": distance / 1000,  # Convertir a km
                        "path_coordinates": path_coordinates,
                        "num_nodes_in_path": len(path_indices),
                        "metrics": {
                            "execution_time": metrics.execution_time,
                            "nodes_processed": metrics.nodes_processed,
                            "edge_relaxations": metrics.edge_relaxations,
                            "memory_peak_mb": metrics.memory_peak_mb
                        }
                    })
                    
                    print(f"    ✅ {distance/1000:.2f} km, {metrics.execution_time:.3f}s, {len(path_indices)} nodos")
                else:
                    print(f"    ❌ No se encontró ruta")
                    
            except Exception as e:
                print(f"    ❌ Error en {algo_name}: {e}")
                continue
    
    # 4. Ordenar por distancia y algoritmo más rápido
    all_routes.sort(key=lambda x: (x['hospital_name'], x['metrics']['execution_time']))
    
    print(f"\n✅ Calculadas {len(all_routes)} rutas exitosamente")
    
    return {
        "success": True,
        "user_location": {
            "lat": request.user_lat,
            "lon": request.user_lon,
            "node_index": user_node
        },
        "region": request.region_key,
        "algorithms_used": request.algorithms,
        "routes": all_routes,
        "total_routes": len(all_routes),
        "hospitals_analyzed": len(hospitales)
    }



# Funciones auxiliares

@app.get("/api/system_info")
async def get_system_info():
    """Obtiene información detallada de CUDA/GPU, CPU y RAM."""
    import platform
    try:
        import psutil  # type: ignore
    except ImportError:
        psutil = None  # type: ignore
    
    # Información del sistema (CPU y RAM)
    if psutil is not None:
        system_info = {
            "cpu": {
                "processor": platform.processor(),
                "architecture": platform.machine(),
                "cores_physical": psutil.cpu_count(logical=False),
                "cores_logical": psutil.cpu_count(logical=True),
                "frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
                "usage_percent": psutil.cpu_percent(interval=0.1)
            },
            "ram": {
                "total_gb": psutil.virtual_memory().total / (1024**3),
                "available_gb": psutil.virtual_memory().available / (1024**3),
                "used_gb": psutil.virtual_memory().used / (1024**3),
                "percent_used": psutil.virtual_memory().percent
            },
            "platform": {
                "os": platform.system(),
                "version": platform.version(),
                "python_version": platform.python_version()
            }
        }
    else:
        # Valores por defecto si psutil no está disponible, para evitar error 500
        system_info = {
            "cpu": {
                "processor": platform.processor(),
                "architecture": platform.machine(),
                "cores_physical": 0,
                "cores_logical": 0,
                "frequency_mhz": 0,
                "usage_percent": 0.0
            },
            "ram": {
                "total_gb": 0.0,
                "available_gb": 0.0,
                "used_gb": 0.0,
                "percent_used": 0.0
            },
            "platform": {
                "os": platform.system(),
                "version": platform.version(),
                "python_version": platform.python_version()
            }
        }
    
    # Primero verificar si hay GPU NVIDIA en el sistema (mediante nvidia-smi)
    nvidia_gpu_detected = False
    nvidia_gpu_name = None
    cuda_version_system = None
    
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], 
                              capture_output=True, text=True, timeout=2)
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            if lines:
                nvidia_gpu_detected = True
                nvidia_gpu_name = lines[0].split(',')[0].strip()
        
        # Obtener versión de CUDA del driver
        result_cuda = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=2)
        if result_cuda.returncode == 0:
            import re
            match = re.search(r'CUDA Version:\s+(\d+\.\d+)', result_cuda.stdout)
            if match:
                cuda_version_system = match.group(1)
    except:
        pass  # Si falla nvidia-smi, asumimos que no hay GPU
    
    # Intentar obtener información de CUDA/GPU mediante CuPy
    try:
        import cupy as cp
        
        cuda_info = {
            "available": True,
            "version": cp.cuda.runtime.runtimeGetVersion(),
            "driver_version": cp.cuda.runtime.driverGetVersion(),
            "device_count": cp.cuda.runtime.getDeviceCount()
        }
        
        devices = []
        for i in range(cuda_info["device_count"]):
            cp.cuda.Device(i).use()
            props = cp.cuda.runtime.getDeviceProperties(i)
            mem_info = cp.cuda.runtime.memGetInfo()
            
            device_info = {
                "id": i,
                "name": props["name"].decode(),
                "compute_capability": f"{props['major']}.{props['minor']}",
                "total_memory_gb": props["totalGlobalMem"] / (1024**3),
                "free_memory_gb": mem_info[0] / (1024**3),
                "used_memory_gb": (props["totalGlobalMem"] - mem_info[0]) / (1024**3),
                "multiprocessors": props["multiProcessorCount"],
                "cuda_cores": props["multiProcessorCount"] * 128,
                "max_threads_per_block": props["maxThreadsPerBlock"],
                "clock_rate_mhz": props["clockRate"] / 1000,
                "memory_clock_rate_mhz": props["memoryClockRate"] / 1000,
                "memory_bus_width": props["memoryBusWidth"],
                "l2_cache_size_mb": props["l2CacheSize"] / (1024**2)
            }
            devices.append(device_info)
        
        cuda_info["devices"] = devices
        recommendation = "✅ Aceleración GPU habilitada (puedes desmarcar el checkbox para forzar solo CPU)"
        
    except ImportError:
        # CuPy no está disponible en este intérprete de Python
        import sys
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

        # Si detectamos GPU NVIDIA en el sistema pero CuPy no está instalado
        if nvidia_gpu_detected:
            if sys.version_info.major == 3 and sys.version_info.minor >= 14:
                install_msg = (
                    f"GPU NVIDIA detectada ({nvidia_gpu_name}), pero CuPy no soporta Python {python_version}. "
                    "Instala Python 3.13 y ejecuta: pip install cupy-cuda13x && uvicorn con ese Python."
                )
                install_cmd = f"# 1) Instalar Python 3.13\n# 2) pip install cupy-cuda13x\n# 3) Ejecutar backend con Python 3.13"
            else:
                install_msg = (
                    f"GPU NVIDIA detectada ({nvidia_gpu_name}, CUDA {cuda_version_system or '?'}), "
                    "pero CuPy no está instalado en este Python. "
                    "Instala CuPy aquí o arranca con otro Python que lo tenga."
                )
                install_cmd = f"pip install cupy-cuda13x  # Para CUDA {cuda_version_system or '12.x/13.x'}"
        else:
            # No se detectó GPU NVIDIA en el sistema
            if sys.version_info.major == 3 and sys.version_info.minor >= 14:
                install_msg = (
                    f"No se detectó GPU NVIDIA. Si tienes una, verifica drivers y que CuPy soporte Python {python_version} "
                    "(actualmente solo hasta 3.13)."
                )
                install_cmd = "nvidia-smi  # Verificar GPU y drivers"
            else:
                install_msg = (
                    "No se detectó GPU NVIDIA o CuPy no está instalado. "
                    "Verifica que tengas GPU NVIDIA con drivers, luego instala CuPy."
                )
                install_cmd = "pip install cupy-cuda13x  # Ajustar XX según tu CUDA"

        cuda_info = {
            "available": False,
            "message": install_msg,
            "install_command": install_cmd,
            "python_version": python_version,
            "nvidia_gpu_detected": nvidia_gpu_detected,
            "nvidia_gpu_name": nvidia_gpu_name,
            "cuda_version_system": cuda_version_system
        }
        recommendation = "⚠️ Aceleración GPU no disponible - usando solo CPU"
        
    except Exception as e:
        # Cualquier otro error al inicializar CuPy/CUDA
        cuda_info = {
            "available": False,
            "error": str(e),
            "message": f"Error al inicializar CUDA: {str(e)}. Revisa drivers, versión de CUDA y CuPy.",
            "nvidia_gpu_detected": nvidia_gpu_detected,
            "nvidia_gpu_name": nvidia_gpu_name
        }
        recommendation = "⚠️ CUDA no pudo inicializarse - cálculos en CPU"
    
    return {
        "success": True,
        "system": system_info,
        "cuda": cuda_info,
        "recommendation": recommendation
    }

def _check_cuda_availability() -> bool:
    """Verifica si CUDA está disponible."""
    try:
        import cupy as cp
        return True
    except ImportError:
        return False

def _get_algorithm_instance(algo_name: str, use_cuda: bool):
    """Obtiene una instancia del algoritmo solicitado."""
    algorithms = {
        'dijkstra': DijkstraAlgorithm,
        'dijkstra_pq': DijkstraPriorityQueue,
        'duan2025': Duan2025Algorithm,
        'khanna2022': Khanna2022Algorithm,
        'wang2021': Wang2021Algorithm
    }
    
    algo_class = algorithms.get(algo_name.lower())
    if algo_class:
        return algo_class(use_cuda=use_cuda)
    return None

def _get_sample_paths(metrics, graph: UrbanGraph, num_samples: int = 5):
    """Obtiene caminos de muestra con sus coordenadas."""
    sample_paths = []
    
    # Obtener hasta num_samples caminos
    path_items = list(metrics.path_to_nodes.items())[:num_samples]
    
    for target, path in path_items:
        path_coordinates = []
        
        for node_idx in path:
            node_info = graph.get_node_info(node_idx)
            if node_info:
                path_coordinates.append({
                    'node': node_idx,
                    'lat': node_info['lat'],
                    'lon': node_info['lon']
                })
        
        sample_paths.append({
            'target': target,
            'distance': metrics.distances_computed.get(target, float('inf')),
            'path': path,
            'coordinates': path_coordinates
        })
    
    return sample_paths

def _create_comparison_table(results: List[Dict]) -> Dict:
    """Crea una tabla comparativa de resultados."""
    comparison = {
        'fastest': None,
        'fewest_nodes': None,
        'fewest_relaxations': None,
        'least_memory': None,
        'speedup_vs_dijkstra': {}
    }
    
    # Encontrar el mejor en cada categoría
    min_time = float('inf')
    min_nodes = float('inf')
    min_relaxations = float('inf')
    min_memory = float('inf')
    dijkstra_time = None
    
    for result in results:
        if 'error' in result:
            continue
        
        metrics = result['metrics']
        algo_name = result['algorithm']
        
        if metrics['execution_time'] < min_time:
            min_time = metrics['execution_time']
            comparison['fastest'] = algo_name
        
        if metrics['nodes_processed'] < min_nodes:
            min_nodes = metrics['nodes_processed']
            comparison['fewest_nodes'] = algo_name
        
        if metrics['edge_relaxations'] < min_relaxations:
            min_relaxations = metrics['edge_relaxations']
            comparison['fewest_relaxations'] = algo_name
        
        if metrics['memory_peak_mb'] < min_memory:
            min_memory = metrics['memory_peak_mb']
            comparison['least_memory'] = algo_name
        
        if 'dijkstra' in algo_name.lower():
            dijkstra_time = metrics['execution_time']
    
    # Calcular speedup respecto a Dijkstra
    if dijkstra_time and dijkstra_time > 0:
        for result in results:
            if 'error' not in result:
                algo_name = result['algorithm']
                algo_time = result['metrics']['execution_time']
                speedup = dijkstra_time / algo_time if algo_time > 0 else 0
                comparison['speedup_vs_dijkstra'][algo_name] = round(speedup, 2)
    
    return comparison


# Montar archivos estáticos del frontend (DEBE IR AL FINAL después de todas las rutas API)
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Iniciando servidor backend...")
    print("📊 API disponible en: http://localhost:8000")
    print("📝 Documentación: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
