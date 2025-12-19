"""
Módulo para cargar y procesar grafos viales desde datos OpenStreetMap.
Incluye soporte para descargar mapas de regiones de Perú.
"""
import json
import numpy as np
from scipy import sparse
from typing import Dict, List, Tuple, Optional
import os

# Importar datos de regiones y hospitales
from regiones import (
    DEPARTAMENTOS_PERU, 
    get_all_departamentos,
    get_provincias as get_provincias_data,
    get_distritos as get_distritos_data,
    get_distrito_query as get_distrito_query_data,
    get_region_query
)
from hospitales import (
    HOSPITALES_PERU,
    get_hospitales_region
)

try:
    import osmnx as ox
    import networkx as nx
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False
    ox = None
    nx = None

# Importar módulos de datos oficiales
try:
    from descargar_mtc import MTCDataDownloader
    from hospitales_minsa import MINSADataDownloader
    MTC_MINSA_AVAILABLE = True
except ImportError:
    MTC_MINSA_AVAILABLE = False
    MTCDataDownloader = None
    MINSADataDownloader = None


class UrbanGraph:
    """Representa un grafo de red vial urbana."""
    
    def __init__(self):
        self.graph = None
        self.adjacency_matrix = None  # Será matriz dispersa (sparse)
        self.adjacency_list = {}  # Lista de adyacencia para acceso rápido
        self.node_mapping = {}  # Mapeo de índice -> ID de nodo OSM
        self.reverse_mapping = {}  # Mapeo de ID de nodo OSM -> índice
        self.node_coordinates = {}  # Coordenadas (lat, lon) de cada nodo
        self.edge_lengths = {}  # Longitudes de aristas
        self.num_nodes = 0
        self.num_edges = 0
    
    def load_from_osm_json(self, filepath: str) -> bool:
        """
        Carga un grafo desde un archivo JSON de OpenStreetMap.
        
        Args:
            filepath: Ruta al archivo .osm.json
            
        Returns:
            True si la carga fue exitosa
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extraer nodos y aristas del JSON
            nodes = {}
            edges = []
            
            # Procesar elementos
            if 'elements' in data:
                for element in data['elements']:
                    if element['type'] == 'node':
                        node_id = element['id']
                        nodes[node_id] = {
                            'lat': element.get('lat', 0.0),
                            'lon': element.get('lon', 0.0),
                            'tags': element.get('tags', {})
                        }
                    elif element['type'] == 'way':
                        # Las vías contienen listas de nodos
                        way_nodes = element.get('nodes', [])
                        tags = element.get('tags', {})
                        
                        # Crear aristas entre nodos consecutivos
                        for i in range(len(way_nodes) - 1):
                            edges.append({
                                'source': way_nodes[i],
                                'target': way_nodes[i + 1],
                                'tags': tags
                            })
                        
                        # Si la vía no es unidireccional, agregar aristas inversas
                        if tags.get('oneway', 'no') != 'yes':
                            for i in range(len(way_nodes) - 1):
                                edges.append({
                                    'source': way_nodes[i + 1],
                                    'target': way_nodes[i],
                                    'tags': tags
                                })
            
            # Crear mapeos de nodos
            node_ids = sorted(nodes.keys())
            for idx, node_id in enumerate(node_ids):
                self.node_mapping[idx] = node_id
                self.reverse_mapping[node_id] = idx
                self.node_coordinates[idx] = (
                    nodes[node_id]['lat'],
                    nodes[node_id]['lon']
                )
            
            self.num_nodes = len(node_ids)
            
            # Crear matriz de adyacencia DISPERSA (sparse) usando lil_matrix para construcción eficiente
            print(f"📊 Creando matriz dispersa para {self.num_nodes} nodos...")
            self.adjacency_matrix = sparse.lil_matrix((self.num_nodes, self.num_nodes), dtype=np.float32)
            self.adjacency_list = {i: [] for i in range(self.num_nodes)}
            
            # Llenar matriz con distancias
            for edge in edges:
                source_id = edge['source']
                target_id = edge['target']
                
                if source_id in self.reverse_mapping and target_id in self.reverse_mapping:
                    source_idx = self.reverse_mapping[source_id]
                    target_idx = self.reverse_mapping[target_id]
                    
                    # Calcular distancia euclidiana (aproximación)
                    lat1, lon1 = self.node_coordinates[source_idx]
                    lat2, lon2 = self.node_coordinates[target_idx]
                    distance = self._haversine_distance(lat1, lon1, lat2, lon2)
                    
                    self.adjacency_matrix[source_idx, target_idx] = distance
                    self.adjacency_list[source_idx].append((target_idx, distance))
                    self.edge_lengths[(source_idx, target_idx)] = distance
                    self.num_edges += 1
            
            # Convertir a CSR (Compressed Sparse Row) para operaciones rápidas
            print(f"🔄 Convirtiendo a formato CSR...")
            self.adjacency_matrix = self.adjacency_matrix.tocsr()
            
            print(f"✅ Grafo cargado: {self.num_nodes} nodos, {self.num_edges} aristas")
            print(f"💾 Memoria de matriz dispersa: ~{self.adjacency_matrix.data.nbytes / 1024 / 1024:.2f} MB")
            return True
            
        except Exception as e:
            print(f"❌ Error al cargar grafo: {e}")
            return False
    
    @staticmethod
    def get_all_departamentos():
        """Retorna todos los departamentos disponibles."""
        return get_all_departamentos()
    
    @staticmethod
    def get_provincias(departamento_key: str):
        """Retorna provincias de un departamento."""
        return get_provincias_data(departamento_key)
    
    @staticmethod
    def get_distritos(departamento_key: str, provincia_key: str):
        """Retorna distritos de una provincia."""
        return get_distritos_data(departamento_key, provincia_key)
    
    @staticmethod
    def get_distrito_query(departamento_key: str, provincia_key: str, distrito_key: str):
        """Retorna la query de OSM para un distrito específico."""
        return get_distrito_query_data(departamento_key, provincia_key, distrito_key)
    
    @staticmethod
    def get_hospitales(departamento_key: str):
        """Retorna hospitales de un departamento."""
        return get_hospitales_region(departamento_key)
    
    def check_map_cache(self, distrito_key: str = None, region_key: str = None) -> dict:
        """Verifica si existe caché para una ubicación."""
        cache_key = distrito_key if distrito_key else region_key
        cache_dir = os.path.join(os.path.dirname(__file__), 'mapas')
        cache_file = os.path.join(cache_dir, f'{cache_key}_graph.pkl')
        
        exists = os.path.exists(cache_file)
        
        return {
            'exists': exists,
            'cache_key': cache_key,
            'cache_file': cache_file if exists else None,
            'message': f"✅ Mapa disponible: {cache_key}" if exists else f"❌ Mapa no disponible. Ejecutar: python cargar_desde_shapefile.py"
        }
    
    def download_region_from_osm(self, region_key: str, network_type: str = 'drive') -> bool:
        """
        Carga mapa desde archivo pickle local (ya procesado desde shapefile).
        YA NO descarga desde OSM - usa datos locales de Geofabrik.
        
        Args:
            region_key: Clave de región (ej: 'cusco')
            network_type: Ignorado (mantiene compatibilidad API)
            
        Returns:
            True si la carga fue exitosa
        """
        if region_key not in DEPARTAMENTOS_PERU:
            print(f"❌ Región '{region_key}' no encontrada.")
            print(f"Regiones disponibles: {', '.join(DEPARTAMENTOS_PERU.keys())}")
            return False
        
        # Archivo pickle ya procesado
        cache_dir = os.path.join(os.path.dirname(__file__), 'mapas')
        pkl_file = os.path.join(cache_dir, f'{region_key}_graph.pkl')
        
        if not os.path.exists(pkl_file):
            print(f"❌ Archivo no encontrado: {pkl_file}")
            print(f"💡 Ejecuta primero: python cargar_desde_shapefile.py")
            return False
        
        try:
            import pickle
            print(f"📦 Cargando {region_key} desde pickle local...")
            
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            self.graph = data['graph']
            self._process_networkx_graph(self.graph)
            
            print(f"✅ Cargado: {self.num_nodes:,} nodos, {self.num_edges:,} aristas")
            return True
            
        except Exception as e:
            print(f"❌ Error al cargar: {e}")
            return False
    
    def download_distrito_from_osm(self, query: str, distrito_key: str, network_type: str = 'drive') -> bool:
        """
        Descarga un distrito específico desde OpenStreetMap.
        
        Args:
            query: Query de OSM (ej: 'Wanchaq, Cusco, Peru')
            distrito_key: Identificador del distrito (ej: 'wanchaq')
            network_type: Tipo de red ('drive')
            
        Returns:
            True si la descarga fue exitosa
        """
        if not OSMNX_AVAILABLE:
            print("❌ OSMnx no está instalado. Instalar con: pip install osmnx")
            return False
        
        # Verificar caché
        cache_dir = os.path.join(os.path.dirname(__file__), 'mapas')
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f'{distrito_key}_{network_type}.json')
        
        if os.path.exists(cache_file):
            print(f"✅ Distrito '{distrito_key}' YA DESCARGADO. Cargando desde caché...")
            success = self._load_from_cache_file(cache_file)
            if success:
                print(f"✅ Carga completada: {self.num_nodes} nodos, {self.num_edges} aristas")
                return True
            else:
                print(f"⚠️  Error al cargar caché. Se descargará nuevamente...")
        
        try:
            print(f"📥 Descargando distrito: {query}")
            print(f"🚗 Red vehicular para ambulancias...")
            
            G = ox.graph_from_place(
                query,
                network_type=network_type,
                simplify=True,
                retain_all=False,
                truncate_by_edge=True
            )
            
            self.graph = G
            self._process_networkx_graph(G)
            
            print(f"💾 Guardando en caché: {cache_file}")
            self.save_to_json(cache_file)
            
            print(f"✅ Descarga completada: {self.num_nodes} nodos, {self.num_edges} aristas")
            return True
            
        except Exception as e:
            print(f"❌ Error al descargar distrito: {e}")
            return False
    
    def _process_networkx_graph(self, G):
        """Procesa un grafo de NetworkX a matriz de adyacencia (dispersa para grafos grandes)."""
        # Obtener lista de nodos
        nodes = list(G.nodes())
        self.num_nodes = len(nodes)
        
        # Crear mapeos
        for idx, node_id in enumerate(nodes):
            self.node_mapping[idx] = node_id
            self.reverse_mapping[node_id] = idx
            
            # Obtener coordenadas (soportar ambos formatos: OSM y MTC)
            node_data = G.nodes[node_id]
            # OSM usa 'y','x' | MTC usa 'lat','lon'
            lat = node_data.get('y', node_data.get('lat', 0.0))
            lon = node_data.get('x', node_data.get('lon', 0.0))
            self.node_coordinates[idx] = (lat, lon)
        
        # DECISIÓN: usar matriz dispersa si el grafo es grande (>10k nodos)
        use_sparse = self.num_nodes > 10000
        
        if use_sparse:
            print(f"📊 Creando matriz dispersa para {self.num_nodes:,} nodos...")
            # Usar matriz dispersa (lil_matrix para construcción eficiente)
            self.adjacency_matrix = sparse.lil_matrix((self.num_nodes, self.num_nodes), dtype=np.float32)
            self.adjacency_list = {i: [] for i in range(self.num_nodes)}
            
            # Llenar con aristas
            for u, v, data in G.edges(data=True):
                u_idx = self.reverse_mapping[u]
                v_idx = self.reverse_mapping[v]
                
                # Usar la longitud de la arista si está disponible
                length = data.get('length', 1.0)
                
                self.adjacency_matrix[u_idx, v_idx] = length
                self.adjacency_list[u_idx].append((v_idx, length))
                self.edge_lengths[(u_idx, v_idx)] = length
                self.num_edges += 1
            
            # Convertir a CSR (Compressed Sparse Row) para operaciones rápidas
            print(f"🔄 Convirtiendo a formato CSR...")
            self.adjacency_matrix = self.adjacency_matrix.tocsr()
            print(f"💾 Memoria de matriz dispersa: ~{self.adjacency_matrix.data.nbytes / 1024 / 1024:.2f} MB")
        else:
            # Matriz densa para grafos pequeños
            print(f"📊 Creando matriz densa para {self.num_nodes:,} nodos...")
            self.adjacency_matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
            
            # Llenar con aristas
            for u, v, data in G.edges(data=True):
                u_idx = self.reverse_mapping[u]
                v_idx = self.reverse_mapping[v]
                
                # Usar la longitud de la arista si está disponible
                length = data.get('length', 1.0)
                
                self.adjacency_matrix[u_idx, v_idx] = length
                self.edge_lengths[(u_idx, v_idx)] = length
                self.num_edges += 1
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calcula la distancia haversine entre dos coordenadas (en metros).
        """
        from math import radians, cos, sin, asin, sqrt
        
        # Radio de la Tierra en metros
        R = 6371000
        
        # Convertir a radianes
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Diferencias
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        # Fórmula haversine
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        return R * c
    
    def get_adjacency_matrix(self):
        """Retorna la matriz de adyacencia del grafo (dispersa o densa según tamaño)."""
        # Si la matriz es dispersa (sparse), devolverla como está
        if sparse.issparse(self.adjacency_matrix):
            return self.adjacency_matrix
        return self.adjacency_matrix
    
    def get_adjacency_list(self) -> Dict:
        """Retorna la lista de adyacencia del grafo."""
        return self.adjacency_list
    
    def get_edge_weight(self, from_node: int, to_node: int) -> float:
        """Retorna el peso de una arista entre dos nodos."""
        # Primero intentar desde edge_lengths
        if (from_node, to_node) in self.edge_lengths:
            return self.edge_lengths[(from_node, to_node)]
        
        # Si no, buscar en la matriz de adyacencia
        if sparse.issparse(self.adjacency_matrix):
            # Para matriz dispersa
            return self.adjacency_matrix[from_node, to_node]
        else:
            # Para matriz densa
            return self.adjacency_matrix[from_node, to_node]
    
    def get_node_info(self, node_idx: int) -> Dict:
        """Obtiene información de un nodo por su índice."""
        if node_idx not in self.node_mapping:
            return None
        
        return {
            'index': node_idx,
            'osm_id': self.node_mapping[node_idx],
            'coordinates': self.node_coordinates.get(node_idx, (0, 0)),
            'lat': self.node_coordinates.get(node_idx, (0, 0))[0],
            'lon': self.node_coordinates.get(node_idx, (0, 0))[1]
        }
    
    def find_nearest_node(self, lat: float, lon: float) -> int:
        """Encuentra el nodo más cercano a una coordenada usando búsqueda optimizada."""
        if not self.node_coordinates:
            print("⚠️ No hay nodos cargados en el grafo")
            return 0
        
        min_dist = float('inf')
        nearest_idx = 0
        
        # Búsqueda optimizada con filtro espacial más agresivo
        # Para grafos grandes (>100k nodos), usar muestreo
        coords_items = list(self.node_coordinates.items())
        
        # Si el grafo es muy grande, primero filtrar por bbox
        if len(coords_items) > 100000:
            # Filtro agresivo: bbox de 0.05 grados (~5.5km)
            filtered = [
                (idx, nlat, nlon) 
                for idx, (nlat, nlon) in coords_items
                if abs(nlat - lat) < 0.05 and abs(nlon - lon) < 0.05
            ]
            
            if not filtered:
                # Si no hay nodos cercanos, ampliar búsqueda
                filtered = [
                    (idx, nlat, nlon) 
                    for idx, (nlat, nlon) in coords_items
                    if abs(nlat - lat) < 0.2 and abs(nlon - lon) < 0.2
                ]
            
            print(f"🔍 Buscando en {len(filtered):,} nodos filtrados de {len(coords_items):,}")
        else:
            # Filtro estándar para grafos pequeños
            filtered = [
                (idx, nlat, nlon) 
                for idx, (nlat, nlon) in coords_items
                if abs(nlat - lat) < 0.1 and abs(nlon - lon) < 0.1
            ]
        
        # Buscar el más cercano en los nodos filtrados
        for idx, node_lat, node_lon in filtered:
            dist = self._haversine_distance(lat, lon, node_lat, node_lon)
            
            if dist < min_dist:
                min_dist = dist
                nearest_idx = idx
        
        # Si no encontramos nada en el filtro, buscar en todos
        if min_dist == float('inf'):
            print(f"⚠️ No se encontraron nodos cercanos, buscando en todo el grafo...")
            for idx, (node_lat, node_lon) in coords_items[:10000]:  # Limitar a primeros 10k
                dist = self._haversine_distance(lat, lon, node_lat, node_lon)
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = idx
        
        # Validar resultado
        if min_dist > 10000:  # > 10km es sospechoso
            print(f"⚠️ Nodo más cercano está a {min_dist/1000:.1f} km")
            print(f"   Buscado: ({lat:.6f}, {lon:.6f})")
            if nearest_idx in self.node_coordinates:
                print(f"   Encontrado en nodo {nearest_idx}: {self.node_coordinates[nearest_idx]}")
        elif min_dist < 1000:  # < 1km está bien
            print(f"✅ Nodo encontrado a {min_dist:.0f} metros")
        
        return nearest_idx
    
    def save_to_json(self, filepath: str) -> bool:
        """Guarda el grafo en formato JSON."""
        try:
            data = {
                'num_nodes': self.num_nodes,
                'num_edges': self.num_edges,
                'node_mapping': {str(k): v for k, v in self.node_mapping.items()},
                'node_coordinates': {
                    str(k): {'lat': v[0], 'lon': v[1]} 
                    for k, v in self.node_coordinates.items()
                },
                'adjacency_list': {}
            }
            
            # Crear lista de adyacencia para reducir tamaño
            for i in range(self.num_nodes):
                neighbors = []
                for j in range(self.num_nodes):
                    if self.adjacency_matrix[i, j] > 0:
                        neighbors.append({
                            'target': j,
                            'weight': float(self.adjacency_matrix[i, j])
                        })
                if neighbors:
                    data['adjacency_list'][str(i)] = neighbors
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            print(f"✅ Grafo guardado en {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error al guardar grafo: {e}")
            return False
    
    def _load_from_cache_file(self, filepath: str) -> bool:
        """Carga un grafo desde un archivo de caché JSON."""
        try:
            print(f"📂 Validando archivo de caché...")
            
            # Verificar que el archivo no esté vacío o corrupto
            file_size = os.path.getsize(filepath)
            if file_size < 100:  # Archivo demasiado pequeño = corrupto
                print(f"⚠️  Archivo de caché corrupto o vacío ({file_size} bytes). Eliminando...")
                os.remove(filepath)
                return False
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validar que tenga los campos necesarios
            required_fields = ['num_nodes', 'num_edges', 'node_mapping', 'adjacency_list']
            for field in required_fields:
                if field not in data:
                    print(f"⚠️  Caché incompleto (falta {field}). Eliminando...")
                    os.remove(filepath)
                    return False
            
            # Validar que tenga datos válidos
            if data['num_nodes'] == 0 or data['num_edges'] == 0:
                print(f"⚠️  Caché sin datos válidos. Eliminando...")
                os.remove(filepath)
                return False
            
            self.num_nodes = data['num_nodes']
            self.num_edges = data['num_edges']
            
            # Reconstruir mapeos
            self.node_mapping = {int(k): v for k, v in data['node_mapping'].items()}
            self.reverse_mapping = {v: int(k) for k, v in data['node_mapping'].items()}
            self.node_coordinates = {
                int(k): (v['lat'], v['lon']) 
                for k, v in data['node_coordinates'].items()
            }
            
            # Reconstruir matriz de adyacencia
            self.adjacency_matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
            
            for source_str, neighbors in data['adjacency_list'].items():
                source = int(source_str)
                for neighbor in neighbors:
                    target = neighbor['target']
                    weight = neighbor['weight']
                    self.adjacency_matrix[source, target] = weight
                    self.edge_lengths[(source, target)] = weight
            
            print(f"✅ Caché validado correctamente")
            return True
            
        except json.JSONDecodeError as e:
            print(f"❌ Archivo de caché corrupto (JSON inválido): {e}")
            print(f"🗑️  Eliminando archivo corrupto...")
            if os.path.exists(filepath):
                os.remove(filepath)
            return False
        except Exception as e:
            print(f"❌ Error al cargar desde caché: {e}")
            print(f"🗑️  Eliminando archivo problemático...")
            if os.path.exists(filepath):
                os.remove(filepath)
            return False
    
    def load_from_mtc(self, region_key: str = 'cusco', incluir_vecinal: bool = True) -> bool:
        """
        Carga red vial oficial del MTC para una región.
        Usa caché si existe, sino descarga.
        
        Args:
            region_key: Región a cargar (ej: 'cusco')
            incluir_vecinal: Si incluir caminos vecinales
            
        Returns:
            True si la carga fue exitosa
        """
        try:
            # Verificar si existe el grafo procesado en caché
            cache_dir = os.path.join(os.path.dirname(__file__), 'mapas')
            pkl_file = os.path.join(cache_dir, f'{region_key}_mtc_graph.pkl')
            pkl_urban_file = os.path.join(cache_dir, f'{region_key}_mtc_urban.pkl')
            
            # Intentar cargar versión urbana (componente conectado principal)
            if os.path.exists(pkl_urban_file):
                print(f"📦 Cargando red vial URBANA desde caché: {pkl_urban_file}")
                print(f"⏳ Cargando grafo urbano conectado...")
                
                import pickle
                import time
                start = time.time()
                
                with open(pkl_urban_file, 'rb') as f:
                    G = pickle.load(f)
                
                print(f"✅ Grafo urbano cargado en {time.time() - start:.1f}s")
                print(f"🔄 Procesando matriz dispersa...")
                
                self.graph = G
                self._process_networkx_graph(G)
                
                print(f"✅ Red vial URBANA lista: {self.num_nodes:,} nodos, {self.num_edges:,} aristas")
                return True
            
            # Si no existe urbano, cargar completo y extraer componente principal
            if os.path.exists(pkl_file):
                print(f"📦 Cargando red vial MTC completa desde caché: {pkl_file}")
                print(f"⏳ Esto puede tardar 10-15 segundos para grafo grande...")
                
                import pickle
                import networkx as nx
                import time
                start = time.time()
                
                with open(pkl_file, 'rb') as f:
                    G_full = pickle.load(f)
                
                print(f"✅ Grafo completo cargado en {time.time() - start:.1f}s")
                print(f"🔍 Extrayendo componente conectado principal (zona urbana)...")
                
                # Extraer el componente conexo más grande
                if G_full.is_directed():
                    # Para grafos dirigidos, usar componente débilmente conectado
                    components = list(nx.weakly_connected_components(G_full))
                else:
                    components = list(nx.connected_components(G_full))
                
                # Obtener el componente más grande
                largest_component = max(components, key=len)
                G = G_full.subgraph(largest_component).copy()
                
                print(f"✅ Componente principal: {len(G.nodes):,} nodos ({len(G.nodes)/len(G_full.nodes)*100:.1f}% del total)")
                print(f"💾 Guardando versión urbana para próximas cargas...")
                
                with open(pkl_urban_file, 'wb') as f:
                    pickle.dump(G, f)
                
                self.graph = G
                self._process_networkx_graph(G)
                
                print(f"✅ Red vial URBANA lista: {self.num_nodes:,} nodos, {self.num_edges:,} aristas")
                return True
            
            # Si no existe en caché, intentar descargar
            if not MTC_MINSA_AVAILABLE:
                print("❌ Módulos MTC no disponibles y no hay caché")
                print("💡 Ejecuta primero: python descargar_mtc.py")
                return False
            
            print(f"🏛️  Descargando red vial oficial del MTC para {region_key}...")
            
            downloader = MTCDataDownloader()
            
            # Descargar red vial
            gdf = downloader.download_red_vial_cusco(
                incluir_vecinal=incluir_vecinal,
                cache=True
            )
            
            if gdf is None:
                return False
            
            # Convertir a grafo NetworkX
            G, metadata = downloader.convert_to_graph(gdf)
            
            # Guardar grafo para próxima vez
            print(f"💾 Guardando grafo procesado en caché...")
            import pickle
            with open(pkl_file, 'wb') as f:
                pickle.dump(G, f)
            
            # Procesar el grafo
            self.graph = G
            self._process_networkx_graph(G)
            
            print(f"✅ Red vial MTC cargada: {self.num_nodes:,} nodos, {self.num_edges:,} aristas")
            return True
            
        except Exception as e:
            print(f"❌ Error al cargar red vial MTC: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_hospitals_minsa(self, region_key: str = 'cusco', solo_hospitales: bool = True) -> List[Dict]:
        """
        Carga datos de hospitales oficiales del MINSA.
        
        Args:
            region_key: Región (ej: 'cusco')
            solo_hospitales: Si solo incluir hospitales (no centros de salud)
            
        Returns:
            Lista de hospitales con información completa
        """
        if not MTC_MINSA_AVAILABLE:
            print("❌ Módulos MINSA no disponibles")
            # Fallback a datos estáticos
            return get_hospitales_region(region_key)
        
        try:
            print(f"🏥 Cargando establecimientos de salud MINSA para {region_key}...")
            
            downloader = MINSADataDownloader()
            
            # Descargar establecimientos
            df = downloader.download_establecimientos_cusco(
                solo_hospitales=solo_hospitales,
                cache=True
            )
            
            if df is None:
                print("⚠️  Usando datos estáticos de hospitales")
                return get_hospitales_region(region_key)
            
            # Convertir a formato para el grafo
            hospitales = downloader.get_hospitales_para_grafo(df)
            
            print(f"✅ Cargados {len(hospitales)} establecimientos del MINSA")
            return hospitales
            
        except Exception as e:
            print(f"⚠️  Error al cargar MINSA: {e}")
            print("⚠️  Usando datos estáticos de hospitales")
            return get_hospitales_region(region_key)
    
    def find_nearest_hospitals(self, lat: float, lon: float, region_key: str, max_hospitals: int = 5) -> List[Dict]:
        """
        Encuentra los hospitales más cercanos a una ubicación.
        Usa datos de hospitales.py automáticamente.
        
        Args:
            lat: Latitud de la ubicación
            lon: Longitud de la ubicación
            region_key: Región donde buscar hospitales
            max_hospitals: Número máximo de hospitales a retornar
            
        Returns:
            Lista de hospitales con nodo más cercano y distancia
        """
        # Obtener hospitales de la región desde hospitales.py
        hospitales = get_hospitales_region(region_key)
        
        if not hospitales:
            print(f"ℹ️  No hay hospitales registrados para {region_key}")
            return []
        
        hospitales_cercanos = []
        
        for hospital in hospitales:
            # Encontrar nodo más cercano al hospital en el mapa cargado
            hospital_node = self.find_nearest_node(hospital['lat'], hospital['lon'])
            
            # Calcular distancia desde ubicación del usuario
            user_node = self.find_nearest_node(lat, lon)
            
            # Distancia haversine directa (línea recta en km)
            dist_directa = self._haversine_distance(
                lat, lon, 
                hospital['lat'], hospital['lon']
            )
            
            hospitales_cercanos.append({
                'name': hospital['name'],
                'tipo': hospital.get('tipo', 'General'),
                'nivel': hospital.get('nivel', 'II-1'),
                'lat': hospital['lat'],
                'lon': hospital['lon'],
                'node_index': hospital_node,
                'distance_direct_m': dist_directa,
                'distance_direct_km': dist_directa / 1000
            })
        
        # Ordenar por distancia
        hospitales_cercanos.sort(key=lambda x: x['distance_direct_m'])
        
        print(f"🏥 Encontrados {len(hospitales_cercanos)} hospitales en {region_key}")
        return hospitales_cercanos[:max_hospitals]
