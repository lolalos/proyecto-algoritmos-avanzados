"""
Cargar red vial de Cusco desde shapefile de Geofabrik
MUCHO MÁS RÁPIDO que OSMnx (30 segundos vs 20 minutos)
"""
import geopandas as gpd
import networkx as nx
import pickle
import time
from pathlib import Path

print("🚀 Cargando red vial de Cusco desde shapefile local...")
start_time = time.time()

# Ruta al shapefile de carreteras
shapefile_path = Path(__file__).parent / 'mapas' / 'peru-251217-free.shp' / 'gis_osm_roads_free_1.shp'

print(f"📂 Leyendo: {shapefile_path}")

# 1. Cargar shapefile completo
roads = gpd.read_file(shapefile_path)
print(f"✅ Cargado: {len(roads):,} segmentos de carretera en todo Perú")

# 2. Filtrar solo región de Cusco
# Cusco está aproximadamente entre:
# Lat: -11.5 a -15.0
# Lon: -70.5 a -73.5
print("🔍 Filtrando solo región de Cusco...")

cusco_bounds = {
    'min_lon': -73.5,
    'max_lon': -70.5,
    'min_lat': -15.0,
    'max_lat': -11.5
}

# Filtrar por bounding box
cusco_roads = roads.cx[cusco_bounds['min_lon']:cusco_bounds['max_lon'], 
                        cusco_bounds['min_lat']:cusco_bounds['max_lat']]

print(f"✅ Carreteras en Cusco: {len(cusco_roads):,} segmentos")

# 3. Filtrar solo vías transitables por vehículos
# Tipos válidos para ambulancias/carros
valid_road_types = [
    'motorway', 'trunk', 'primary', 'secondary', 'tertiary',
    'unclassified', 'residential', 'motorway_link', 'trunk_link',
    'primary_link', 'secondary_link', 'tertiary_link', 'living_street',
    'service', 'road'
]

if 'fclass' in cusco_roads.columns:
    cusco_roads = cusco_roads[cusco_roads['fclass'].isin(valid_road_types)]
    print(f"✅ Después de filtrar por tipo: {len(cusco_roads):,} segmentos")

# 4. Convertir a grafo NetworkX
print("🔄 Construyendo grafo de red vial...")
G = nx.MultiDiGraph()

node_id_counter = 0
node_coords_to_id = {}  # (lat, lon) -> node_id

for idx, road in cusco_roads.iterrows():
    try:
        coords = list(road.geometry.coords)
        
        # Agregar nodos y aristas
        for i in range(len(coords) - 1):
            # Nodo inicio
            start_coord = (round(coords[i][1], 6), round(coords[i][0], 6))  # (lat, lon)
            if start_coord not in node_coords_to_id:
                node_coords_to_id[start_coord] = node_id_counter
                G.add_node(node_id_counter, lat=start_coord[0], lon=start_coord[1])
                node_id_counter += 1
            
            # Nodo fin
            end_coord = (round(coords[i+1][1], 6), round(coords[i+1][0], 6))  # (lat, lon)
            if end_coord not in node_coords_to_id:
                node_coords_to_id[end_coord] = node_id_counter
                G.add_node(node_id_counter, lat=end_coord[0], lon=end_coord[1])
                node_id_counter += 1
            
            # Calcular distancia (en metros aproximadamente)
            from math import radians, sin, cos, sqrt, atan2
            lat1, lon1 = radians(start_coord[0]), radians(start_coord[1])
            lat2, lon2 = radians(end_coord[0]), radians(end_coord[1])
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            distance = 6371000 * c  # Radio de la Tierra en metros
            
            # Agregar arista
            start_id = node_coords_to_id[start_coord]
            end_id = node_coords_to_id[end_coord]
            
            G.add_edge(start_id, end_id, 
                      length=distance,
                      highway=road.get('fclass', 'unknown'),
                      name=road.get('name', ''))
            
    except Exception as e:
        continue

print(f"✅ Grafo construido: {G.number_of_nodes():,} nodos, {G.number_of_edges():,} aristas")

# 5. Guardar grafo procesado
output_file = Path(__file__).parent / 'mapas' / 'cusco_graph.pkl'
print(f"💾 Guardando grafo en: {output_file}")

graph_data = {
    'graph': G,
    'num_nodes': G.number_of_nodes(),
    'num_edges': G.number_of_edges(),
    'bounds': cusco_bounds,
    'source': 'geofabrik_shapefile'
}

with open(output_file, 'wb') as f:
    pickle.dump(graph_data, f)

elapsed = time.time() - start_time

print(f"\n✅ ¡COMPLETADO!")
print(f"⏱️  Tiempo total: {elapsed:.2f} segundos")
print(f"📊 Nodos: {G.number_of_nodes():,}")
print(f"🔗 Aristas: {G.number_of_edges():,}")
print(f"💾 Archivo: {output_file}")
print(f"\n🎯 Ahora reinicia el servidor y carga el mapa desde la interfaz web")
