"""
Módulo para descargar y procesar datos oficiales de red vial del MTC Perú.
Fuentes:
- Portal de Datos Abiertos del MTC
- GeoPerú (Infraestructura Nacional de Datos Espaciales - IDEP)
- IDE MTC (Infraestructura de Datos Espaciales del MTC)
"""
import requests
import pandas as pd
import geopandas as gpd
import networkx as nx
import pickle
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import time

class MTCDataDownloader:
    """Descargador de datos oficiales de red vial del MTC."""
    
    # URLs de servicios WFS del MTC
    MTC_WFS_BASE = "https://portal.mtc.gob.pe/geoservicios/services"
    
    # URLs alternativas de datos abiertos
    GEOPERU_BASE = "https://www.gob.pe/geoperu"
    DATOSABIERTOS_MTC = "https://portal.mtc.gob.pe/estadisticas/datos_abiertos.html"
    
    # Datasets conocidos (estos IDs pueden variar)
    DATASETS = {
        'red_vial_nacional': {
            'name': 'Red Vial Nacional',
            'description': 'Carreteras nacionales del Sistema Nacional de Carreteras (SINAC)',
            'layer': 'RedVialNacional'
        },
        'red_vial_departamental': {
            'name': 'Red Vial Departamental', 
            'description': 'Carreteras departamentales',
            'layer': 'RedVialDepartamental'
        },
        'red_vial_vecinal': {
            'name': 'Red Vial Vecinal',
            'description': 'Carreteras vecinales y rurales',
            'layer': 'RedVialVecinal'
        }
    }
    
    def __init__(self, cache_dir: str = "backend/mapas/mtc"):
        """
        Args:
            cache_dir: Directorio para cachear descargas
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def download_red_vial_cusco(
        self, 
        incluir_vecinal: bool = True,
        cache: bool = True
    ) -> Optional[gpd.GeoDataFrame]:
        """
        Descarga la red vial del departamento de Cusco.
        
        Args:
            incluir_vecinal: Si incluir caminos vecinales (más completo pero más lento)
            cache: Si usar caché local
            
        Returns:
            GeoDataFrame con la red vial o None si falla
        """
        cache_file = self.cache_dir / f"cusco_vial_{'completa' if incluir_vecinal else 'principal'}.geojson"
        
        # Intentar cargar desde caché
        if cache and cache_file.exists():
            print(f"📦 Cargando desde caché: {cache_file}")
            try:
                return gpd.read_file(cache_file)
            except Exception as e:
                print(f"⚠️  Error al cargar caché: {e}")
        
        print("🌐 Descargando red vial de Cusco desde fuentes oficiales...")
        
        # Estrategia 1: Intentar servicio WFS del MTC
        gdf = self._download_from_wfs_cusco(incluir_vecinal)
        
        # Estrategia 2: Si WFS falla, descargar shapefile completo y filtrar
        if gdf is None:
            gdf = self._download_shapefile_and_filter_cusco(incluir_vecinal)
        
        # Estrategia 3: Usar datos locales de Geofabrik como fallback
        if gdf is None:
            print("⚠️  No se pudo descargar de MTC, usando Geofabrik como respaldo")
            gdf = self._load_from_geofabrik_cusco()
        
        # Guardar en caché
        if gdf is not None and cache:
            print(f"💾 Guardando en caché: {cache_file}")
            gdf.to_file(cache_file, driver='GeoJSON')
        
        return gdf
    
    def _download_from_wfs_cusco(self, incluir_vecinal: bool) -> Optional[gpd.GeoDataFrame]:
        """Intenta descargar desde servicio WFS con filtro de Cusco."""
        try:
            # Bounding box de Cusco
            bbox = "-73.5,-15.0,-70.5,-11.5"  # minx,miny,maxx,maxy
            
            layers = ['red_vial_nacional', 'red_vial_departamental']
            if incluir_vecinal:
                layers.append('red_vial_vecinal')
            
            gdfs = []
            
            for layer_key in layers:
                layer_info = self.DATASETS[layer_key]
                print(f"  📍 Descargando {layer_info['name']}...")
                
                # Construir URL WFS (esto es un ejemplo, la URL real puede variar)
                wfs_url = (
                    f"{self.MTC_WFS_BASE}/wfs?"
                    f"service=WFS&version=2.0.0&request=GetFeature&"
                    f"typeName={layer_info['layer']}&"
                    f"bbox={bbox}&"
                    f"outputFormat=json"
                )
                
                try:
                    response = requests.get(wfs_url, timeout=30)
                    if response.status_code == 200:
                        gdf_layer = gpd.read_file(response.text)
                        gdfs.append(gdf_layer)
                        print(f"    ✅ {len(gdf_layer)} segmentos")
                except Exception as e:
                    print(f"    ⚠️  Error: {e}")
            
            if gdfs:
                # Combinar todas las capas
                combined = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
                print(f"✅ Total descargado: {len(combined)} segmentos")
                return combined
            
        except Exception as e:
            print(f"⚠️  Error en descarga WFS: {e}")
        
        return None
    
    def _download_shapefile_and_filter_cusco(self, incluir_vecinal: bool) -> Optional[gpd.GeoDataFrame]:
        """
        Descarga shapefile completo del Perú y filtra Cusco.
        Nota: Este método requiere URLs reales de descarga directa.
        """
        try:
            print("📥 Descargando shapefile nacional completo...")
            
            # URLs de ejemplo (necesitan ser actualizadas con URLs reales del MTC)
            # Estas URLs son placeholders y deben ser reemplazadas
            urls = {
                'nacional': 'https://datos.gob.pe/dataset/red-vial-nacional.zip',
                'departamental': 'https://datos.gob.pe/dataset/red-vial-departamental.zip',
            }
            
            if incluir_vecinal:
                urls['vecinal'] = 'https://datos.gob.pe/dataset/red-vial-vecinal.zip'
            
            # Por ahora, retornar None para usar el fallback
            # TODO: Implementar descarga cuando se tengan las URLs correctas
            print("⚠️  URLs de descarga directa no configuradas")
            return None
            
        except Exception as e:
            print(f"⚠️  Error en descarga de shapefile: {e}")
            return None
    
    def _load_from_geofabrik_cusco(self) -> Optional[gpd.GeoDataFrame]:
        """Carga datos de Geofabrik como fallback."""
        try:
            shapefile_path = Path(__file__).parent / 'mapas' / 'peru-251217-free.shp' / 'gis_osm_roads_free_1.shp'
            
            if not shapefile_path.exists():
                print(f"❌ Shapefile no encontrado: {shapefile_path}")
                return None
            
            print(f"📂 Cargando desde Geofabrik: {shapefile_path}")
            roads = gpd.read_file(shapefile_path)
            
            # Filtrar Cusco por bbox
            cusco_roads = roads.cx[-73.5:-70.5, -15.0:-11.5]
            
            # Filtrar solo vías transitables
            valid_types = [
                'motorway', 'trunk', 'primary', 'secondary', 'tertiary',
                'unclassified', 'residential', 'motorway_link', 'trunk_link',
                'primary_link', 'secondary_link', 'tertiary_link', 
                'living_street', 'service', 'road'
            ]
            
            if 'fclass' in cusco_roads.columns:
                cusco_roads = cusco_roads[cusco_roads['fclass'].isin(valid_types)]
            
            print(f"✅ Cargado desde Geofabrik: {len(cusco_roads)} segmentos")
            return cusco_roads
            
        except Exception as e:
            print(f"❌ Error cargando Geofabrik: {e}")
            return None
    
    def convert_to_graph(self, gdf: gpd.GeoDataFrame) -> Tuple[nx.MultiDiGraph, Dict]:
        """
        Convierte GeoDataFrame de red vial a grafo NetworkX.
        
        Args:
            gdf: GeoDataFrame con geometrías de líneas
            
        Returns:
            (grafo, metadata) tupla con grafo NetworkX y metadatos
        """
        print("🔄 Convirtiendo red vial a grafo...")
        
        G = nx.MultiDiGraph()
        node_id_counter = 0
        node_coords_to_id = {}
        
        for idx, row in gdf.iterrows():
            try:
                if row.geometry.geom_type not in ['LineString', 'MultiLineString']:
                    continue
                
                # Manejar MultiLineString
                if row.geometry.geom_type == 'MultiLineString':
                    lines = list(row.geometry.geoms)
                else:
                    lines = [row.geometry]
                
                for line in lines:
                    coords = list(line.coords)
                    
                    for i in range(len(coords) - 1):
                        # Crear/obtener nodos
                        start_coord = (round(coords[i][1], 6), round(coords[i][0], 6))
                        end_coord = (round(coords[i+1][1], 6), round(coords[i+1][0], 6))
                        
                        if start_coord not in node_coords_to_id:
                            node_coords_to_id[start_coord] = node_id_counter
                            G.add_node(node_id_counter, lat=start_coord[0], lon=start_coord[1])
                            node_id_counter += 1
                        
                        if end_coord not in node_coords_to_id:
                            node_coords_to_id[end_coord] = node_id_counter
                            G.add_node(node_id_counter, lat=end_coord[0], lon=end_coord[1])
                            node_id_counter += 1
                        
                        # Calcular distancia Haversine
                        distance = self._haversine_distance(
                            start_coord[0], start_coord[1],
                            end_coord[0], end_coord[1]
                        )
                        
                        # Agregar arista con atributos del MTC
                        start_id = node_coords_to_id[start_coord]
                        end_id = node_coords_to_id[end_coord]
                        
                        edge_attrs = {
                            'length': distance,
                            'highway': row.get('fclass', 'unknown'),
                            'name': row.get('name', ''),
                            'ref': row.get('ref', ''),  # Código de ruta (ej: PE-3S)
                        }
                        
                        G.add_edge(start_id, end_id, **edge_attrs)
                        
            except Exception as e:
                continue
        
        metadata = {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'source': 'MTC',
            'region': 'Cusco'
        }
        
        print(f"✅ Grafo creado: {metadata['nodes']:,} nodos, {metadata['edges']:,} aristas")
        return G, metadata
    
    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calcula distancia Haversine entre dos puntos (en metros)."""
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371000  # Radio de la Tierra en metros
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    def save_graph(self, G: nx.MultiDiGraph, filepath: str):
        """Guarda grafo en formato pickle."""
        with open(filepath, 'wb') as f:
            pickle.dump(G, f)
        print(f"💾 Grafo guardado: {filepath}")


def main():
    """Script de prueba para descargar red vial de Cusco."""
    print("=" * 60)
    print("🏛️  DESCARGA DE RED VIAL OFICIAL - MTC PERÚ")
    print("=" * 60)
    
    start = time.time()
    
    downloader = MTCDataDownloader()
    
    # Descargar red vial de Cusco
    gdf = downloader.download_red_vial_cusco(incluir_vecinal=True)
    
    if gdf is None:
        print("❌ No se pudo obtener datos de red vial")
        return
    
    # Convertir a grafo
    G, metadata = downloader.convert_to_graph(gdf)
    
    # Guardar
    output_path = "backend/mapas/cusco_mtc_graph.pkl"
    downloader.save_graph(G, output_path)
    
    elapsed = time.time() - start
    
    print("\n" + "=" * 60)
    print(f"✅ COMPLETADO EN {elapsed:.1f} segundos")
    print(f"📊 Nodos: {metadata['nodes']:,}")
    print(f"🔗 Aristas: {metadata['edges']:,}")
    print(f"💾 Archivo: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
