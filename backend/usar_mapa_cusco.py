"""
Script para usar el mapa de Cusco con algoritmos de caminos más cortos.
Demuestra el uso de la red vial oficial con hospitales del MINSA.
"""
import json
import pickle
from pathlib import Path
from graph import UrbanGraph
from algorithms.dijkstra import DijkstraAlgorithm
import time

print("=" * 70)
print("🚑 SISTEMA DE RUTAS ÓPTIMAS DE AMBULANCIAS - CUSCO")
print("=" * 70)
print()

# 1. Cargar el grafo de Cusco (ya procesado)
print("📦 Cargando grafo de Cusco...")
graph = UrbanGraph()

# Cargar desde area.osm.json (más manejable: ~1.8M nodos con matriz dispersa)
area_json = Path(__file__).parent.parent / 'area.osm.json'

if area_json.exists():
    print(f"✅ Cargando desde area.osm.json (con matriz dispersa)")
    success = graph.load_from_osm_json(str(area_json))
    if not success:
        print("❌ Error al cargar el grafo")
        exit(1)
else:
    print("❌ No se encontró area.osm.json")
    print("💡 Asegúrate de que area.osm.json esté en la raíz del proyecto")
    exit(1)

print(f"📊 Grafo cargado: {graph.num_nodes:,} nodos, {graph.num_edges:,} aristas")
print()

# 2. Cargar hospitales
print("🏥 Cargando hospitales de Cusco...")
hospitales_file = Path(__file__).parent / 'mapas' / 'cusco_hospitales_grafo.json'

if hospitales_file.exists():
    with open(hospitales_file, 'r', encoding='utf-8') as f:
        hospitales = json.load(f)
    print(f"✅ Cargados {len(hospitales)} hospitales")
else:
    print("⚠️  Usando hospitales por defecto")
    hospitales = [
        {
            'id': 'MINSA001',
            'nombre': 'Hospital Regional del Cusco',
            'lat': -13.5226,
            'lon': -71.9673,
            'emergencia': True,
            'uci': True
        },
        {
            'id': 'ESSALUD001',
            'nombre': 'Hospital Adolfo Guevara Velasco',
            'lat': -13.5188,
            'lon': -71.9644,
            'emergencia': True,
            'uci': True
        }
    ]

print()
print("🏥 Hospitales disponibles:")
print("-" * 70)
for i, h in enumerate(hospitales, 1):
    print(f"{i}. {h['nombre']}")
    print(f"   📍 Lat: {h['lat']}, Lon: {h['lon']}")
    if h.get('uci'):
        print(f"   🏥 UCI: Sí | Emergencia: {'Sí' if h.get('emergencia') else 'No'}")
    print()

# 3. Encontrar nodos de hospitales en el grafo
print("🔍 Mapeando hospitales a nodos del grafo...")
hospitales_nodos = []

for hospital in hospitales:
    nodo = graph.find_nearest_node(hospital['lat'], hospital['lon'])
    nodo_info = graph.get_node_info(nodo)
    
    hospitales_nodos.append({
        'hospital': hospital['nombre'],
        'nodo': nodo,
        'lat': nodo_info['lat'],
        'lon': nodo_info['lon'],
        'original_lat': hospital['lat'],
        'original_lon': hospital['lon']
    })
    
    print(f"  ✅ {hospital['nombre'][:40]:40} → Nodo {nodo:,}")

print()

# 4. Ejemplo: Usuario en Plaza de Armas de Cusco
print("=" * 70)
print("📍 EJEMPLO: Usuario en Plaza de Armas de Cusco")
print("=" * 70)
print()

usuario_lat = -13.5164  # Plaza de Armas
usuario_lon = -71.9784
usuario_nodo = graph.find_nearest_node(usuario_lat, usuario_lon)

print(f"Usuario ubicado en: ({usuario_lat}, {usuario_lon})")
print(f"Nodo más cercano: {usuario_nodo:,}")
print()

# 5. Ejecutar Dijkstra desde cada hospital
print("🚀 Calculando rutas desde hospitales con Dijkstra...")
print("-" * 70)

dijkstra = DijkstraAlgorithm(use_cuda=False)  # CPU para compatibilidad
matriz = graph.get_adjacency_matrix()

resultados = []

for h_data in hospitales_nodos:
    print(f"\n🏥 {h_data['hospital']}")
    print(f"   Nodo: {h_data['nodo']:,}")
    
    start_time = time.time()
    
    # Ejecutar Dijkstra desde el hospital
    metrics = dijkstra.compute_shortest_paths(
        matriz,
        source_node=h_data['nodo']
    )
    
    elapsed = time.time() - start_time
    
    # Obtener distancia al usuario
    if usuario_nodo < len(metrics.distances_computed):
        # distances_computed es un diccionario
        distancia = metrics.distances_computed.get(usuario_nodo, float('inf'))
        
        if distancia != float('inf'):
            distancia_km = distancia / 1000  # Convertir a km
            tiempo_estimado = distancia_km / 40 * 60  # Asumiendo 40 km/h, en minutos
            
            resultados.append({
                'hospital': h_data['hospital'],
                'nodo': h_data['nodo'],
                'distancia_m': distancia,
                'distancia_km': distancia_km,
                'tiempo_min': tiempo_estimado,
                'tiempo_calculo': elapsed
            })
            
            print(f"   📏 Distancia a Plaza de Armas: {distancia_km:.2f} km")
            print(f"   ⏱️  Tiempo estimado: {tiempo_estimado:.1f} minutos")
            print(f"   💻 Tiempo de cálculo: {elapsed:.3f} segundos")
        else:
            print(f"   ❌ No hay ruta disponible")
    else:
        print(f"   ⚠️  Nodo fuera de rango")

# 6. Mostrar hospital más cercano
print()
print("=" * 70)
print("🎯 RESULTADO: HOSPITAL MÁS CERCANO")
print("=" * 70)

if resultados:
    resultados_ordenados = sorted(resultados, key=lambda x: x['distancia_km'])
    
    print()
    for i, r in enumerate(resultados_ordenados, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        print(f"{emoji} {r['hospital']}")
        print(f"   📏 Distancia: {r['distancia_km']:.2f} km")
        print(f"   ⏱️  Tiempo estimado: {r['tiempo_min']:.1f} minutos")
        print(f"   💻 Cálculo en: {r['tiempo_calculo']:.3f} segundos")
        print()
    
    mejor = resultados_ordenados[0]
    print("=" * 70)
    print(f"🚑 RECOMENDACIÓN: Enviar ambulancia desde {mejor['hospital']}")
    print(f"   Ruta óptima: {mejor['distancia_km']:.2f} km (~{mejor['tiempo_min']:.1f} min)")
    print("=" * 70)
else:
    print("❌ No se encontraron rutas disponibles")

print()
print("✅ Análisis completado")
print()
print("💡 Para usar ubicaciones personalizadas, modifica las coordenadas")
print("   en este script (usuario_lat, usuario_lon)")
