"""Script para descargar mapa de Cusco de forma acelerada"""
import sys
import time
from graph import UrbanGraph

print("🚀 Descargando mapa de Cusco (solo red de carros)...")
print("⏱️  Esto puede tardar 2-5 minutos dependiendo de tu conexión.\n")

start_time = time.time()

# Crear instancia del grafo
graph = UrbanGraph()

# Descargar Cusco con red de carros (drive) para máxima velocidad
success = graph.download_region_from_osm(
    region_key="cusco",
    network_type="drive"  # Solo carros, más rápido que 'all'
)

elapsed = time.time() - start_time

if success:
    print(f"\n✅ ¡DESCARGA COMPLETADA!")
    print(f"⏱️  Tiempo total: {elapsed:.2f} segundos")
    print(f"📊 Nodos: {graph.num_nodes:,}")
    print(f"🔗 Aristas: {graph.num_edges:,}")
    print(f"💾 Guardado en: backend/mapas/cusco_graph.pkl")
    print(f"\n🎯 Ahora puedes usar el sistema desde http://localhost:8000")
else:
    print(f"\n❌ Error en la descarga")
    sys.exit(1)
