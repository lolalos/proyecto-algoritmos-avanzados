"""Script para descargar SOLO la provincia de Cusco (ciudad) de forma rápida"""
import sys
import time
from graph import UrbanGraph

print("🚀 Descargando SOLO provincia de Cusco (ciudad)")
print("⚡ Mucho más rápido que el departamento completo\n")

start_time = time.time()

# Crear instancia del grafo
graph = UrbanGraph()

# Descargar solo el distrito de Cusco (ciudad)
success = graph.download_distrito_from_osm(
    query="Cusco, Cusco, Peru",
    distrito_key="cusco_cusco_cusco"
)

elapsed = time.time() - start_time

if success:
    print(f"\n✅ ¡DESCARGA COMPLETADA!")
    print(f"⏱️  Tiempo total: {elapsed:.2f} segundos")
    print(f"📊 Nodos: {graph.num_nodes:,}")
    print(f"🔗 Aristas: {graph.num_edges:,}")
    print(f"💾 Guardado en: backend/mapas/cusco_cusco_cusco_graph.pkl")
    print(f"\n🎯 Ahora carga este mapa desde la interfaz web")
else:
    print(f"\n❌ Error en la descarga")
    sys.exit(1)
