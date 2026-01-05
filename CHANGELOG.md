# Changelog

Todas las versiones importantes de este proyecto serán documentadas en este archivo.

## [2.0.0] - 2026-01-05

### Añadido
- Soporte completo para GPU/CUDA con detección automática
- Algoritmos paralelos avanzados: Dijkstra, Duan2025, Khanna2022, Wang2021
- Backend FastAPI con endpoints para cálculo de rutas a hospitales
- Descarga automática de datos MTC (red vial oficial del Perú)
- Sistema de caché para grafos y mapas procesados
- Frontend interactivo con Leaflet para visualización de rutas
- Soporte para múltiples regiones: Cusco, Lima, etc.
- Documentación LaTeX para informe académico

### Mejorado
- Sistema de entorno virtual unificado (venv313)
- Gestión automática de dependencias con fallback CPU/GPU
- Optimización de memoria para grafos grandes (1M+ nodos)
- Script `start.bat` con auto-configuración

### Técnico
- Python 3.13
- NetworkX para grafos
- CuPy para operaciones CUDA
- GeoPandas para procesamiento geoespacial
- Matriz dispersa CSR para eficiencia de memoria
