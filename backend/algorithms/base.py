"""
Clase base abstracta para algoritmos de caminos más cortos.
Define la interfaz común y las métricas de evaluación.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import time
import tracemalloc
import numpy as np


@dataclass
class AlgorithmMetrics:
    """Métricas de desempeño para un algoritmo."""
    algorithm_name: str
    execution_time: float = 0.0  # Tiempo de ejecución en segundos
    nodes_processed: int = 0  # Número de nodos procesados
    edge_relaxations: int = 0  # Número de relajaciones de aristas
    memory_peak_mb: float = 0.0  # Uso máximo de memoria en MB
    distances_computed: Dict[int, float] = field(default_factory=dict)  # Distancias finales
    path_to_nodes: Dict[int, List[int]] = field(default_factory=dict)  # Caminos óptimos
    
    def to_dict(self):
        """Convierte las métricas a diccionario para serialización."""
        return {
            'algorithm_name': self.algorithm_name,
            'execution_time': self.execution_time,
            'nodes_processed': self.nodes_processed,
            'edge_relaxations': self.edge_relaxations,
            'memory_peak_mb': self.memory_peak_mb,
            'num_distances_computed': len(self.distances_computed),
            'avg_distance': np.mean(list(self.distances_computed.values())) if self.distances_computed else 0.0,
            'max_distance': max(self.distances_computed.values()) if self.distances_computed else 0.0
        }


class ShortestPathAlgorithm(ABC):
    """Clase base abstracta para algoritmos de caminos más cortos."""
    
    def __init__(self, name: str, use_cuda: bool = True):
        """
        Args:
            name: Nombre del algoritmo
            use_cuda: Si True, usa aceleración GPU con CUDA
        """
        self.name = name
        self.use_cuda = use_cuda
        self.metrics = AlgorithmMetrics(algorithm_name=name)
    
    @abstractmethod
    def compute_shortest_paths(
        self, 
        graph_matrix: np.ndarray, 
        source_node: int,
        node_mapping: Optional[Dict] = None
    ) -> AlgorithmMetrics:
        """
        Calcula los caminos más cortos desde un nodo fuente.
        
        Args:
            graph_matrix: Matriz de adyacencia del grafo (NxN)
            source_node: Índice del nodo fuente
            node_mapping: Mapeo opcional de índices a IDs de nodos
            
        Returns:
            AlgorithmMetrics con resultados y métricas de desempeño
        """
        pass
    
    def _start_metrics_tracking(self):
        """Inicia el rastreo de métricas (tiempo y memoria)."""
        tracemalloc.start()
        self.metrics.nodes_processed = 0
        self.metrics.edge_relaxations = 0
        self.start_time = time.perf_counter()
    
    def _stop_metrics_tracking(self):
        """Detiene el rastreo de métricas."""
        self.metrics.execution_time = time.perf_counter() - self.start_time
        current, peak = tracemalloc.get_traced_memory()
        self.metrics.memory_peak_mb = peak / (1024 * 1024)  # Convertir a MB
        tracemalloc.stop()
    
    def get_metrics(self) -> AlgorithmMetrics:
        """Retorna las métricas recopiladas."""
        return self.metrics
    
    def reset_metrics(self):
        """Reinicia las métricas del algoritmo."""
        self.metrics = AlgorithmMetrics(algorithm_name=self.name)
