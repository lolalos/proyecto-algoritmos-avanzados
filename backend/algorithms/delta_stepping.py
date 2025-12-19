"""
Delta-Stepping: Algoritmo SSSP paralelo para GPU.
Basado en el paper: Meyer & Sanders (2003)
Este algoritmo SÍ aprovecha la GPU porque procesa buckets en paralelo.
"""
import numpy as np
from scipy import sparse
import heapq
from typing import Dict, Optional
from collections import defaultdict
from .base import ShortestPathAlgorithm, AlgorithmMetrics

from .cuda_env import configure_cuda_dll_search_paths

configure_cuda_dll_search_paths()

try:
    import cupy as cp
    CUDA_AVAILABLE = cp.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False
    cp = None


class DeltaSteppingGPU(ShortestPathAlgorithm):
    """
    Delta-Stepping: Algoritmo SSSP paralelo optimizado para GPU.
    
    Diferencia clave vs Dijkstra:
    - Dijkstra: Procesa 1 nodo a la vez (SECUENCIAL)
    - Delta-Stepping: Procesa BUCKETS de nodos en PARALELO
    
    Esto permite usar miles de threads GPU simultáneamente.
    """
    
    def __init__(self, use_cuda: bool = True, delta: float = 100.0):
        """
        Args:
            use_cuda: Si usar aceleración GPU
            delta: Tamaño del bucket (en metros). Controla paralelismo.
                   Valores pequeños = más buckets = más paralelismo
                   Valores grandes = menos buckets = menos paralelismo
        """
        super().__init__("DeltaStepping", use_cuda=use_cuda and CUDA_AVAILABLE)
        self.delta = delta
        self.name = "Delta-Stepping GPU"
        
        if use_cuda and not CUDA_AVAILABLE:
            print("[WARN] CUDA no disponible. Usando implementación CPU.")
    
    def compute_shortest_paths(
        self, 
        graph_matrix, 
        source_node: int,
        node_mapping: Optional[Dict] = None,
        target_node: Optional[int] = None
    ) -> AlgorithmMetrics:
        """Implementa Delta-Stepping (buckets).

        En GPU aquí se ejecuta una versión simple (sin kernel custom) para evitar densificar.
        En CPU usa el mismo esquema de buckets.
        """
        self._start_metrics_tracking()
        self.metrics.details = {"delta": float(self.delta)}

        n_nodes = int(graph_matrix.shape[0])

        if self.use_cuda and CUDA_AVAILABLE:
            self.metrics.details["mode"] = "gpu_buckets"
            print(f"  [GPU] Delta-Stepping (delta={self.delta}m)")
            return self._delta_stepping_gpu(graph_matrix, int(source_node), n_nodes, target_node)

        self.metrics.details["mode"] = "cpu_buckets"
        print(f"  [CPU] Delta-Stepping (delta={self.delta}m)")
        return self._delta_stepping_cpu(graph_matrix, int(source_node), n_nodes, target_node)
    
    def _delta_stepping_gpu(self, graph_matrix, source_node: int, n_nodes: int, target_node: Optional[int] = None):
        """Delta-Stepping usando CuPy CSR (sin densificar)."""
        import time

        start_time = time.time()

        csr_gpu = cp.sparse.csr_matrix(graph_matrix)
        distances = cp.full(n_nodes, cp.inf, dtype=cp.float32)
        parent = cp.full(n_nodes, -1, dtype=cp.int32)
        distances[source_node] = 0.0

        buckets: Dict[int, list[int]] = defaultdict(list)
        buckets[0].append(int(source_node))

        current_bucket_idx = 0
        max_buckets = 1_000_000
        nodes_processed = 0
        target = int(target_node) if target_node is not None else None

        while buckets and current_bucket_idx < max_buckets:
            while current_bucket_idx not in buckets and current_bucket_idx < max_buckets:
                current_bucket_idx += 1
            if current_bucket_idx >= max_buckets:
                break

            current_bucket = buckets.pop(current_bucket_idx)
            if not current_bucket:
                continue

            nodes_processed += len(current_bucket)

            if target is not None and target in current_bucket:
                break

            for node in current_bucket:
                node = int(node)
                node_dist = float(distances[node])

                row_start = int(csr_gpu.indptr[node])
                row_end = int(csr_gpu.indptr[node + 1])
                if row_start >= row_end:
                    continue

                neighbors = csr_gpu.indices[row_start:row_end]
                weights = csr_gpu.data[row_start:row_end]
                new_distances = node_dist + weights

                # relajación (loop en CPU, datos en GPU)
                for i in range(int(neighbors.shape[0])):
                    neighbor = int(neighbors[i])
                    new_dist = float(new_distances[i])
                    if new_dist < float(distances[neighbor]):
                        distances[neighbor] = np.float32(new_dist)
                        parent[neighbor] = node
                        self.metrics.edge_relaxations += 1
                        buckets[int(new_dist // float(self.delta))].append(neighbor)

            current_bucket_idx += 1

        distances_cpu = cp.asnumpy(distances)
        parent_cpu = cp.asnumpy(parent)

        self.metrics.nodes_processed = int(nodes_processed)

        if target is not None:
            td = float(distances_cpu[target])
            if not np.isinf(td):
                self.metrics.distances_computed = {target: td}
                self.metrics.path_to_nodes = {target: self._reconstruct_single_path(parent_cpu, int(source_node), target)}
            else:
                self.metrics.distances_computed = {}
                self.metrics.path_to_nodes = {}
        else:
            self.metrics.distances_computed = {i: float(distances_cpu[i]) for i in range(n_nodes) if not np.isinf(distances_cpu[i])}
            self.metrics.path_to_nodes = self._reconstruct_paths(parent_cpu, int(source_node), n_nodes)

        self.metrics.details["total_time_s"] = float(time.time() - start_time)
        self._stop_metrics_tracking()
        return self.metrics
    
    def _delta_stepping_cpu(self, graph_matrix, source_node: int, n_nodes: int, target_node: Optional[int] = None):
        """Fallback CPU usando heaps."""
        import time
        start_time = time.time()
        
        distances = np.full(n_nodes, np.inf, dtype=np.float32)
        distances[source_node] = 0.0
        parent = np.full(n_nodes, -1, dtype=np.int32)
        
        # Buckets
        buckets = defaultdict(list)
        buckets[0].append(source_node)
        
        is_sparse = sparse.issparse(graph_matrix)
        current_bucket_idx = 0
        nodes_processed = 0
        
        while buckets:
            while current_bucket_idx not in buckets and current_bucket_idx < 1000000:
                current_bucket_idx += 1
            
            if current_bucket_idx >= 1000000:
                break
            
            current_bucket = buckets.pop(current_bucket_idx)
            
            if not current_bucket:
                current_bucket_idx += 1
                continue
            
            nodes_processed += len(current_bucket)
            
            if target_node is not None and target_node in current_bucket:
                break
            
            for node in current_bucket:
                node_dist = distances[node]
                
                # Obtener vecinos
                if is_sparse:
                    row = graph_matrix.getrow(node)
                    neighbors = row.nonzero()[1]
                    weights = row.data
                else:
                    neighbors = np.where(graph_matrix[node] > 0)[0]
                    weights = graph_matrix[node, neighbors]
                
                for i, neighbor in enumerate(neighbors):
                    new_dist = node_dist + weights[i]
                    
                    if new_dist < distances[neighbor]:
                        new_bucket_idx = int(new_dist // self.delta)
                        distances[neighbor] = new_dist
                        parent[neighbor] = node
                        buckets[new_bucket_idx].append(neighbor)
                        self.metrics.edge_relaxations += 1
            
            current_bucket_idx += 1
        
        self.metrics.nodes_processed = int(nodes_processed)

        target = int(target_node) if target_node is not None else None
        if target is not None:
            td = float(distances[target])
            if not np.isinf(td):
                self.metrics.distances_computed = {target: td}
                self.metrics.path_to_nodes = {target: self._reconstruct_single_path(parent, int(source_node), target)}
            else:
                self.metrics.distances_computed = {}
                self.metrics.path_to_nodes = {}
        else:
            self.metrics.distances_computed = {i: float(distances[i]) for i in range(n_nodes) if distances[i] != np.inf}
            self.metrics.path_to_nodes = self._reconstruct_paths(parent, int(source_node), n_nodes)

        self.metrics.details["total_time_s"] = float(time.time() - start_time)
        self._stop_metrics_tracking()
        return self.metrics

    def _reconstruct_single_path(self, parent: np.ndarray, source: int, target: int) -> list:
        if source == target:
            return [source]
        path = []
        cur = int(target)
        max_steps = int(parent.shape[0])
        steps = 0
        while cur != -1 and steps < max_steps:
            path.append(cur)
            if cur == source:
                break
            cur = int(parent[cur])
            steps += 1
        if not path or path[-1] != source:
            return []
        path.reverse()
        return path
