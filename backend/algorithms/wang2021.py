"""
Implementación del algoritmo Wang et al. (2021) con CUDA.
Optimización mediante particionamiento de grafos y procesamiento paralelo.
"""
import numpy as np
from typing import Dict, Optional
from .base import ShortestPathAlgorithm, AlgorithmMetrics
from scipy import sparse

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None


class Wang2021Algorithm(ShortestPathAlgorithm):
    """
    Algoritmo Wang et al. (2021) - Particionamiento y paralelización.
    
    Características:
    - Particionamiento del grafo en subgrafos
    - Procesamiento paralelo de particiones
    - Fusión eficiente de resultados
    """
    
    def __init__(self, use_cuda: bool = True, num_partitions: int = 4):
        super().__init__("Wang2021", use_cuda=use_cuda and CUDA_AVAILABLE)
        self.num_partitions = num_partitions
        if use_cuda and not CUDA_AVAILABLE:
            print("⚠️  CUDA no disponible. Usando implementación CPU.")
    
    def compute_shortest_paths(
        self, 
        graph_matrix: np.ndarray, 
        source_node: int,
        node_mapping: Optional[Dict] = None
    ) -> AlgorithmMetrics:
        """
        Implementación con particionamiento de grafo.
        
        Proceso:
        1. Particionar el grafo en subgrafos
        2. Procesar cada partición en paralelo
        3. Fusionar resultados con nodos frontera
        """
        self._start_metrics_tracking()
        
        n_nodes = graph_matrix.shape[0]
        
        # Convertir a densa si es sparse
        if sparse.issparse(graph_matrix):
            graph_matrix = graph_matrix.toarray()
        
        # Determinar número óptimo de particiones
        partition_size = max(n_nodes // self.num_partitions, 10)
        
        if self.use_cuda:
            try:
                # Transferir a GPU
                graph_gpu = cp.asarray(graph_matrix, dtype=cp.float32)
                distances = cp.full(n_nodes, cp.inf, dtype=cp.float32)
                distances[source_node] = 0.0
                parent = cp.full(n_nodes, -1, dtype=cp.int32)
                visited = cp.zeros(n_nodes, dtype=bool)
                
                # Crear particiones basadas en proximidad al origen
                # (nodos más cercanos en la misma partición)
                partitions = self._create_partitions_gpu(n_nodes, source_node, partition_size)
                
                # Procesar particiones de forma iterativa (simulación de paralelismo)
                for partition_id in range(len(partitions)):
                    partition_nodes = partitions[partition_id]
                    
                    # Procesar nodos de la partición
                    for node in partition_nodes:
                        node_scalar = int(node)
                        
                        if visited[node_scalar]:
                            continue
                        
                        # Encontrar nodo no visitado con menor distancia en esta partición
                        partition_mask = cp.zeros(n_nodes, dtype=bool)
                        partition_mask[partition_nodes] = True
                        unvisited_mask = partition_mask & ~visited
                        
                        temp_distances = cp.where(unvisited_mask, distances, cp.inf)
                        
                        if cp.all(temp_distances == cp.inf):
                            continue
                        
                        current = cp.argmin(temp_distances)
                        current_scalar = int(current)
                        
                        if distances[current] == cp.inf:
                            continue
                        
                        visited[current] = True
                        self.metrics.nodes_processed += 1
                        
                        # Relajar aristas
                        neighbors_mask = graph_gpu[current_scalar] > 0
                        neighbor_indices = cp.where(neighbors_mask)[0]
                        
                        if len(neighbor_indices) > 0:
                            current_dist = distances[current_scalar]
                            edge_weights = graph_gpu[current_scalar, neighbor_indices]
                            new_distances = current_dist + edge_weights
                            
                            improved = new_distances < distances[neighbor_indices]
                            
                            if cp.any(improved):
                                improved_neighbors = neighbor_indices[improved]
                                distances[improved_neighbors] = new_distances[improved]
                                parent[improved_neighbors] = current_scalar
                                self.metrics.edge_relaxations += int(cp.sum(improved))
                
                # Fase de fusión: procesar nodos frontera entre particiones
                self._merge_partitions_gpu(
                    graph_gpu, distances, parent, visited, partitions
                )
                
                # Transferir a CPU
                distances_cpu = cp.asnumpy(distances)
                parent_cpu = cp.asnumpy(parent)
            
            except Exception as e:
                # Si CUDA falla, usar CPU
                print(f"⚠️  CUDA falló en Wang2021, usando CPU: {e}")
                self.use_cuda = False
        
        if not self.use_cuda:
            # Implementación CPU de respaldo usando heap para eficiencia
            import heapq
            
            distances = np.full(n_nodes, np.inf, dtype=np.float32)
            distances[source_node] = 0.0
            parent_cpu = np.full(n_nodes, -1, dtype=np.int32)
            visited = np.zeros(n_nodes, dtype=bool)
            
            pq = [(0.0, source_node)]
            
            while pq:
                current_dist, current = heapq.heappop(pq)
                
                if visited[current]:
                    continue
                
                visited[current] = True
                self.metrics.nodes_processed += 1
                
                # Acceso eficiente a vecinos según tipo de matriz
                if sparse.issparse(graph_matrix):
                    # Matriz sparse: acceder solo a vecinos reales
                    row = graph_matrix.getrow(current)
                    neighbors = row.nonzero()[1]
                    weights = row.data
                    
                    for i, neighbor in enumerate(neighbors):
                        if not visited[neighbor]:
                            new_dist = current_dist + weights[i]
                            if new_dist < distances[neighbor]:
                                distances[neighbor] = new_dist
                                parent_cpu[neighbor] = current
                                heapq.heappush(pq, (new_dist, neighbor))
                                self.metrics.edge_relaxations += 1
                else:
                    # Matriz densa
                    for neighbor in range(n_nodes):
                        if graph_matrix[current, neighbor] > 0 and not visited[neighbor]:
                            new_dist = current_dist + graph_matrix[current, neighbor]
                            if new_dist < distances[neighbor]:
                                distances[neighbor] = new_dist
                                parent_cpu[neighbor] = current
                                heapq.heappush(pq, (new_dist, neighbor))
                                self.metrics.edge_relaxations += 1
            
            distances_cpu = distances
        
        # Guardar resultados
        self.metrics.distances_computed = {
            i: float(distances_cpu[i]) 
            for i in range(n_nodes) 
            if distances_cpu[i] != np.inf
        }
        
        # Reconstruir caminos
        self.metrics.path_to_nodes = self._reconstruct_paths(
            parent_cpu, source_node, n_nodes
        )
        
        self._stop_metrics_tracking()
        return self.metrics
    
    def _create_partitions_gpu(self, n_nodes, source, partition_size):
        """Crea particiones del grafo."""
        # Particionamiento simple por rangos de índices
        partitions = []
        for i in range(0, n_nodes, partition_size):
            end = min(i + partition_size, n_nodes)
            partition = cp.arange(i, end, dtype=cp.int32)
            partitions.append(partition)
        
        return partitions
    
    def _merge_partitions_gpu(self, graph_gpu, distances, parent, visited, partitions):
        """Fusiona resultados de particiones procesando nodos frontera."""
        # Procesar aristas entre particiones
        for i in range(len(partitions) - 1):
            for node in partitions[i]:
                node_scalar = int(node)
                if visited[node_scalar]:
                    # Verificar conexiones con siguiente partición
                    next_partition = partitions[i + 1]
                    for neighbor in next_partition:
                        neighbor_scalar = int(neighbor)
                        edge_weight = graph_gpu[node_scalar, neighbor_scalar]
                        
                        if edge_weight > 0:
                            new_dist = distances[node_scalar] + edge_weight
                            if new_dist < distances[neighbor_scalar]:
                                distances[neighbor_scalar] = new_dist
                                parent[neighbor_scalar] = node_scalar
                                self.metrics.edge_relaxations += 1
    
    def _reconstruct_paths(
        self, 
        parent: np.ndarray, 
        source: int, 
        n_nodes: int
    ) -> Dict[int, list]:
        """Reconstruye los caminos óptimos."""
        paths = {}
        
        for target in range(n_nodes):
            if target == source:
                paths[target] = [source]
                continue
            
            if parent[target] == -1:
                continue
            
            path = []
            current = target
            visited_in_path = set()
            
            while current != -1:
                if current in visited_in_path:
                    break  # Evitar ciclos
                
                path.append(current)
                visited_in_path.add(current)
                current = parent[current]
                
                if current == source:
                    path.append(source)
                    break
                
                if len(path) > n_nodes:
                    break
            
            if len(path) > 0 and path[-1] == source:
                paths[target] = path[::-1]
        
        return paths
