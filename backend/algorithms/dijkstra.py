"""
Implementación del algoritmo de Dijkstra optimizado con CUDA.
Soporta ejecución tanto en CPU (NumPy) como en GPU (CuPy).
Usa matrices dispersas para grafos grandes.
"""
import numpy as np
from scipy import sparse
import heapq
from typing import Dict, Optional
from .base import ShortestPathAlgorithm, AlgorithmMetrics

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None


class DijkstraAlgorithm(ShortestPathAlgorithm):
    """Implementación del algoritmo de Dijkstra con soporte CUDA."""
    
    def __init__(self, use_cuda: bool = True):
        super().__init__("Dijkstra", use_cuda=use_cuda and CUDA_AVAILABLE)
        if use_cuda and not CUDA_AVAILABLE:
            print("⚠️  CUDA no disponible. Usando implementación CPU con NumPy.")
    
    def compute_shortest_paths(
        self, 
        graph_matrix, 
        source_node: int,
        node_mapping: Optional[Dict] = None
    ) -> AlgorithmMetrics:
        """
        Implementa Dijkstra optimizado para matrices dispersas y densas.
        
        Complejidad: O((V + E) log V) con heap binario
        """
        self._start_metrics_tracking()
        
        n_nodes = graph_matrix.shape[0]
        is_sparse = sparse.issparse(graph_matrix)
        
        # Para grafos dispersos grandes, usar implementación con heap
        if is_sparse or n_nodes > 10000:
            return self._dijkstra_sparse(graph_matrix, source_node, n_nodes, is_sparse)
        
        # Para grafos pequeños, usar implementación vectorizada
        if self.use_cuda:
            return self._dijkstra_cuda(graph_matrix, source_node, n_nodes)
        else:
            return self._dijkstra_numpy(graph_matrix, source_node, n_nodes)
    
    def _dijkstra_sparse(self, graph_matrix, source_node: int, n_nodes: int, is_sparse: bool):
        """Dijkstra con heap para grafos dispersos grandes."""
        distances = np.full(n_nodes, np.inf, dtype=np.float32)
        distances[source_node] = 0.0
        parent = np.full(n_nodes, -1, dtype=np.int32)
        visited = set()
        
        # Heap: (distancia, nodo)
        heap = [(0.0, source_node)]
        
        while heap:
            current_dist, current = heapq.heappop(heap)
            
            if current in visited:
                continue
            
            visited.add(current)
            self.metrics.nodes_processed += 1
            
            # Obtener vecinos
            if is_sparse:
                # Para matriz dispersa CSR
                row = graph_matrix.getrow(current)
                neighbors = row.nonzero()[1]
                weights = row.data
            else:
                # Para matriz densa
                neighbors = np.where(graph_matrix[current] > 0)[0]
                weights = graph_matrix[current, neighbors]
            
            # Relajar aristas
            for i, neighbor in enumerate(neighbors):
                if neighbor not in visited:
                    new_dist = current_dist + weights[i]
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        parent[neighbor] = current
                        heapq.heappush(heap, (new_dist, neighbor))
                        self.metrics.edge_relaxations += 1
        
        # Construir resultados
        self.metrics.distances_computed = {
            i: float(distances[i]) 
            for i in range(n_nodes) 
            if distances[i] != np.inf
        }
        
        self.metrics.path_to_nodes = self._reconstruct_paths(
            parent, source_node, n_nodes
        )
        
        self._stop_metrics_tracking()
        return self.metrics
    
    def _dijkstra_cuda(self, graph_matrix: np.ndarray, source_node: int, n_nodes: int):
        """Dijkstra vectorizado en GPU para grafos pequeños."""
        # Transferir datos a GPU
        graph_gpu = cp.asarray(graph_matrix)
        distances = cp.full(n_nodes, cp.inf, dtype=cp.float32)
        distances[source_node] = 0.0
        visited = cp.zeros(n_nodes, dtype=bool)
        parent = cp.full(n_nodes, -1, dtype=cp.int32)
        
        # Algoritmo de Dijkstra en GPU
        for _ in range(n_nodes):
            # Encontrar el nodo no visitado con menor distancia
            unvisited_mask = ~visited
            temp_distances = cp.where(unvisited_mask, distances, cp.inf)
            current = cp.argmin(temp_distances)
            
            # Si la distancia mínima es infinita, no hay más nodos alcanzables
            if distances[current] == cp.inf:
                break
            
            visited[current] = True
            self.metrics.nodes_processed += 1
            
            # Relajar las aristas del nodo actual
            neighbors = graph_gpu[current] > 0
            neighbor_indices = cp.where(neighbors)[0]
            
            for neighbor in neighbor_indices:
                if not visited[neighbor]:
                    new_distance = distances[current] + graph_gpu[current, neighbor]
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        parent[neighbor] = current
                        self.metrics.edge_relaxations += 1
        
        # Transferir resultados de vuelta a CPU
        distances_cpu = cp.asnumpy(distances)
        parent_cpu = cp.asnumpy(parent)
        
        # Construir diccionario de distancias
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
    
    def _dijkstra_numpy(self, graph_matrix: np.ndarray, source_node: int, n_nodes: int):
        """Dijkstra vectorizado en CPU para grafos pequeños."""
        # Implementación CPU con NumPy
        distances = np.full(n_nodes, np.inf, dtype=np.float32)
        distances[source_node] = 0.0
        visited = np.zeros(n_nodes, dtype=bool)
        parent_cpu = np.full(n_nodes, -1, dtype=np.int32)
        
        for _ in range(n_nodes):
            # Encontrar nodo no visitado con menor distancia
            unvisited_mask = ~visited
            temp_distances = np.where(unvisited_mask, distances, np.inf)
            current = np.argmin(temp_distances)
            
            if distances[current] == np.inf:
                break
            
            visited[current] = True
            self.metrics.nodes_processed += 1
            
            # Relajar aristas
            for neighbor in range(n_nodes):
                if graph_matrix[current, neighbor] > 0 and not visited[neighbor]:
                    new_distance = distances[current] + graph_matrix[current, neighbor]
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        parent_cpu[neighbor] = current
                        self.metrics.edge_relaxations += 1
        
        distances_cpu = distances
        
        # Construir diccionario de distancias
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
    
    def _reconstruct_paths(
        self, 
        parent: np.ndarray, 
        source: int, 
        n_nodes: int
    ) -> Dict[int, list]:
        """Reconstruye los caminos óptimos desde el nodo fuente."""
        paths = {}
        
        for target in range(n_nodes):
            if target == source:
                paths[target] = [source]
                continue
            
            if parent[target] == -1:
                continue  # Nodo inalcanzable
            
            path = []
            current = target
            while current != -1:
                path.append(current)
                current = parent[current]
                if current == source:
                    path.append(source)
                    break
            
            paths[target] = path[::-1]  # Invertir para obtener camino de source a target
        
        return paths


class DijkstraPriorityQueue(ShortestPathAlgorithm):
    """Dijkstra optimizado con cola de prioridad (heap) - más eficiente para grafos dispersos."""
    
    def __init__(self, use_cuda: bool = True):
        super().__init__("Dijkstra-PriorityQueue", use_cuda=use_cuda and CUDA_AVAILABLE)
        if use_cuda and not CUDA_AVAILABLE:
            print("⚠️  CUDA no disponible. Usando implementación CPU.")
    
    def compute_shortest_paths(
        self, 
        graph_matrix: np.ndarray, 
        source_node: int,
        node_mapping: Optional[Dict] = None
    ) -> AlgorithmMetrics:
        """
        Dijkstra con heap binario para mejor eficiencia en grafos grandes.
        Complejidad: O((V + E) log V)
        """
        import heapq
        
        self._start_metrics_tracking()
        
        n_nodes = graph_matrix.shape[0]
        distances = {i: float('inf') for i in range(n_nodes)}
        distances[source_node] = 0.0
        parent = {i: -1 for i in range(n_nodes)}
        
        # Cola de prioridad: (distancia, nodo)
        pq = [(0.0, source_node)]
        visited = set()
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current in visited:
                continue
            
            visited.add(current)
            self.metrics.nodes_processed += 1
            
            # Explorar vecinos
            for neighbor in range(n_nodes):
                edge_weight = graph_matrix[current, neighbor]
                
                if edge_weight > 0 and neighbor not in visited:
                    new_distance = current_dist + edge_weight
                    
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        parent[neighbor] = current
                        heapq.heappush(pq, (new_distance, neighbor))
                        self.metrics.edge_relaxations += 1
        
        # Guardar resultados
        self.metrics.distances_computed = {
            k: v for k, v in distances.items() if v != float('inf')
        }
        
        # Reconstruir caminos
        self.metrics.path_to_nodes = {}
        for target in range(n_nodes):
            if distances[target] != float('inf'):
                path = []
                current = target
                while current != -1:
                    path.append(current)
                    current = parent[current]
                self.metrics.path_to_nodes[target] = path[::-1]
        
        self._stop_metrics_tracking()
        return self.metrics
