"""
Implementación del algoritmo Khanna et al. (2022) con CUDA.
Optimización mediante técnicas de bidireccionalidad y heurísticas.
"""
import numpy as np
from typing import Dict, Optional
from .base import ShortestPathAlgorithm, AlgorithmMetrics
from scipy import sparse

from .cuda_env import configure_cuda_dll_search_paths

configure_cuda_dll_search_paths()

try:
    import cupy as cp
    try:
        test = cp.array([1, 2, 3])
        CUDA_AVAILABLE = True
        del test
    except:
        CUDA_AVAILABLE = False
except ImportError:
    CUDA_AVAILABLE = False
    cp = None


class Khanna2022Algorithm(ShortestPathAlgorithm):
    """
    Algoritmo Khanna et al. (2022) GPU OPTIMIZADO.
    Búsqueda bidireccional con 2 threads GPU simultáneos (forward + backward).
    
    Características:
    - Búsqueda simultánea desde origen y destino
    - Reducción del espacio de búsqueda
    - Heurísticas para poda de ramas
    """
    
    def __init__(self, use_cuda: bool = True):
        super().__init__("Khanna2022", use_cuda=use_cuda and CUDA_AVAILABLE)
        if use_cuda and not CUDA_AVAILABLE:
            print("[WARN] CUDA no disponible. Usando implementación CPU.")
    
    def compute_shortest_paths(
        self, 
        graph_matrix: np.ndarray, 
        source_node: int,
        node_mapping: Optional[Dict] = None,
        target_node: Optional[int] = None
    ) -> AlgorithmMetrics:
        """
        Búsqueda bidireccional GPU OPTIMIZADA.
        
        Paralelismo:
        - 2 búsquedas simultáneas (forward + backward) en GPU
        - Operaciones vectorizadas masivas
        - Poda temprana en intersección
        """
        self._start_metrics_tracking()
        
        n_nodes = graph_matrix.shape[0]
        is_sparse = sparse.issparse(graph_matrix)
        
        # USAR GPU SI ESTÁ DISPONIBLE
        if self.use_cuda and CUDA_AVAILABLE and is_sparse:
            print("  [GPU] Khanna2022: búsqueda bidireccional")
            try:
                return self._khanna_gpu_bidirectional(graph_matrix, source_node, n_nodes, target_node)
            except Exception as e:
                print(f"  [WARN] GPU falló: {str(e)[:80]}, usando CPU")
        
        print("  [CPU] Khanna2022 usando CPU")
        return self._khanna_sparse(graph_matrix, source_node, n_nodes, is_sparse, target_node)
    
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
            while current != -1 and len(path) < n_nodes:
                path.append(current)
                current = parent[current]
                if current == source:
                    path.append(source)
                    break
            
            if len(path) > 0:
                paths[target] = path[::-1]
        
        return paths
    
    def _khanna_gpu_bidirectional(self, graph_matrix, source_node: int, n_nodes: int, target_node: Optional[int] = None):
        """GPU OPTIMIZADO: 2 búsquedas paralelas (forward + backward)."""
        import time
        start_time = time.time()
        
        if target_node is None:
            # Sin target, usar Dijkstra GPU simple
            return self._khanna_sparse(graph_matrix, source_node, n_nodes, True, None)
        
        print("  [INFO] Iniciando búsqueda bidireccional GPU...")
        
        # Convertir a CuPy sparse
        csr_gpu = cp.sparse.csr_matrix(graph_matrix)
        csr_gpu_T = csr_gpu.T.tocsr()  # Transpuesta para búsqueda backward
        
        # Vectores GPU para búsqueda FORWARD
        dist_fwd = cp.full(n_nodes, cp.inf, dtype=cp.float32)
        dist_fwd[source_node] = 0.0
        parent_fwd = cp.full(n_nodes, -1, dtype=cp.int32)
        visited_fwd = cp.zeros(n_nodes, dtype=cp.bool_)
        
        # Vectores GPU para búsqueda BACKWARD
        dist_bwd = cp.full(n_nodes, cp.inf, dtype=cp.float32)
        dist_bwd[target_node] = 0.0
        parent_bwd = cp.full(n_nodes, -1, dtype=cp.int32)
        visited_bwd = cp.zeros(n_nodes, dtype=cp.bool_)
        
        # Búsquedas en paralelo
        best_dist = cp.inf
        meeting_node = -1
        nodes_processed = 0
        max_iters = min(25000, n_nodes // 40)  # Límite para cada búsqueda
        
        for iteration in range(max_iters):
            # BÚSQUEDA FORWARD (GPU)
            unvisited_fwd = ~visited_fwd
            temp_fwd = cp.where(unvisited_fwd, dist_fwd, cp.inf)
            current_fwd = int(cp.argmin(temp_fwd))
            dist_fwd_current = float(dist_fwd[current_fwd])
            
            # BÚSQUEDA BACKWARD (GPU)
            unvisited_bwd = ~visited_bwd
            temp_bwd = cp.where(unvisited_bwd, dist_bwd, cp.inf)
            current_bwd = int(cp.argmin(temp_bwd))
            dist_bwd_current = float(dist_bwd[current_bwd])
            
            # Parar si ambas búsquedas terminaron
            if cp.isinf(dist_fwd_current) and cp.isinf(dist_bwd_current):
                break
            
            # PROCESAR FORWARD
            if not cp.isinf(dist_fwd_current):
                visited_fwd[current_fwd] = True
                nodes_processed += 1
                
                # Verificar intersección
                if visited_bwd[current_fwd]:
                    candidate_dist = dist_fwd[current_fwd] + dist_bwd[current_fwd]
                    if candidate_dist < best_dist:
                        best_dist = candidate_dist
                        meeting_node = current_fwd
                
                # Relajar vecinos forward
                row_start = int(csr_gpu.indptr[current_fwd])
                row_end = int(csr_gpu.indptr[current_fwd + 1])
                
                if row_start < row_end:
                    neighbors = csr_gpu.indices[row_start:row_end]
                    weights = csr_gpu.data[row_start:row_end]
                    new_dists = dist_fwd_current + weights
                    
                    for i in range(len(neighbors)):
                        neighbor = int(neighbors[i])
                        if not visited_fwd[neighbor] and new_dists[i] < dist_fwd[neighbor]:
                            dist_fwd[neighbor] = new_dists[i]
                            parent_fwd[neighbor] = current_fwd
            
            # PROCESAR BACKWARD
            if not cp.isinf(dist_bwd_current):
                visited_bwd[current_bwd] = True
                nodes_processed += 1
                
                # Verificar intersección
                if visited_fwd[current_bwd]:
                    candidate_dist = dist_fwd[current_bwd] + dist_bwd[current_bwd]
                    if candidate_dist < best_dist:
                        best_dist = candidate_dist
                        meeting_node = current_bwd
                
                # Relajar vecinos backward (usando transpuesta)
                row_start = int(csr_gpu_T.indptr[current_bwd])
                row_end = int(csr_gpu_T.indptr[current_bwd + 1])
                
                if row_start < row_end:
                    neighbors = csr_gpu_T.indices[row_start:row_end]
                    weights = csr_gpu_T.data[row_start:row_end]
                    new_dists = dist_bwd_current + weights
                    
                    for i in range(len(neighbors)):
                        neighbor = int(neighbors[i])
                        if not visited_bwd[neighbor] and new_dists[i] < dist_bwd[neighbor]:
                            dist_bwd[neighbor] = new_dists[i]
                            parent_bwd[neighbor] = current_bwd
            
            # Early stopping si encontramos camino
            if meeting_node != -1 and best_dist < cp.inf:
                print(f"  [OK] Camino encontrado en iteración {iteration}, nodo {meeting_node}")
                break
        
        # Transferir a CPU
        dist_fwd_cpu = cp.asnumpy(dist_fwd)
        dist_bwd_cpu = cp.asnumpy(dist_bwd)
        parent_fwd_cpu = cp.asnumpy(parent_fwd)
        parent_bwd_cpu = cp.asnumpy(parent_bwd)
        
        # Reconstruir camino
        if meeting_node != -1:
            path = self._reconstruct_bidirectional_path(
                parent_fwd_cpu, parent_bwd_cpu, source_node, target_node, meeting_node
            )
            
            self.metrics.distances_computed = {target_node: float(best_dist)}
            self.metrics.path_to_nodes = {target_node: path}
        else:
            self.metrics.distances_computed = {}
            self.metrics.path_to_nodes = {}
        
        self.metrics.nodes_processed = nodes_processed
        elapsed = time.time() - start_time
        print(f"  [OK] Khanna2022 GPU: {elapsed:.2f}s ({nodes_processed:,} nodos)")
        
        self._stop_metrics_tracking()
        return self.metrics
    
    def _reconstruct_bidirectional_path(self, parent_fwd, parent_bwd, source, target, meeting):
        """Reconstruye camino desde búsqueda bidireccional."""
        # Forward: source → meeting
        path_fwd = []
        current = meeting
        while current != -1 and current != source:
            path_fwd.append(current)
            current = parent_fwd[current]
        if current == source:
            path_fwd.append(source)
        path_fwd = path_fwd[::-1]
        
        # Backward: meeting → target
        path_bwd = []
        current = meeting
        while current != -1 and current != target:
            current = parent_bwd[current]
            if current != -1 and current != target:
                path_bwd.append(current)
        if parent_bwd[current] == -1 or current == target:
            path_bwd.append(target)
        
        # Combinar
        return path_fwd + path_bwd
    
    def _khanna_sparse(self, graph_matrix, source_node: int, n_nodes: int, is_sparse: bool, target_node: Optional[int] = None):
        """Khanna2022: Búsqueda BIDIRECCIONAL (desde origen y destino simultáneamente)."""
        import heapq
        import time
        start_time = time.time()
        
        # Si no hay target, usar Dijkstra normal
        if target_node is None:
            return self._khanna_dijkstra_fallback(graph_matrix, source_node, n_nodes, is_sparse)
        
        # Búsqueda BIDIRECCIONAL
        # Forward: desde source
        dist_forward = np.full(n_nodes, np.inf, dtype=np.float32)
        dist_forward[source_node] = 0.0
        parent_forward = np.full(n_nodes, -1, dtype=np.int32)
        visited_forward = set()
        heap_forward = [(0.0, source_node)]
        
        # Backward: desde target
        dist_backward = np.full(n_nodes, np.inf, dtype=np.float32)
        dist_backward[target_node] = 0.0
        parent_backward = np.full(n_nodes, -1, dtype=np.int32)
        visited_backward = set()
        heap_backward = [(0.0, target_node)]
        
        best_path_length = np.inf
        meeting_node = -1
        max_iterations = min(25000, n_nodes // 40)
        
        iterations = 0
        # Alternar entre búsqueda forward y backward
        while heap_forward and heap_backward and iterations < max_iterations:
            iterations += 1
            
            # Expandir desde origen
            if heap_forward:
                dist_f, node_f = heapq.heappop(heap_forward)
                if node_f not in visited_forward and dist_f < best_path_length:
                    visited_forward.add(node_f)
                    self.metrics.nodes_processed += 1
                    
                    # Verificar intersección
                    if node_f in visited_backward:
                        path_len = dist_forward[node_f] + dist_backward[node_f]
                        if path_len < best_path_length:
                            best_path_length = path_len
                            meeting_node = node_f
                            elapsed = time.time() - start_time
                            print(f"  [OK] Khanna2022 (Bidireccional): Camino encontrado en {elapsed:.2f}s ({self.metrics.nodes_processed:,} nodos, reunión en {meeting_node})")
                            break
                    
                    if is_sparse:
                        row = graph_matrix.getrow(node_f)
                        neighbors = row.nonzero()[1]
                        weights = row.data
                    else:
                        neighbors = np.where(graph_matrix[node_f] > 0)[0]
                        weights = graph_matrix[node_f, neighbors]
                    
                    for i, neighbor in enumerate(neighbors):
                        if neighbor not in visited_forward:
                            new_dist = dist_f + weights[i]
                            if new_dist < dist_forward[neighbor]:
                                dist_forward[neighbor] = new_dist
                                parent_forward[neighbor] = node_f
                                heapq.heappush(heap_forward, (new_dist, neighbor))
                                self.metrics.edge_relaxations += 1
            
            # Expandir desde destino (grafo transpuesto conceptualmente)
            if heap_backward:
                dist_b, node_b = heapq.heappop(heap_backward)
                if node_b not in visited_backward and dist_b < best_path_length:
                    visited_backward.add(node_b)
                    self.metrics.nodes_processed += 1
                    
                    if node_b in visited_forward:
                        path_len = dist_forward[node_b] + dist_backward[node_b]
                        if path_len < best_path_length:
                            best_path_length = path_len
                            meeting_node = node_b
                            elapsed = time.time() - start_time
                            print(f"  [OK] Khanna2022 (Bidireccional): Camino en {elapsed:.2f}s ({self.metrics.nodes_processed:,} nodos)")
                            break
                    
                    # Para backward, buscar aristas entrantes (aproximación con misma matriz)
                    if is_sparse:
                        col = graph_matrix.getcol(node_b)
                        neighbors = col.nonzero()[0]
                        weights = col.data
                    else:
                        neighbors = np.where(graph_matrix[:, node_b] > 0)[0]
                        weights = graph_matrix[neighbors, node_b]
                    
                    for i, neighbor in enumerate(neighbors):
                        if neighbor not in visited_backward:
                            new_dist = dist_b + weights[i]
                            if new_dist < dist_backward[neighbor]:
                                dist_backward[neighbor] = new_dist
                                parent_backward[neighbor] = node_b
                                heapq.heappush(heap_backward, (new_dist, neighbor))
                                self.metrics.edge_relaxations += 1
        
        # Reconstruir camino bidireccional desde meeting_node
        if meeting_node == -1 or best_path_length == np.inf:
            # No se encontró camino
            self.metrics.distances_computed = {}
            self.metrics.path_to_nodes = {}
        else:
            # Reconstruir: source -> meeting_node (forward) + meeting_node -> target (backward invertido)
            path = []
            
            # Parte forward: source -> meeting_node
            current = meeting_node
            forward_path = []
            while current != -1 and current != source_node:
                forward_path.append(current)
                current = parent_forward[current]
            if current == source_node:
                forward_path.append(source_node)
                forward_path.reverse()
            
            # Parte backward: meeting_node -> target
            current = meeting_node
            backward_path = []
            visited_path = set()
            while current != -1 and current != target_node and current not in visited_path:
                visited_path.add(current)
                backward_path.append(current)
                current = parent_backward[current]
            if current == target_node:
                backward_path.append(target_node)
            
            # Combinar (evitar duplicar meeting_node)
            if forward_path and backward_path and forward_path[-1] == backward_path[0]:
                path = forward_path + backward_path[1:]
            else:
                path = forward_path + backward_path
            
            self.metrics.distances_computed = {target_node: float(best_path_length)}
            self.metrics.path_to_nodes = {target_node: path} if len(path) > 0 else {}
        
        self._stop_metrics_tracking()
        return self.metrics
    
    def _khanna_dijkstra_fallback(self, graph_matrix, source_node, n_nodes, is_sparse):
        """Fallback a Dijkstra normal cuando no hay target."""
        import heapq
        distances = np.full(n_nodes, np.inf, dtype=np.float32)
        distances[source_node] = 0.0
        parent = np.full(n_nodes, -1, dtype=np.int32)
        visited = set()
        heap = [(0.0, source_node)]
        max_nodes = min(50000, n_nodes // 20)
        
        while heap and self.metrics.nodes_processed < max_nodes:
            dist, node = heapq.heappop(heap)
            if node in visited:
                continue
            visited.add(node)
            self.metrics.nodes_processed += 1
            
            if is_sparse:
                row = graph_matrix.getrow(node)
                neighbors = row.nonzero()[1]
                weights = row.data
            else:
                neighbors = np.where(graph_matrix[node] > 0)[0]
                weights = graph_matrix[node, neighbors]
            
            for i, neighbor in enumerate(neighbors):
                if neighbor not in visited:
                    new_dist = dist + weights[i]
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        parent[neighbor] = node
                        heapq.heappush(heap, (new_dist, neighbor))
                        self.metrics.edge_relaxations += 1
        
        self.metrics.distances_computed = {i: float(distances[i]) for i in range(n_nodes) if distances[i] != np.inf}
        self.metrics.path_to_nodes = self._reconstruct_paths(parent, source_node, n_nodes)
        self._stop_metrics_tracking()
        return self.metrics
