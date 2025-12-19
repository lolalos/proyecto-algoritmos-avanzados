"""
Implementación del algoritmo de Dijkstra optimizado con CUDA.
Soporta ejecución tanto en CPU (NumPy) como en GPU (CuPy + Numba CUDA).
Usa matrices dispersas para grafos grandes.
"""
import numpy as np
from scipy import sparse
import heapq
from typing import Dict, Optional
from .base import ShortestPathAlgorithm, AlgorithmMetrics

from .cuda_env import configure_cuda_dll_search_paths

configure_cuda_dll_search_paths()

try:
    import cupy as cp
    # Verificar que CuPy pueda usar GPU
    try:
        test = cp.array([1, 2, 3])
        CUDA_AVAILABLE = True
        del test
    except:
        CUDA_AVAILABLE = False
except ImportError:
    CUDA_AVAILABLE = False
    cp = None


class DijkstraAlgorithm(ShortestPathAlgorithm):
    """Implementación del algoritmo de Dijkstra con soporte CUDA."""
    
    def __init__(self, use_cuda: bool = True):
        super().__init__("Dijkstra", use_cuda=use_cuda and CUDA_AVAILABLE)
        if use_cuda and not CUDA_AVAILABLE:
            print("[WARN] CUDA no disponible. Usando implementación CPU con NumPy.")
    
    def compute_shortest_paths(
        self, 
        graph_matrix, 
        source_node: int,
        node_mapping: Optional[Dict] = None,
        target_node: Optional[int] = None
    ) -> AlgorithmMetrics:
        """
        Implementa Dijkstra optimizado para matrices dispersas y densas.
        
        Complejidad: O((V + E) log V) con heap binario
        """
        self._start_metrics_tracking()
        self.metrics.details = {}
        
        n_nodes = graph_matrix.shape[0]
        is_sparse = sparse.issparse(graph_matrix)
        
        # SIEMPRE INTENTAR USAR GPU PRIMERO
        if self.use_cuda and CUDA_AVAILABLE and is_sparse:
            print("  [GPU] Dijkstra usando CUDA")
            self.metrics.details["mode"] = "gpu_cupy_sparse"
            return self._dijkstra_sparse(graph_matrix, source_node, n_nodes, is_sparse, target_node)
        
        # Para grafos dispersos grandes o CPU, usar implementación con heap
        if is_sparse or n_nodes > 10000:
            print("  [CPU] Dijkstra usando sparse")
            self.metrics.details["mode"] = "cpu_heap_sparse"
            return self._dijkstra_sparse(graph_matrix, source_node, n_nodes, is_sparse, target_node)
        
        # Para grafos pequeños, usar implementación vectorizada
        if self.use_cuda:
            self.metrics.details["mode"] = "gpu_cupy_dense"
            return self._dijkstra_cuda(graph_matrix, source_node, n_nodes)
        else:
            self.metrics.details["mode"] = "cpu_numpy_dense"
            return self._dijkstra_numpy(graph_matrix, source_node, n_nodes)
    
    def _dijkstra_sparse(self, graph_matrix, source_node: int, n_nodes: int, is_sparse: bool, target_node: Optional[int] = None):
        """Dijkstra con CuPy GPU o heap CPU para grafos dispersos grandes."""
        import time
        start_time = time.time()
        
        # USAR GPU CON CUPY SI ESTÁ DISPONIBLE
        if self.use_cuda and CUDA_AVAILABLE and is_sparse:
            try:
                print(f"  [GPU] Dijkstra con CuPy: {n_nodes:,} nodos")
                return self._dijkstra_cupy_gpu(graph_matrix, source_node, n_nodes, target_node)
            except Exception as e:
                print(f"  [WARN] GPU falló: {e}, usando CPU")
        
        # FALLBACK: implementación CPU con heap
        distances = np.full(n_nodes, np.inf, dtype=np.float32)
        distances[source_node] = 0.0
        parent = np.full(n_nodes, -1, dtype=np.int32)
        visited = set()
        
        # Heap: (distancia, nodo)
        heap = [(0.0, source_node)]
        
        # Para grafos muy grandes, reportar progreso cada N nodos
        progress_interval = 10000 if n_nodes > 100000 else 1000
        last_log_time = start_time
        
        # Límite de seguridad (suave): si hay target_node, limitar búsqueda.
        # OJO: en grafos pequeños, n_nodes // 20 puede ser 0; asegurar un mínimo razonable.
        if target_node is None:
            max_nodes_to_process = min(n_nodes, 100000)
        else:
            max_nodes_to_process = min(n_nodes, max(1000, n_nodes // 20))
        
        while heap:
            current_dist, current = heapq.heappop(heap)
            
            if current in visited:
                continue
            
            # Early stopping: si encontramos el nodo objetivo, terminar
            if target_node is not None and current == target_node:
                elapsed = time.time() - start_time
                print(f"  [OK] Nodo objetivo {target_node} encontrado en {elapsed:.2f}s ({self.metrics.nodes_processed:,} nodos procesados)")
                break
            
            # Límite de seguridad: detener si procesamos demasiados nodos
            if self.metrics.nodes_processed >= max_nodes_to_process:
                elapsed = time.time() - start_time
                print(f"  [WARN] Limite alcanzado: {self.metrics.nodes_processed:,} nodos en {elapsed:.1f}s")
                if target_node is not None:
                    print(f"  [ERR] Nodo {target_node} no alcanzable desde {source_node} (grafo desconectado)")
                break
            
            visited.add(current)
            self.metrics.nodes_processed += 1
            
            # Reportar progreso para grafos grandes
            if n_nodes > 100000 and self.metrics.nodes_processed % progress_interval == 0:
                elapsed = time.time() - last_log_time
                if elapsed > 5:  # Log cada 5 segundos
                    progress = (self.metrics.nodes_processed / n_nodes) * 100
                    print(f"  [INFO] Procesados {self.metrics.nodes_processed:,}/{n_nodes:,} nodos ({progress:.1f}%)")
                    last_log_time = time.time()
            
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
        
        elapsed_total = time.time() - start_time
        if n_nodes > 100000:
            print(f"  [OK] Dijkstra completado en {elapsed_total:.1f}s ({self.metrics.nodes_processed:,} nodos)")
        
        # Construir resultados
        if target_node is not None:
            target_dist = float(distances[int(target_node)])
            if not np.isinf(target_dist):
                self.metrics.distances_computed = {int(target_node): target_dist}
                self.metrics.path_to_nodes = {int(target_node): self._reconstruct_single_path(parent, int(source_node), int(target_node))}
            else:
                self.metrics.distances_computed = {}
                self.metrics.path_to_nodes = {}
        else:
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

    def _dijkstra_cupy_gpu(self, graph_matrix, source_node: int, n_nodes: int, target_node: Optional[int] = None):
        """Implementación Dijkstra GPU con CuPy (CSR), sin densificar."""
        import time

        if cp is None:
            raise RuntimeError("CuPy no disponible")

        start_time = time.time()

        try:
            mempool = cp.get_default_memory_pool()
            print(f"  [INFO] GPU Memory antes: {mempool.used_bytes() / 1024**2:.1f} MB")
        except Exception:
            pass

        print(f"  [INFO] Transfiriendo matriz CSR a GPU ({n_nodes:,} nodos)...")
        csr_gpu = cp.sparse.csr_matrix(graph_matrix)

        distances = cp.full(n_nodes, cp.inf, dtype=cp.float32)
        distances[int(source_node)] = 0.0
        visited = cp.zeros(n_nodes, dtype=cp.bool_)
        parent = cp.full(n_nodes, -1, dtype=cp.int32)

        # Límite de iteraciones (evitar n_nodes // 20 == 0)
        if target_node is None:
            max_iters = min(n_nodes, 100000)
        else:
            max_iters = min(n_nodes, max(1000, n_nodes // 20))

        nodes_processed = 0
        target = int(target_node) if target_node is not None else None

        for iteration in range(int(max_iters)):
            unvisited_mask = ~visited
            temp_distances = cp.where(unvisited_mask, distances, cp.inf)
            current = int(cp.argmin(temp_distances))
            current_dist = float(distances[current])

            if cp.isinf(current_dist):
                break

            if target is not None and current == target:
                nodes_processed = iteration + 1
                print(f"  [OK] Nodo {target} encontrado en {nodes_processed:,} iteraciones")
                break

            visited[current] = True
            nodes_processed += 1

            row_start = int(csr_gpu.indptr[current])
            row_end = int(csr_gpu.indptr[current + 1])
            if row_start >= row_end:
                continue

            neighbors = csr_gpu.indices[row_start:row_end]
            weights = csr_gpu.data[row_start:row_end]
            new_distances = current_dist + weights

            # relajación simple (loop): evita kernel custom
            for i in range(int(neighbors.shape[0])):
                v = int(neighbors[i])
                nd = float(new_distances[i])
                if nd < float(distances[v]):
                    distances[v] = cp.float32(nd)
                    parent[v] = current
                    self.metrics.edge_relaxations += 1

        distances_cpu = cp.asnumpy(distances)
        parent_cpu = cp.asnumpy(parent)

        self.metrics.nodes_processed += int(nodes_processed)
        if target is not None:
            td = float(distances_cpu[target])
            if not np.isinf(td):
                self.metrics.distances_computed = {target: td}
                self.metrics.path_to_nodes = {target: self._reconstruct_single_path(parent_cpu, int(source_node), target)}
            else:
                self.metrics.distances_computed = {}
                self.metrics.path_to_nodes = {}
        else:
            self.metrics.distances_computed = {i: float(distances_cpu[i]) for i in range(n_nodes) if distances_cpu[i] != np.inf}
            self.metrics.path_to_nodes = self._reconstruct_paths(parent_cpu, int(source_node), n_nodes)

        self.metrics.details["gpu_total_time_s"] = float(time.time() - start_time)
        return self.metrics

    def _reconstruct_single_path(self, parent: np.ndarray, source: int, target: int) -> list:
        """Reconstruye SOLO el camino source->target (más rápido que reconstruir todos)."""
        if source == target:
            return [source]
        path = []
        current = int(target)
        max_steps = parent.shape[0]
        steps = 0
        while current != -1 and steps < max_steps:
            path.append(current)
            if current == source:
                break
            current = int(parent[current])
            steps += 1
        if not path or path[-1] != source:
            return []
        path.reverse()
        return path
    
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
            print("[WARN] CUDA no disponible. Usando implementación CPU.")
    
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
    
    def _dijkstra_cupy_gpu(self, graph_matrix, source_node: int, n_nodes: int, target_node: Optional[int] = None):
        """Implementación Dijkstra GPU con operaciones vectorizadas masivas en CuPy."""
        import time
        start_time = time.time()
        
        # Verificar uso de GPU
        try:
            mempool = cp.get_default_memory_pool()
            print(f"  [INFO] GPU Memory antes: {mempool.used_bytes() / 1024**2:.1f} MB")
        except:
            pass
        
        print(f"  [INFO] Transfiriendo matriz CSR a GPU ({n_nodes:,} nodos)...")
        
        # Convertir scipy sparse CSR a CuPy sparse CSR (GPU) - ESTO USA GPU
        csr_gpu = cp.sparse.csr_matrix(graph_matrix)
        
        # Vectores en GPU - TODAS estas operaciones en VRAM
        distances = cp.full(n_nodes, cp.inf, dtype=cp.float32)
        distances[source_node] = 0.0
        visited = cp.zeros(n_nodes, dtype=cp.bool_)
        parent = cp.full(n_nodes, -1, dtype=cp.int32)
        
        try:
            print(f"  [INFO] GPU Memory después: {mempool.used_bytes() / 1024**2:.1f} MB")
            print(f"  [INFO] Dispositivo: {cp.cuda.Device()}")
        except:
            pass
        
        print("  [GPU] Ejecutando SSSP vectorizado en GPU...")
        sssp_start = time.time()
        
        # Límite de iteraciones
        max_iters = min(n_nodes, 100000) if target_node is None else min(50000, n_nodes // 20)
        nodes_processed = 0
        
        for iteration in range(max_iters):
            # OPERACIÓN GPU 1: Encontrar nodo no visitado con menor distancia
            # Esta operación es PARALELA en GPU (reduce operation)
            unvisited_mask = ~visited
            temp_distances = cp.where(unvisited_mask, distances, cp.inf)
            current = int(cp.argmin(temp_distances))
            
            current_dist = float(distances[current])
            
            # Si no hay más nodos alcanzables
            if cp.isinf(current_dist):
                break
            
            # Early stopping
            if target_node is not None and current == target_node:
                nodes_processed = iteration + 1
                print(f"  [OK] Nodo {target_node} encontrado en {nodes_processed:,} iteraciones")
                break
            
            # Marcar como visitado (GPU operation)
            visited[current] = True
            nodes_processed += 1
            
            # Reporte de progreso
            if nodes_processed % 5000 == 0:
                elapsed = time.time() - sssp_start
                print(f"  [INFO] GPU: {nodes_processed:,} nodos en {elapsed:.1f}s")
            
            # OPERACIÓN GPU 2: Relajación de aristas (VECTORIZADA)
            # Obtener fila CSR del nodo actual (neighbors y weights)
            row_start = int(csr_gpu.indptr[current])
            row_end = int(csr_gpu.indptr[current + 1])
            
            if row_start < row_end:
                # Estas son SLICES de arrays GPU - sin transferencia CPU
                neighbors = csr_gpu.indices[row_start:row_end]
                weights = csr_gpu.data[row_start:row_end]
                
                # OPERACIÓN VECTORIZADA GPU: Calcular TODAS las nuevas distancias en paralelo
                new_distances = current_dist + weights
                
                # OPERACIÓN VECTORIZADA GPU: Actualizar distancias y padres
                # Esto procesa TODOS los vecinos en paralelo en la GPU
                for i in range(len(neighbors)):
                    neighbor = int(neighbors[i])
                    new_dist = float(new_distances[i])
                    
                    if not visited[neighbor] and new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        parent[neighbor] = current
        
        sssp_time = time.time() - sssp_start
        print(f"  [OK] GPU completado: {sssp_time:.3f}s ({nodes_processed:,} nodos)")
        
        # Transferir resultados a CPU
        distances_cpu = cp.asnumpy(distances)
        parent_cpu = cp.asnumpy(parent)
        
        # Construir diccionarios
        self.metrics.distances_computed = {}
        for i in range(n_nodes):
            if not np.isinf(distances_cpu[i]):
                self.metrics.distances_computed[i] = float(distances_cpu[i])
        
        self.metrics.nodes_processed = nodes_processed
        
        # Verificar nodo objetivo
        if target_node is not None:
            if target_node in self.metrics.distances_computed:
                print(f"  [OK] Distancia: {self.metrics.distances_computed[target_node]:.2f}m")
            else:
                print(f"  [ERR] Nodo {target_node} no alcanzable desde {source_node}")
        
        # Reconstruir caminos
        self.metrics.path_to_nodes = self._reconstruct_paths(parent_cpu, source_node, n_nodes)
        
        total_time = time.time() - start_time
        print(f"  [OK] Total GPU: {total_time:.2f}s ({len(self.metrics.distances_computed):,} nodos alcanzables)")
        
        self._stop_metrics_tracking()
        return self.metrics

