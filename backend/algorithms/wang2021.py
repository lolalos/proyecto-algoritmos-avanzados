"""Algoritmo Wang et al. (2021).

En esta base de código, Wang2021 se implementa como un *scheduler por particiones*
(multi-colas por partición) sobre una matriz de adyacencia (idealmente CSR).

Nota: el resultado de distancia debe coincidir con otros métodos (son caminos mínimos),
pero las métricas internas pueden diferir por la disciplina de colas y el particionado.
"""

import numpy as np
from typing import Dict, Optional, Any

from .base import ShortestPathAlgorithm, AlgorithmMetrics
from scipy import sparse

from .cuda_env import configure_cuda_dll_search_paths

configure_cuda_dll_search_paths()

try:
    import cupy as cp
    import dask.array as da
    from dask import delayed
    try:
        test = cp.array([1, 2, 3])
        CUDA_AVAILABLE = True
        DASK_AVAILABLE = True
        del test
    except:
        CUDA_AVAILABLE = False
        DASK_AVAILABLE = False
except ImportError:
    CUDA_AVAILABLE = False
    DASK_AVAILABLE = False
    cp = None
    da = None


class Wang2021Algorithm(ShortestPathAlgorithm):
    """
    Algoritmo Wang et al. (2021) GPU + DASK OPTIMIZADO.
    Particionamiento paralelo con procesamiento GPU distribuido.
    
    Características:
    - Particionamiento del grafo en subgrafos
    - Procesamiento paralelo de particiones
    - Fusión eficiente de resultados
    """
    
    def __init__(self, use_cuda: bool = True, num_partitions: int = 4):
        super().__init__("Wang2021", use_cuda=use_cuda and CUDA_AVAILABLE)
        self.num_partitions = num_partitions
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
        Implementación con particionamiento de grafo.
        
        Proceso:
        1. Particionar el grafo en subgrafos
        2. Procesar cada partición en paralelo
        3. Fusionar resultados con nodos frontera
        """
        self._start_metrics_tracking()
        self.metrics.details = {"num_partitions": int(self.num_partitions)}
        
        n_nodes = int(graph_matrix.shape[0])
        is_sparse = bool(sparse.issparse(graph_matrix))
        
        # Metodología Wang2021 aquí: scheduler por particiones (CPU). Si CUDA está disponible,
        # se deja explícito en `details` que el modo actual es este scheduler (no densifica CSR).
        self.metrics.details["mode"] = "partition_scheduler"
        if self.use_cuda and CUDA_AVAILABLE:
            self.metrics.details["cuda_available"] = True
            self.metrics.details["note"] = "scheduler por particiones (CPU) sobre CSR; sin kernel GPU para CSR en esta implementación"
        else:
            self.metrics.details["cuda_available"] = False

        return self._wang_partition_scheduler(
            graph_matrix,
            int(source_node),
            n_nodes,
            is_sparse,
            target_node,
        )
    
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
    
    def _wang_partition_scheduler(self, graph_matrix, source_node: int, n_nodes: int, is_sparse: bool, target_node: Optional[int] = None):
        """Wang2021: Scheduler por particiones.

        Idea:
        - Divide el grafo en particiones.
        - Mantiene una cola (heap) por partición.
        - Procesa particiones activas (puede re-procesar nodos si su distancia mejora).

        Esto NO es el mismo flujo que Dijkstra con un heap global, y por eso
        las métricas (nodos/relajaciones/tiempo) suelen diferir.
        """
        import heapq
        from scipy import sparse as sp

        csr = graph_matrix.tocsr() if is_sparse else sp.csr_matrix(graph_matrix)

        num_partitions = max(int(self.num_partitions), 1)
        partition_size = max(int(np.ceil(n_nodes / num_partitions)), 1)
        batch_size = 256

        self.metrics.details["partition_size"] = int(partition_size)
        self.metrics.details["partition_batch"] = int(batch_size)

        def part_id(node: int) -> int:
            pid = int(node // partition_size)
            if pid < 0:
                return 0
            if pid >= num_partitions:
                return int(num_partitions - 1)
            return pid

        # Importante: usar float64 para que las llaves del heap (Python float)
        # coincidan con `dist[u]` sin pérdidas por redondeo (float32 puede hacer
        # que casi todas las entradas parezcan "stale").
        dist = np.full(n_nodes, np.inf, dtype=np.float64)
        parent = np.full(n_nodes, -1, dtype=np.int32)
        dist[source_node] = 0.0

        heaps = [[] for _ in range(num_partitions)]
        pid0 = part_id(source_node)
        heaps[pid0].append((0.0, int(source_node)))

        # Scheduler: prioriza particiones por el menor candidato (heap head)
        active_pq = []
        in_active = [False] * num_partitions
        version = [0] * num_partitions

        def schedule(pid: int) -> None:
            if not heaps[pid]:
                return
            version[pid] += 1
            in_active[pid] = True
            heapq.heappush(active_pq, (float(heaps[pid][0][0]), int(pid), int(version[pid])))

        schedule(pid0)

        partitions_touched = set([pid0])
        partition_pops = [0] * num_partitions
        intra_relax = 0
        cross_relax = 0

        target = int(target_node) if target_node is not None else None

        # Límite para evitar explosión en grafos enormes.
        # Para búsquedas con target en grafos grandes, el límite anterior (250k) era muy agresivo
        # y podía cortar antes de alcanzar el objetivo.
        if target is not None:
            max_pops = int(min(max(2_000_000, n_nodes // 5), 5_000_000))
        else:
            max_pops = int(min(max(500_000, n_nodes // 2), 10_000_000))
        pops = 0

        self.metrics.details["scheduler"] = "min_partition_head"

        while active_pq and pops < max_pops:
            key, pid, ver = heapq.heappop(active_pq)
            pid = int(pid)
            ver = int(ver)

            # entrada stale
            if not in_active[pid] or ver != version[pid] or not heaps[pid]:
                continue

            in_active[pid] = False
            heap = heaps[pid]

            local_pops = 0
            while heap and pops < max_pops and local_pops < batch_size:
                du, u = heapq.heappop(heap)
                u = int(u)
                # Entrada stale: no usar igualdad exacta (puede fallar por precisión).
                if float(du) > float(dist[u]) + 1e-9:
                    continue

                self.metrics.nodes_processed += 1
                pops += 1
                local_pops += 1
                partition_pops[pid] += 1

                # Early-stop: si target ya está asentado y no hay ninguna cola con prioridad menor
                if target is not None and u == target:
                    best_other = None
                    for h in heaps:
                        if h:
                            cand = float(h[0][0])
                            best_other = cand if best_other is None else min(best_other, cand)
                    if best_other is None or best_other >= float(dist[target]):
                        heap.clear()
                        active_pq.clear()
                        for j in range(num_partitions):
                            in_active[j] = False
                        break

                row_start = int(csr.indptr[u])
                row_end = int(csr.indptr[u + 1])
                if row_start >= row_end:
                    continue

                nbrs = csr.indices[row_start:row_end]
                wts = csr.data[row_start:row_end]

                for i in range(len(nbrs)):
                    v = int(nbrs[i])
                    w = float(wts[i])
                    if w <= 0:
                        continue
                    nd = float(du) + w
                    if nd < float(dist[v]) - 1e-12:
                        dist[v] = float(nd)
                        parent[v] = u
                        self.metrics.edge_relaxations += 1

                        if part_id(u) == part_id(v):
                            intra_relax += 1
                        else:
                            cross_relax += 1

                        pv = part_id(v)
                        # Si la partición ya tiene trabajo, solo forzamos re-schedule
                        # si el mínimo realmente disminuyó.
                        old_min = float(heaps[pv][0][0]) if heaps[pv] else None
                        heapq.heappush(heaps[pv], (nd, v))
                        partitions_touched.add(pv)
                        # Programar partición destino (si ya está activa, refrescar prioridad si bajó el mínimo)
                        new_min = float(heaps[pv][0][0])
                        if (not in_active[pv]) or (old_min is None) or (new_min < old_min - 1e-12):
                            schedule(pv)

            # Re-activar la partición si aún queda trabajo
            if heap and pops < max_pops:
                schedule(pid)

        self.metrics.details["partitions_touched"] = int(len(partitions_touched))
        self.metrics.details["partition_pops"] = partition_pops
        self.metrics.details["intra_partition_relaxations"] = int(intra_relax)
        self.metrics.details["cross_partition_relaxations"] = int(cross_relax)
        self.metrics.details["max_pops"] = int(max_pops)
        self.metrics.details["max_pops_reached"] = bool(pops >= max_pops)

        # Guardar resultados: si hay target, solo target
        if target is not None:
            td = float(dist[target])
            if not np.isinf(td):
                self.metrics.distances_computed = {target: td}
                self.metrics.path_to_nodes = {target: self._reconstruct_single_path(parent, source_node, target)}
            else:
                self.metrics.distances_computed = {}
                self.metrics.path_to_nodes = {}
        else:
            self.metrics.distances_computed = {i: float(dist[i]) for i in range(n_nodes) if not np.isinf(dist[i])}
            self.metrics.path_to_nodes = self._reconstruct_paths(parent, source_node, n_nodes)

        self._stop_metrics_tracking()
        return self.metrics

    def _reconstruct_single_path(self, parent: np.ndarray, source: int, target: int) -> list:
        if source == target:
            return [source]
        path = []
        cur = int(target)
        max_steps = parent.shape[0]
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
