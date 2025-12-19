"""Utilidades para configurar CUDA en Windows (búsqueda de DLLs).

En VS Code/PowerShell a veces `CUDA_PATH`/`PATH` no se aplican a procesos ya abiertos,
y CuPy puede fallar cargando DLLs como `cusparse*.dll` o `nvrtc*.dll`.

Este módulo añade rutas candidatas al search path del loader de DLLs ANTES de
importar CuPy.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable


def _iter_cuda_bin_candidates() -> Iterable[str]:
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        yield str(Path(cuda_path) / "bin")

    # Ruta estándar de instalación del Toolkit en Windows
    base = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
    if base.is_dir():
        # Probar versiones en orden descendente (v13.1, v13.0, ...)
        for child in sorted(base.iterdir(), reverse=True):
            if child.is_dir():
                yield str(child / "bin")


def configure_cuda_dll_search_paths() -> None:
    """Configura rutas de búsqueda de DLLs para CUDA (Windows).

    - Usa `os.add_dll_directory` (Windows/Python >= 3.8).
    - También antepone las rutas al `PATH` del proceso para compatibilidad.

    Es idempotente y segura de llamar múltiples veces.
    """

    if sys.platform != "win32":
        return

    # `PATH` puede no existir en casos raros, garantizarlo
    os.environ.setdefault("PATH", "")

    seen: set[str] = set()
    for bin_dir in _iter_cuda_bin_candidates():
        if not bin_dir:
            continue
        if bin_dir in seen:
            continue
        seen.add(bin_dir)

        if not Path(bin_dir).is_dir():
            continue

        try:
            os.add_dll_directory(bin_dir)
        except Exception:
            # En algunos entornos no está permitido; igual intentamos vía PATH
            pass

        # Anteponer para que tenga prioridad
        path_entries = os.environ["PATH"].split(";") if os.environ["PATH"] else []
        if bin_dir not in path_entries:
            os.environ["PATH"] = bin_dir + (";" + os.environ["PATH"] if os.environ["PATH"] else "")
