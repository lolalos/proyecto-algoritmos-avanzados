@echo off
setlocal EnableExtensions

set "NO_PAUSE=0"
if /I "%~1"=="--no-pause" set "NO_PAUSE=1"

REM Script único para ejecutar el backend.
REM - Crea/usa venv313 si existe
REM - Instala dependencias (con fallback si CuPy/CUDA no está disponible)
REM - Arranca FastAPI (python main.py)

cd /d "%~dp0"

for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"
set "VENV_DIR=%PROJECT_ROOT%\venv313"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

echo ========================================
echo   INICIANDO BACKEND (AUTO CPU/CUDA)
echo ========================================
echo.

REM Resolver launcher de Python
set "HAVE_PY_LAUNCHER=0"
py --version >nul 2>&1
if not errorlevel 1 set "HAVE_PY_LAUNCHER=1"

REM Verificar/crear entorno virtual (sin bloques con paréntesis aquí)
if exist "%VENV_PY%" goto VENV_OK
echo [INFO] Entorno virtual no encontrado: %VENV_DIR%
echo [INFO] Creando entorno virtual...
call :CREATE_VENV
if errorlevel 1 exit /b 1
if not exist "%VENV_PY%" (
    echo [ERROR] No se pudo crear el entorno virtual.
    pause
    exit /b 1
)

:VENV_OK
echo [OK] Entorno virtual listo: %VENV_DIR%

REM Activar entorno
call "%VENV_DIR%\Scripts\activate.bat" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] No se pudo activar el entorno virtual.
    pause
    exit /b 1
)

python -m pip install --upgrade pip >nul 2>&1

REM Instalar dependencias: primero intenta requirements.txt completo.
echo [INFO] Verificando dependencias...
python -c "import fastapi, uvicorn, numpy, scipy" >nul 2>&1
if not errorlevel 1 goto DEPS_OK

echo [INFO] Instalando dependencias (requirements.txt)...
pip install -r "%~dp0requirements.txt"
if not errorlevel 1 goto DEPS_OK

echo [WARNING] Fallo la instalacion completa (probable CuPy/CUDA).
call :INSTALL_BASE_DEPS

:DEPS_OK
echo [OK] Dependencias listas

REM Detectar CUDA disponible (sin fallar si no existe CuPy)
set "CUDA_OK=0"
python -c "import sys; import cupy as cp; sys.exit(0 if cp.cuda.is_available() else 1)" >nul 2>&1
if not errorlevel 1 set "CUDA_OK=1"

echo.
echo ========================================
echo   SERVIDOR BACKEND
echo ========================================
if "%CUDA_OK%"=="1" (
    echo   Modo: CUDA disponible (se puede usar use_cuda=true^)
) else (
    echo   Modo: CPU (CUDA no disponible / CuPy no instalado)
)
echo   URL:  http://localhost:8000
echo   Docs: http://localhost:8000/docs
echo ========================================
echo.

python "%~dp0main.py"

echo.
if "%NO_PAUSE%"=="0" pause

goto :eof

:CREATE_VENV
REM Crear venv preferentemente con Python 3.13 si existe, si no con el default
if "%HAVE_PY_LAUNCHER%"=="1" (
    py -3.13 --version >nul 2>&1
    if not errorlevel 1 (
        py -3.13 -m venv "%VENV_DIR%"
    ) else (
        py -m venv "%VENV_DIR%"
    )
) else (
    python --version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] No se encontro Python (ni el launcher 'py').
        echo [INFO] Instala Python y vuelve a ejecutar este script.
        pause
        exit /b 1
    )
    python -m venv "%VENV_DIR%"
)

exit /b 0

:INSTALL_BASE_DEPS
echo [INFO] Instalando dependencias base (sin CuPy)...
pip install fastapi "uvicorn[standard]" python-multipart numpy scipy pandas networkx geopandas pyogrio shapely pyproj pydantic python-dotenv psutil numba "dask[distributed]"

echo [INFO] Intentando instalar CuPy (opcional)...
where nvidia-smi >nul 2>&1
if errorlevel 1 exit /b 0
pip install cupy-cuda13x >nul 2>&1
exit /b 0
