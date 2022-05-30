echo on
setlocal
for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"  -latest -property installationPath`) do (
      call "%%i\VC\Auxiliary\Build\vcvarsall.bat" amd64
)

SET TOOLCHAIN=VS19
SET GENERATOR="Visual Studio 17 2022"
SET "BUILDDIR=%~dp0\build"
SET "CONFIG=Release"
set "CudaToolkitDir=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7"

cmake -T host=x64 -G%GENERATOR% -A x64 -B%BUILDDIR% -H%~dp0 -DCMAKE_TOOLCHAIN_FILE="%~dp0\toolchains\%TOOLCHAIN%.cmake" -DCMAKE_CUDA_COMPILER_TOOLKIT_ROOT="%CUDA_PATH%" -DCMAKE_CUDA_COMPILER_LIBRARY_ROOT="%CUDA_PATH%/lib/x64" -DCMAKE_BUILD_TYPE=%CONFIG%
cmake --build %BUILDDIR% --config %CONFIG%