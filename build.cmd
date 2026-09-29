@echo off
rem Build PRender.
rem   build.cmd            renderer (CMake/Ninja, Release) + WinUI 3 app
rem   build.cmd renderer   renderer only
rem   build.cmd ui         WinUI 3 app only
rem   build.cmd test       renderer + run unit/integration tests
rem   build.cmd gpu        prender_gpu.dll (SYCL path tracer; needs Intel oneAPI DPC++)
setlocal
set "TARGET=%~1"
if "%TARGET%"=="" set "TARGET=all"
set "ROOT=%~dp0"

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%i"
if not defined VSINSTALL (
  echo Visual Studio with the C++ x64 tools was not found.
  exit /b 1
)

if /i "%TARGET%"=="ui" goto :ui
if /i "%TARGET%"=="gpu" goto :gpu

call "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cmake -S "%ROOT%." -B "%ROOT%build" -G Ninja -DCMAKE_BUILD_TYPE=Release || exit /b 1
cmake --build "%ROOT%build" || exit /b 1
if /i "%TARGET%"=="test" (
  "%ROOT%build\bin\prender_tests.exe" || exit /b 1
  goto :eof
)
if /i "%TARGET%"=="renderer" goto :eof

:ui
rem WinUI 3 needs Visual Studio's MSBuild (the dotnet CLI lacks the PRI/AppX build tasks).
"%VSINSTALL%\MSBuild\Current\Bin\amd64\MSBuild.exe" "%ROOT%PRender.sln" -restore -m -v:minimal ^
  -p:Configuration=Release -p:Platform=x64 || exit /b 1
echo.
echo UI: %ROOT%src\ui\PRenderUI\bin\x64\Release\net8.0-windows10.0.19041.0\win-x64\PRenderUI.exe
goto :eof

:gpu
rem The GPU module is a separate DLL built with Intel's DPC++ compiler; prender.exe loads it at
rem runtime and falls back to the CPU when it (or its runtime) is absent.
if not defined ONEAPI_ROOT set "ONEAPI_ROOT=%ProgramFiles(x86)%\Intel\oneAPI"
set "ONEAPI_BIN=%ONEAPI_ROOT%\compiler\latest\bin"
if not exist "%ONEAPI_BIN%\icx-cl.exe" (
  echo Intel oneAPI DPC++ compiler not found under "%ONEAPI_ROOT%".
  exit /b 1
)
call "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set "PATH=%ONEAPI_BIN%;%PATH%"
set "LIB=%ONEAPI_ROOT%\compiler\latest\lib;%LIB%"
if not exist "%ROOT%build\gpu" mkdir "%ROOT%build\gpu"
if not exist "%ROOT%build\bin" mkdir "%ROOT%build\bin"
rem Ahead-of-time code for Arc (DG2, override with GPU_AOT_DEVICE) with the large register file
rem (the megakernel spills otherwise: ~1.4-1.8x slower), plus generic SPIR-V JIT-compiled for
rem any other GPU.
if not defined GPU_AOT_DEVICE set "GPU_AOT_DEVICE=dg2"
set "PATH=%ONEAPI_ROOT%\ocloc\latest\bin;%PATH%"
icx-cl -fsycl -fsycl-targets=spir64_gen,spir64 ^
  -Xsycl-target-backend=spir64_gen "-device %GPU_AOT_DEVICE% -options -ze-opt-large-register-file" ^
  /O2 /EHsc /std:c++17 /MD /W3 /LD /I "%ROOT%src" "%ROOT%src\gpu\prender_gpu.cpp" ^
  /Fo"%ROOT%build\gpu\prender_gpu.obj" /Fe"%ROOT%build\bin\prender_gpu.dll" || exit /b 1
rem Ship the SYCL / Unified Runtime DLLs next to prender.exe.
for %%f in (sycl8.dll ur_loader.dll ur_adapter_level_zero.dll ur_adapter_opencl.dll ur_win_proxy_loader.dll libmmd.dll svml_dispmd.dll) do (
  copy /y "%ONEAPI_BIN%\%%f" "%ROOT%build\bin\" >nul || exit /b 1
)
for %%f in ("%ONEAPI_ROOT%\umf\latest\bin\umf.dll" "%ONEAPI_ROOT%\tcm\latest\bin\tcm.dll" "%ONEAPI_ROOT%\tcm\latest\bin\libhwloc-15.dll") do (
  if exist %%f copy /y %%f "%ROOT%build\bin\" >nul
)
echo GPU module: %ROOT%build\bin\prender_gpu.dll
endlocal
