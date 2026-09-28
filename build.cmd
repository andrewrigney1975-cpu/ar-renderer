@echo off
rem Build PRender.
rem   build.cmd            renderer (CMake/Ninja, Release) + WinUI 3 app
rem   build.cmd renderer   renderer only
rem   build.cmd ui         WinUI 3 app only
rem   build.cmd test       renderer + run unit/integration tests
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
endlocal
