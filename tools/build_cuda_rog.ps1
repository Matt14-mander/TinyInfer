param(
    [string]$ToolkitRoot = (Join-Path $PSScriptRoot '../build-cuda-sdk-12.5.1/toolkit'),
    [string]$BuildDirectory = (Join-Path $PSScriptRoot '../build-phase54-cuda')
)
$ErrorActionPreference = 'Stop'
# Invoke with powershell/pwsh -File to keep these environment changes isolated.
# No registry, user/system PATH, installer or driver changes are performed.
$tinyinferProject = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$tinyinferToolkit = (Resolve-Path -LiteralPath $ToolkitRoot).Path
$tinyinferBuild = [IO.Path]::GetFullPath($BuildDirectory)
$tinyinferDeveloperCommand = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/2022/BuildTools/Common7/Tools/VsDevCmd.bat'
if (-not (Test-Path -LiteralPath $tinyinferDeveloperCommand)) { throw 'Visual Studio 2022 Build Tools not found' }
if (-not (Test-Path -LiteralPath (Join-Path $tinyinferToolkit 'bin/nvcc.exe'))) { throw 'Toolkit nvcc.exe not found' }
$tinyinferEnvironment = & cmd.exe /c ('"' + $tinyinferDeveloperCommand + '" -arch=x64 -host_arch=x64 -vcvars_ver=14.29 >nul && set')
if ($LASTEXITCODE -ne 0) { throw 'MSVC 14.29 environment setup failed' }
foreach ($tinyinferEntry in $tinyinferEnvironment) {
    if ($tinyinferEntry -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}
$env:CUDA_PATH = $tinyinferToolkit
$env:PATH = (Join-Path $tinyinferToolkit 'bin') + ';' + $env:PATH
$tinyinferArguments = @('-S', $tinyinferProject, '-B', $tinyinferBuild, '-G', 'Ninja',
    '-DCMAKE_BUILD_TYPE=Release', '-DTINYINFER_BUILD_TESTS=ON',
    '-DTINYINFER_BUILD_EXAMPLES=ON', '-DTINYINFER_ENABLE_NATIVE_ARCH=ON',
    '-DTINYINFER_ENABLE_METAL=OFF', '-DTINYINFER_ENABLE_CUDA=ON',
    '-DTINYINFER_CUDA_REQUIRE_DEVICE=ON', '-DCMAKE_CXX_FLAGS=/EHsc /permissive-',
    "-DCMAKE_CUDA_COMPILER=$tinyinferToolkit/bin/nvcc.exe",
    "-DCUDAToolkit_ROOT=$tinyinferToolkit", '-DCMAKE_CUDA_ARCHITECTURES=89-real')
& cmake @tinyinferArguments
if ($LASTEXITCODE -ne 0) { throw 'CUDA configure failed; use a separate build directory for a different compiler' }
& cmake --build $tinyinferBuild --parallel 8
if ($LASTEXITCODE -ne 0) { throw 'CUDA build failed' }
& ctest --test-dir $tinyinferBuild --output-on-failure
if ($LASTEXITCODE -ne 0) { throw 'CUDA/CPU tests failed' }
