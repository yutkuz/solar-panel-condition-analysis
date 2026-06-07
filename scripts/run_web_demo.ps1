param(
    [int]$Port = 8000,
    [switch]$EagerLoad
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$env:PYTHONNOUSERSITE = "1"

if ($EagerLoad) {
    $env:DEMO_EAGER_LOAD = "1"
}

$condaPython = Join-Path $env:USERPROFILE "anaconda3\envs\solar-panel-cls\python.exe"
$python = if (Test-Path $condaPython) { $condaPython } else { "python" }

& $python -m uvicorn demo_web.app:app --host 127.0.0.1 --port $Port
