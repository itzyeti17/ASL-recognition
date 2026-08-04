$ErrorActionPreference = "Stop"
$MainScript = Join-Path $PSScriptRoot "main.py"
$VenvPython = Join-Path $env:USERPROFILE ".asl-recognizer-venv\Scripts\python.exe"

if (Test-Path $VenvPython) {
    & $VenvPython $MainScript @args
    exit $LASTEXITCODE
}

Write-Error "The project Windows environment is missing. Run ./setup.sh from WSL first."
exit 1
