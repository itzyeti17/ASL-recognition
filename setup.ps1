$ErrorActionPreference = "Stop"
$VenvDir = Join-Path $env:USERPROFILE ".asl-recognizer-venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$Requirements = Join-Path $PSScriptRoot "requirements.txt"
$ModelDir = Join-Path $PSScriptRoot "models"
$HandModel = Join-Path $ModelDir "hand_landmarker.task"
$HandModelUrl = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating the project Windows environment at $VenvDir"
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv $VenvDir
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv $VenvDir
    }
    else {
        Write-Error "Windows Python is not installed."
        exit 1
    }
}

Write-Host "Installing project dependencies..."
& $VenvPython -m pip install --upgrade pip
# MediaPipe needs the contrib build. Removing both variants first prevents two
# packages from trying to provide the same cv2 module in an existing setup.
& $VenvPython -m pip uninstall --yes opencv-python opencv-contrib-python
& $VenvPython -m pip install -r $Requirements

if (-not (Test-Path $HandModel)) {
    Write-Host "Downloading the MediaPipe hand-landmark model..."
    New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
    Invoke-WebRequest -Uri $HandModelUrl -OutFile $HandModel
}

Write-Host "Setup complete. Start the app from WSL with ./run.sh"
