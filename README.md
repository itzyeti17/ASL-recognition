# ASL Recognizer

This opens a mirrored webcam preview and uses MediaPipe to track the 21
landmarks on each visible hand.

## Run from WSL

WSL generally cannot access the built-in Windows webcam as `/dev/video0`.
`run.sh` handles that by launching the Python process on Windows, where OpenCV
can use the camera normally, while keeping the project files in WSL.

Create an isolated Windows Python environment and install the dependencies once:

```bash
./setup.sh
```

The environment is stored at `%USERPROFILE%\.asl-recognizer-venv` on Windows.

Then start it from WSL:

```bash
./run.sh
```

Use another camera with `./run.sh --camera 1`. Press **Q**, **Esc**, or
**Ctrl+C** to quit.

## Native Linux

Install the dependencies, download the hand-landmark model, and use the same
launcher:

```bash
sudo apt install python3-venv libgles2
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p models
curl -L \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task \
  -o models/hand_landmarker.task
./run.sh
```

Use `./run.sh --camera 1` to select a different camera.
