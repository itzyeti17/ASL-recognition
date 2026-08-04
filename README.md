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

Install the dependency and use the same launcher:

```bash
python3 -m pip install -r requirements.txt
./run.sh
```
