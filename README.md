# ASL Recognizer

ASL Recognizer is an American Sign Language project for recognizing static
fingerspelled letters. The application opens a mirrored webcam preview, uses
MediaPipe to detect up to two hands, and classifies their 21 landmarks with a
trained support vector machine (SVM).

> [!NOTE]
> The classifier recognizes 24 static letters. It does not recognize `J` and
> `Z` because those letters require motion.

## Features

- Live, mirrored webcam preview
- MediaPipe hand detection and tracking
- Landmark and hand-connection overlays
- Left/right handedness labels
- Confidence-filtered static-letter predictions
- Seven-frame prediction smoothing to reduce flicker
- Keyboard capture for building words from recognized letters
- Configurable camera index
- Dataset inspection and landmark preprocessing utilities

## Run from WSL

WSL generally cannot access a built-in Windows webcam as `/dev/video0`.
`run.sh` detects WSL and launches the application with Windows Python, allowing
OpenCV to access the camera while the project files remain in WSL.

Create the isolated Windows Python environment, install the dependencies,
download the MediaPipe hand-landmark model, and train the letter classifier:

```bash
./setup.sh
```

The environment is stored at `%USERPROFILE%\.asl-recognizer-venv` on Windows.
Setup requires Windows Python and an internet connection.

Start the application from WSL:

```bash
./run.sh
```

## Run on native Linux

Install the system packages, create a virtual environment, install the Python
dependencies, and download the MediaPipe model:

```bash
sudo apt install python3-venv libgles2
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p models
curl -L \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task \
  -o models/hand_landmarker.task
python train_model.py
./run.sh
```

## Use the interface

Hold a static ASL letter in the camera view. A hand card shows the tracked
landmarks, handedness, predicted letter, and confidence. The header shows the
current stabilized prediction, while the bottom panel holds the text you have
captured.

Predictions are smoothed across recent frames to reduce flicker. A letter must
meet the confidence threshold before it can be displayed or captured. When two
hands have confident predictions, **Enter** captures the one with the higher
confidence.

Keyboard controls:

- **Enter** captures the currently displayed letter.
- **Space** adds a gap between words. Repeated spaces are ignored.
- **Backspace** removes the last captured character.
- **C** clears all captured text.
- **Q**, **Esc**, or **Ctrl+C** quits.

The default camera index is `0`. Select another camera with:

```bash
./run.sh --camera 1
```

Predictions must reach 65% confidence by default. Choose another threshold from
`0` to `1` with:

```bash
./run.sh --confidence 0.75
```

Both options can be combined:

```bash
./run.sh --camera 1 --confidence 0.75
```

Captured text remains on screen for the current session; it is not written to
a file.

## Dataset

The included dataset is stored at `data/asl_landmarks_final.csv`. It contains
2,203 examples and 64 columns:

- 63 numeric values (`x0`, `y0`, `z0` through `x20`, `y20`, `z20`) representing
  the 21 MediaPipe hand landmarks
- One `label` column containing the corresponding letter or command

The dataset includes the letters **A-Z** as well as `del` and `space`. Its
source and license should be recorded here before the dataset is redistributed.

## Inspect and prepare the data

Run these commands in a Python environment where `requirements.txt` has been
installed.

Inspect the first rows, dimensions, columns, and label counts:

```bash
python inspect_data.py
```

Normalize the landmark data and create a reproducible 80/20 training/test
split:

```bash
python prepare_data.py
```

The preparation script:

1. Excludes `J` and `Z` because those signs require motion.
2. Excludes the non-letter labels `del` and `space`.
3. Moves every hand relative to its wrist landmark.
4. Scales each hand using its largest 2D distance from the wrist.
5. Creates a stratified 80/20 training/test split using random seed `42`.

The script currently prints the resulting shapes and a normalized example. It
does not save the processed arrays or train a classifier.

## Train and evaluate the classifier

Training uses the same normalization code as live prediction. The training
script creates a reproducible, stratified 80/20 split; trains a scaled,
class-balanced SVM; calibrates its probability estimates; prints an accuracy
report and confusion matrix; and saves the result:

```bash
python train_model.py
```

The default output is `models/asl_classifier.joblib`. On the current dataset,
the fixed held-out test split reaches approximately **92.7% accuracy**. The
most difficult distinctions in that split are among visually similar letters,
especially `M`/`N` and `R`/`U`/`V`.

You can select another output path if needed:

```bash
python train_model.py --output models/another_classifier.joblib
```

For WSL webcam use, prefer `./setup.sh`: it trains the model with the same
Windows Python environment that runs the application, avoiding saved-model
compatibility problems between scikit-learn versions.

## Project structure

```text
main.py                 Live hand tracking and static-letter recognition
data_pipeline.py        Shared loading, normalization, and splitting logic
inspect_data.py         Dataset summary utility
prepare_data.py         Landmark normalization and dataset split
train_model.py          Classifier training, evaluation, and saving
requirements.txt        Python dependencies
run.sh / run.ps1        WSL, Linux, and Windows launchers
setup.sh / setup.ps1    WSL/Windows setup scripts
data/                    Landmark dataset
models/                  Downloaded MediaPipe model (not tracked by Git)
```

## Current limitations

- Motion-based letters `J` and `Z` are not recognized.
- The preprocessing output exists only while the script is running.
- Captured text is kept only for the current webcam session.
- Held-out dataset accuracy may be higher than accuracy on new people, camera
  angles, lighting conditions, and backgrounds.
- Webcam behavior depends on the available camera and operating-system drivers.

The next major improvements are to test the classifier with live webcam data,
collect more diverse samples for commonly confused letters, and add temporal
recognition for `J` and `Z`.
