"""Train, evaluate, and save the static ASL letter classifier."""

import argparse
from pathlib import Path

import joblib
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data_pipeline import EXCLUDED_LABELS, load_dataset, split_dataset


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent / "models" / "asl_classifier.joblib"
)
RANDOM_STATE = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the static ASL letter classifier."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"model output path (default: {DEFAULT_MODEL_PATH})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    features, labels = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(
        features, labels, random_state=RANDOM_STATE
    )

    classifier = make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(
            SVC(
                kernel="rbf",
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            cv=5,
            ensemble=False,
        ),
    )
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    label_order = classifier.classes_.tolist()

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")
    print(f"Accuracy:         {accuracy:.3%}\n")
    print(classification_report(y_test, predictions, labels=label_order))
    print("Confusion matrix (rows=true, columns=predicted):")
    print("Labels:", " ".join(label_order))
    print(confusion_matrix(y_test, predictions, labels=label_order))

    artifact = {
        "model": classifier,
        "labels": label_order,
        "excluded_labels": list(EXCLUDED_LABELS),
        "test_accuracy": accuracy,
        "random_state": RANDOM_STATE,
        "sklearn_version": sklearn.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)
    print(f"\nSaved classifier to {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
