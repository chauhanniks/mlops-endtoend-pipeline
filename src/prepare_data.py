"""Export the demo dataset as a file that DVC can version."""
import os
from pathlib import Path

from sklearn.datasets import load_breast_cancer

DATA_PATH = Path(os.getenv("DATA_PATH", "data/breast_cancer.csv"))


def prepare_data(output_path: Path = DATA_PATH) -> Path:
    """Save the scikit-learn demo dataset, including its target column."""
    dataset = load_breast_cancer(as_frame=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.frame.to_csv(output_path, index=False)
    print(f"Prepared {len(dataset.frame)} rows at {output_path}")
    return output_path


if __name__ == "__main__":
    prepare_data()
