from pathlib import Path

from src.classifier import (
    load_dataset,
    classify_images,
)
from src.metrics import evaluate_model
from src.visualization import show_samples


def main():

    dataset_path = Path("data/Q3_Dataset")

    image_files = load_dataset(dataset_path)

    show_samples(dataset_path)

    y_true, y_pred = classify_images(dataset_path, image_files)

    evaluate_model(y_true, y_pred)


if __name__ == "__main__":
    main()