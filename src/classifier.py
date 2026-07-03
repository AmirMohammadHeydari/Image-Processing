from pathlib import Path

import cv2
import numpy as np


RED = np.array([255, 0, 0])
BLUE = np.array([0, 0, 255])


def load_dataset(dataset_path: Path):
    """
    Returns all image filenames.
    """
    return sorted([file.name for file in dataset_path.iterdir()])


def get_true_label(filename: str):
    """
    Determine ground-truth label from filename.
    """
    if filename.lower().startswith("c"):
        return "Chelsea"

    return "Manchester"


def average_color(image_path: Path):

    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image.mean(axis=(0, 1))


def predict_label(mean_color):

    red_distance = np.linalg.norm(mean_color - RED)
    blue_distance = np.linalg.norm(mean_color - BLUE)

    if red_distance >= blue_distance:
        return "Chelsea"

    return "Manchester"


def classify_images(dataset_path, image_files):

    y_true = []
    y_pred = []

    for filename in image_files:

        image_path = dataset_path / filename

        mean_color = average_color(image_path)

        prediction = predict_label(mean_color)

        y_true.append(get_true_label(filename))
        y_pred.append(prediction)

    return y_true, y_pred