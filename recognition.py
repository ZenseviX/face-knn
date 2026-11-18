"""Basic webcam region recognition using KNN on normalized grayscale pixels.

Run: python recognition.py --dataset . --neighbors 3
This is an educational pixel-based classifier, not biometric verification.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
ROI_FRACTIONS = (800 / 1920, 100 / 1080, 1200 / 1920, 650 / 1080)


def crop_bounds(frame):
    height, width = frame.shape[:2]
    if height < 2 or width < 2:
        raise ValueError("Camera frame is too small")
    x1, y1, x2, y2 = (round(v * dim) for v, dim in zip(
        ROI_FRACTIONS, (width, height, width, height)
    ))
    x1, y1 = min(x1, width - 1), min(y1, height - 1)
    x2, y2 = max(x1 + 1, min(x2, width)), max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


class FaceDataset:
    IMAGE_SIZE = (100, 100)

    def __init__(self, directory):
        self.directory = Path(directory)

    @classmethod
    def preprocess(cls, image):
        if image is None or image.size == 0:
            raise ValueError("Image is empty")
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim != 2 or image.dtype != np.uint8:
            raise ValueError("Expected an 8-bit grayscale or BGR image")
        # Resize first: equalization on fewer pixels is cheaper and consistent
        # for both training and webcam inputs.
        resized = cv2.resize(image, cls.IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        return cv2.equalizeHist(resized).reshape(-1).astype(np.float32) / 255.0

    def load(self):
        if not self.directory.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.directory}")
        samples, labels = [], []
        for person_dir in sorted(self.directory.iterdir()):
            if not person_dir.is_dir() or person_dir.name.startswith("."):
                continue
            for image_path in sorted(person_dir.iterdir()):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Skipping unreadable image: {image_path}")
                    continue
                samples.append(self.preprocess(image))
                labels.append(person_dir.name)
        if not samples:
            raise ValueError(f"No readable images under person folders in: {self.directory}")
        return np.stack(samples), np.asarray(labels)


class KNNRecognizer:
    def __init__(self, samples, labels, neighbors=3):
        samples = np.asarray(samples, dtype=np.float32)
        labels = np.asarray(labels)
        if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] == 0:
            raise ValueError("samples must be a nonempty 2D array")
        if labels.ndim != 1 or len(labels) != len(samples):
            raise ValueError("labels must match sample count")
        if not np.isfinite(samples).all():
            raise ValueError("samples must be finite")
        if isinstance(neighbors, bool) or not isinstance(neighbors, (int, np.integer)) or neighbors < 1:
            raise ValueError("neighbors must be a positive integer")
        self.samples = np.ascontiguousarray(samples)
        self.labels = labels
        self.neighbors = min(int(neighbors), len(samples))
        self.sample_norms = np.einsum("ij,ij->i", self.samples, self.samples)

    def predict(self, sample):
        query = np.asarray(sample, dtype=np.float32)
        if query.shape != (self.samples.shape[1],) or not np.isfinite(query).all():
            raise ValueError("Sample has invalid shape or values")
        # Accumulate in float32, avoiding the original uint8 subtraction/square
        # overflow. O(N) temporary storage, not O(N * pixels).
        distances = self.sample_norms + float(query @ query) - 2.0 * (self.samples @ query)
        np.maximum(distances, 0.0, out=distances)
        nearest = np.argpartition(distances, self.neighbors - 1)[:self.neighbors]
        # Use deterministic nearest selection at the kth-distance boundary.
        cutoff = distances[nearest].max()
        if np.count_nonzero(distances <= cutoff) > self.neighbors:
            nearest = np.lexsort((np.arange(len(distances)), distances))[:self.neighbors]
        else:
            nearest = nearest[np.lexsort((nearest, distances[nearest]))]
        classes, votes = np.unique(self.labels[nearest], return_counts=True)
        return classes[np.argmax(votes)]


class FaceRecognitionApp:
    def __init__(self, dataset_dir, camera_index=0, neighbors=3):
        self.dataset = FaceDataset(dataset_dir)
        samples, labels = self.dataset.load()
        self.recognizer = KNNRecognizer(samples, labels, neighbors)
        self.camera_index = camera_index

    def run(self):
        camera = cv2.VideoCapture(self.camera_index)
        try:
            if not camera.isOpened():
                raise RuntimeError(f"Cannot open camera {self.camera_index}")
            while True:
                ok, frame = camera.read()
                if not ok or frame is None:
                    print("Camera frame unavailable; stopping")
                    break
                x1, y1, x2, y2 = crop_bounds(frame)
                face = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                name = self.recognizer.predict(self.dataset.preprocess(face))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, str(name), (x1, max(18, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Face Recognition (Q: quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            camera.release()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="KNN webcam region recognizer")
    parser.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--neighbors", type=int, default=3)
    args = parser.parse_args()
    FaceRecognitionApp(args.dataset, args.camera, args.neighbors).run()


if __name__ == "__main__":
    main()
