from pathlib import Path

import cv2
import numpy as np


IMAGE_SIZE = (100, 100)
ROI = (800, 100, 1200, 650)
DEFAULT_K = 3


def knn(X, y, z, k=DEFAULT_K):
    if len(X) == 0:
        raise ValueError("ไม่พบข้อมูลสำหรับฝึก KNN")

    k = max(1, min(k, len(X)))
    distances = np.sum((X - z) ** 2, axis=1)
    nearest = np.argsort(distances)[:k]
    classes, votes = np.unique(y[nearest], return_counts=True)
    return classes[np.argmax(votes)]


def preprocess(image):
    image = cv2.equalizeHist(image)
    return cv2.resize(image, IMAGE_SIZE).flatten()


def load_dataset(dataset_dir):
    samples = []
    labels = []

    for person_dir in sorted(dataset_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        for image_path in sorted(person_dir.glob("*.jpg")):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                samples.append(preprocess(image))
                labels.append(person_dir.name)

    if not samples:
        raise ValueError(f"ไม่พบรูปภาพ .jpg ใน Dataset: {dataset_dir}")

    return np.array(samples), np.array(labels)


def main():
    dataset_dir = Path(__file__).resolve().parent
    X, y = load_dataset(dataset_dir)
    print("Shape ของข้อมูล X:", X.shape)
    print("รายชื่อใน Dataset:", np.unique(y))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("ไม่สามารถเปิดกล้องได้")

    x1, y1, x2, y2 = ROI

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("ไม่สามารถอ่านภาพจากกล้องได้")
                break

            height, width = frame.shape[:2]
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise RuntimeError(
                    f"ROI {ROI} อยู่นอกขนาดภาพจากกล้อง ({width}x{height})"
                )

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            face = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            name = knn(X, y, preprocess(face))

            cv2.putText(
                frame,
                str(name),
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
