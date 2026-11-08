from pathlib import Path

import cv2
import numpy as np


class FaceDataset:
    IMAGE_SIZE = (100, 100)

    def __init__(self, directory):
        self.directory = Path(directory)
        self.samples = []
        self.labels = []

    @classmethod
    def preprocess(cls, image):
        gray_image = cv2.equalizeHist(image)
        return cv2.resize(gray_image, cls.IMAGE_SIZE).flatten()

    def load(self):
        for person_dir in sorted(self.directory.iterdir()):
            if not person_dir.is_dir():
                continue

            for image_path in sorted(person_dir.glob("*.jpg")):
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    self.samples.append(self.preprocess(image))
                    self.labels.append(person_dir.name)

        if not self.samples:
            raise ValueError(f"ไม่พบรูปภาพ .jpg ใน Dataset: {self.directory}")

        return np.array(self.samples), np.array(self.labels)


class KNNRecognizer:
    def __init__(self, samples, labels, neighbors=3):
        if len(samples) == 0:
            raise ValueError("ไม่พบข้อมูลสำหรับฝึก KNN")
        self.samples = samples
        self.labels = labels
        self.neighbors = max(1, min(neighbors, len(samples)))

    def predict(self, sample):
        distances = np.sum((self.samples - sample) ** 2, axis=1)
        nearest = np.argsort(distances)[: self.neighbors]
        classes, votes = np.unique(self.labels[nearest], return_counts=True)
        return classes[np.argmax(votes)]


class FaceRecognitionApp:
    ROI = (800, 100, 1200, 650)

    def __init__(self, dataset_dir, camera_index=0):
        dataset = FaceDataset(dataset_dir)
        samples, labels = dataset.load()
        self.dataset = dataset
        self.recognizer = KNNRecognizer(samples, labels)
        self.camera_index = camera_index

    def run(self):
        camera = cv2.VideoCapture(self.camera_index)
        if not camera.isOpened():
            raise RuntimeError("ไม่สามารถเปิดกล้องได้")

        x1, y1, x2, y2 = self.ROI
        try:
            while True:
                success, frame = camera.read()
                if not success or frame is None:
                    print("ไม่สามารถอ่านภาพจากกล้องได้")
                    break

                height, width = frame.shape[:2]
                if not (x2 <= width and y2 <= height):
                    raise RuntimeError(
                        f"ROI {self.ROI} อยู่นอกขนาดภาพจากกล้อง ({width}x{height})"
                    )

                face = cv2.cvtColor(
                    frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY
                )
                name = self.recognizer.predict(self.dataset.preprocess(face))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    frame, str(name), (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                )
                cv2.imshow("Face Recognition", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            camera.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    FaceRecognitionApp(project_dir).run()
