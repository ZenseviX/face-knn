from pathlib import Path

import cv2


class FaceCollector:
    def __init__(self, person_name, output_dir=None, camera_index=0):
        project_dir = Path(__file__).resolve().parent
        self.output_dir = Path(output_dir or project_dir / person_name)
        self.camera_index = camera_index
        self.person_name = person_name

    def _next_image_number(self):
        numbers = []
        for image_path in self.output_dir.glob("*.jpg"):
            if image_path.stem.isdigit():
                numbers.append(int(image_path.stem))
        return max(numbers, default=0) + 1

    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_number = self._next_image_number()
        camera = cv2.VideoCapture(self.camera_index)

        if not camera.isOpened():
            raise RuntimeError("ไม่สามารถเปิดกล้องได้")

        try:
            while True:
                success, frame = camera.read()
                if not success or frame is None:
                    print("ไม่สามารถอ่านภาพจากกล้องได้")
                    break

                face = frame[100:650, 800:1200]
                cv2.rectangle(frame, (800, 100), (1200, 650), (0, 0, 255), 2)
                cv2.imshow("Face Cropped", face)
                cv2.imshow("Full Frame", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    image_path = self.output_dir / f"{image_number}.jpg"
                    if cv2.imwrite(str(image_path), cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)):
                        print(f"บันทึก: {image_path}")
                        image_number += 1
                elif key == ord("q"):
                    break
        finally:
            camera.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    FaceCollector("nod").run()