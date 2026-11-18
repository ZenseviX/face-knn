"""Collect grayscale face-region samples from a webcam (S=save, Q=quit).

Run: python facesknn_collect.py --name nod --camera 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

# Region corresponding to (800, 100, 1200, 650) at a 1920x1080 resolution.
# Shared with recognition.py, so capture and inference use the same crop.
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


class FaceCollector:
    def __init__(self, person_name, output_dir=None, camera_index=0):
        if not person_name or not person_name.strip() or Path(person_name).name != person_name:
            raise ValueError("person_name must be a nonempty directory name")
        self.person_name = person_name
        self.output_dir = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent / person_name
        self.camera_index = camera_index

    def _next_image_number(self):
        return 1 + max(
            (int(p.stem) for p in self.output_dir.glob("*.jpg") if p.stem.isdigit()),
            default=0,
        )

    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_number = self._next_image_number()
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
                crop = frame[y1:y2, x1:x2]
                # Convert only when saving; avoid doing extra processing each frame.
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.imshow("Face Cropped", crop)
                cv2.imshow("Full Frame (S: save, Q: quit)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    filename = self.output_dir / f"{image_number}.jpg"
                    if cv2.imwrite(str(filename), cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)):
                        print(f"Saved {filename}")
                        image_number += 1
                    else:
                        print(f"Could not save {filename}")
                elif key == ord("q"):
                    break
        finally:
            camera.release()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Collect labeled face crops from webcam")
    parser.add_argument("--name", default="nod", help="Person label/directory name")
    parser.add_argument("--output", type=Path, help="Custom directory for the person's images")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    FaceCollector(args.name, args.output, args.camera).run()


if __name__ == "__main__":
    main()
