import cv2
import os

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError('ไม่สามารถเปิดกล้องได้')

name = 'nod'
os.makedirs(name, exist_ok=True)
j = 1

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print('ไม่สามารถอ่านภาพจากกล้องได้')
            break

        cv2.rectangle(frame, (800, 100), (1200, 650), (0, 0, 255), 2)
        face = cv2.cvtColor(frame[100:650, 800:1200], cv2.COLOR_BGR2GRAY)
        cv2.imshow('Face Cropped', face)
        cv2.imshow('Full Frame', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            cv2.imwrite(os.path.join(name, str(j) + '.jpg'), face)
            j += 1
        elif key == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()