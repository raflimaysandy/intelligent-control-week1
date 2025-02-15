import cv2
import numpy as np

# Inisialisasi kamera
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rentang warna dalam HSV
    color_ranges = {
        "Merah": [(np.array([0, 120, 70]), np.array([10, 255, 255])), (0, 0, 255)],
        "Hijau": [(np.array([35, 100, 100]), np.array([85, 255, 255])), (0, 255, 0)],
        "Biru": [(np.array([100, 150, 0]), np.array([140, 255, 255])), (255, 0, 0)]
    }

    for color, (ranges, frame_color) in color_ranges.items():
        lower, upper = ranges
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), frame_color, 2)
                cv2.putText(frame, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, frame_color, 2)

    cv2.imshow("Deteksi Warna", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()