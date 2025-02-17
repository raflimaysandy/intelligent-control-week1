import cv2
import numpy as np

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_area = frame.shape[0] * frame.shape[1]  # Luas area frame

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
            area = cv2.contourArea(cnt)
            if area > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), frame_color, 2)
                cv2.putText(frame, color, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, frame_color, 2)
                
                # Menghitung akurasi per objek
                box_area = w * h
                accuracy = (area / box_area) * 100
                accuracy_text = f"Akurasi: {accuracy:.2f}%"
                
                # Menampilkan akurasi di atas bounding box
                cv2.putText(frame, accuracy_text, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, frame_color, 2)

    cv2.imshow("Deteksi Warna dengan Akurasi", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
