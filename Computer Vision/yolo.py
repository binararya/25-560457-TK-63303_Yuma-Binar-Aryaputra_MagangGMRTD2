import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")


#Target classes: person, bottle, cell phone
TARGET_CLASSES = [0, 39, 67] 

CONFIDENCE_THRESHOLD = 0.5

#Inisialisasi webcam untuk kamera default
cap = cv2.VideoCapture(0)

# Mengecek apakah webcam berhasil terbuka
if not cap.isOpened():
    print("Error: Tidak dapat mengakses webcam.")
    exit()

#Membaca Frame dan Melakukan Deteksi
while True:
    ret, frame = cap.read()

    if not ret:
        print("Gagal membaca frame dari webcam.")
        break

    results = model(
        frame,
        classes=TARGET_CLASSES,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )

    result = results[0]

    # Menggambar bounding box

    for box in result.boxes:
        # Koordinat bounding box dalam format (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # ID kelas objek hasil deteksi
        cls_id = int(box.cls[0])

        # String nama kelas objek 
        class_name = model.names[cls_id]

        # Confidence score 
        confidence = float(box.conf[0])

        # Menggambar kotak (bounding box) pada frame dengan parameter 
        # gambar, titik awal, titik akhir, warna (BGR), dan ketebalan garis
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Menyusun teks label nama kelas dan confidence score (2 desimal)
        label = f"{class_name} {confidence:.2f}"

        # Menuliskan teks label di atas bounding box dengan parameter
        # gambar, teks, posisi, font, skala font, warna, ketebalan
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


    # Menampilkan hasil
    cv2.imshow("Deteksi Objek - YOLOv8n", frame)

    # Menggunakan tombol 'q' untuk keluar loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Pembersihan resource, melepas akses ke webcam dan menutup jendela
cap.release()          
cv2.destroyAllWindows()  

