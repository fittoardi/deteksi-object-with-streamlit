# deteksi-object-with-streamlit

code python deteksi object nya
```
import streamlit as st
import cv2
from PIL import Image
from ultralytics import YOLO
import numpy as np
import tempfile

# Load model YOLOv8 (gunakan 'yolov8n.pt', 'yolov8s.pt', atau model custom kamu)
model = YOLO("yolov8n.pt")  # ganti dengan model kamu jika pakai model sendiri

st.title("Deteksi Objek - UAS Computer Vision")
st.write("Upload gambar atau video untuk dideteksi menggunakan YOLOv8")

# Pilih jenis input
option = st.radio("Pilih jenis input:", ["Gambar", "Video"])

# Fungsi deteksi pada gambar
def detect_image(img):
    results = model(img)
    result_img = results[0].plot()  # hasil dengan bounding box
    return result_img

# Fungsi deteksi pada video
def detect_video(uploaded_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)[0]
        annotated_frame = results.plot()

        stframe.image(annotated_frame, channels="BGR")

    cap.release()

# Gambar
if option == "Gambar":
    uploaded_img = st.file_uploader("Upload gambar", type=["jpg", "png", "jpeg"])
    if uploaded_img is not None:
        img = Image.open(uploaded_img)
        img_array = np.array(img)
        result_img = detect_image(img_array)
        st.image(result_img, caption="Hasil Deteksi")

# Video
elif option == "Video":
    uploaded_vid = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])
    if uploaded_vid is not None:
        detect_video(uploaded_vid)
```
