import streamlit as st
from PIL import Image
import torch
import tempfile

# Load model hasil training
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

# Judul halaman web
st.set_page_config(page_title="Weather Classification", layout="centered")
st.title("🌤️ Weather Classification with YOLOv5")
st.write("Upload an image, and the model will classify the weather condition.")

# Upload gambar
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Simpan gambar ke file sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image.save(temp.name)
        results = model(temp.name)  # Prediksi

        # Tampilkan hasil deteksi
        st.subheader("🔍 Detection Results")
        results.render()
        st.image(results.ims[0], caption="Detected Image", use_column_width=True)

        # Tampilkan label
        labels = results.pandas().xyxy[0]['name'].unique()
        st.success(f"Detected: {', '.join(labels)}")
