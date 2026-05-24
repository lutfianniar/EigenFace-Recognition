import os
import time
import streamlit as st
from PIL import Image

from eigenface_core import EigenFaceRecognizer

st.set_page_config(page_title="Face Recognition App - Eigenface", layout="wide")

if "recognizer" not in st.session_state:
    st.session_state.recognizer = EigenFaceRecognizer()
if "is_trained" not in st.session_state:
    st.session_state.is_trained = False

st.title("Face Recognition System")
st.markdown("---")

col_controls, col_test_view, col_result_view = st.columns([1, 1.2, 1.2])

with col_controls:
    st.markdown("### **Input Panel**")
    
    dataset_path = st.text_input("Insert Your Dataset Folder Path", placeholder="Contoh: D:/dataset/faces")
    
    if st.button("Train Dataset", type="primary"):
        if dataset_path and os.path.exists(dataset_path):
            with st.spinner("Melakukan kalkulasi matriks kovarian & nilai eigen..."):
                success, msg = st.session_state.recognizer.train(dataset_path)
                if success:
                    st.session_state.is_trained = True
                    st.success("Sukses: Karakteristik wajah berhasil dipelajari!")
                else:
                    st.error(msg)
        else:
            st.error("Gagal: Path folder database tidak ditemukan!")
            
    st.write("")
    
    uploaded_file = st.file_uploader("Insert Your Image File", type=["jpg", "png", "jpeg"])

    st.write("") 
    st.markdown("### **Metrics & Information**")
    output_name = st.empty()
    output_similarity = st.empty()
    output_threshold = st.empty()
    output_time = st.empty()
    output_dist = st.empty()

if uploaded_file is not None:
    with col_test_view:
        st.markdown("### **Test Image**")
        test_img = Image.open(uploaded_file)
        st.image(test_img, use_container_width=True)
        
    if st.session_state.is_trained:
        temp_filename = "temp_stream_upload.jpg"
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        start_time = time.time()
        
        matched_path, distance, label, similarity, terkenali = st.session_state.recognizer.recognize(temp_filename)
        
        end_time = time.time()
        
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
        if terkenali and matched_path and os.path.exists(matched_path):
            with col_result_view:
                st.markdown("### **Closest Result**")
                matched_img = Image.open(matched_path)
                st.image(matched_img, use_container_width=True)
                
            output_name.markdown(f"**Result Identification:** `{label}`")
            output_similarity.markdown(f"**Tingkat Kemiripan:** `{similarity}%`")
            output_threshold.markdown(f"**Threshold Active:** `Jarak < 5000 (Lolos)`")
            output_time.markdown(f"**Execution Time:** `{end_time - start_time:.4f} detik`")
            output_dist.markdown(f"**Euclidean Distance:** `{distance:.2f}`")
        else:
            with col_result_view:
                st.markdown("### **Closest Result**")
                st.error("Wajah Tidak Dikenali di Dalam Sistem Dataset!")
            
            output_name.markdown(f"**Result Identification:** `{label}`")
            output_similarity.markdown(f"**Tingkat Kemiripan Terbaik:** `{similarity}%`")
            output_threshold.markdown(f"**Threshold Active:** `Jarak < 5000 (Ditolak)`")
            output_time.markdown(f"**Execution Time:** `{end_time - start_time:.4f} detik`")
            output_dist.markdown(f"**Euclidean Distance:** `{distance:.2f}`")
    else:
        with col_result_view:
            st.markdown("### **Closest Result**")
            st.warning("Silakan tekan tombol 'Train Dataset' terlebih dahulu sebelum memproses.")
else:
    with col_test_view:
        st.markdown("### **Test Image**")
        st.info("Menunggu berkas citra uji dimasukkan...")
    with col_result_view:
        st.markdown("### **Closest Result**")
        st.info("Hasil komparasi terdekat akan muncul di sini.")
