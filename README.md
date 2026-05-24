# Facial Recognition System menggunakan Metode Eigenface
Sistem pengenalan wajah berbasis web menggunakan Streamlit. Proyek ini mengimplementasikan algoritma Principal Component Analysis (PCA) secara manual menggunakan metode *Power Iteration* dan *Deflation* untuk mengekstraksi nilai Eigen.

### Anggota Kelompok
* Annisa Salma Tabina (L0125004)
* Lutfiannisa Tri Yuniarti (L0125020)
* Gita Florensia Adi (L0125101)

---

### Persyaratan Sistem
Pastikan Python sudah terinstal di komputer Anda. Anda sangat disarankan menggunakan Virtual Environment (`.venv`). Berikut adalah library yang dibutuhkan untuk menjalankan program ini:
* `streamlit`
* `opencv-python`
* `numpy`
* `Pillow`

Cara instalasi semua library sekaligus:
`pip install streamlit opencv-python numpy Pillow`

---

### Cara Menjalankan Program (How to Run)
1. Buka terminal (atau Command Prompt / VS Code terminal) dan pastikan posisi direktori sudah berada di dalam folder utama proyek ini.
2. Jika menggunakan Virtual Environment, aktifkan terlebih dahulu:
   `.\.venv\Scripts\activate` (Untuk Windows)
3. Jalankan aplikasi Streamlit dengan perintah berikut:
   `streamlit run src/app_frontend.py`
4. Program akan otomatis terbuka di browser web Anda (biasanya di `http://localhost:8501`).

### Panduan Penggunaan Aplikasi
1. Pada menu di sebelah kiri, masukkan lokasi folder dataset wajah pada bagian **Dataset Folder Path** (Contoh: `test/dataset_training`).
2. Klik tombol **Train Dataset**. Tunggu beberapa saat hingga matriks bobot selesai dihitung.
3. Unggah foto wajah yang ingin diuji melalui kolom **Insert Your Image File** (Gunakan gambar dari folder `test/gambar_uji`).
4. Sistem akan menampilkan wajah yang paling mirip atau menolak wajah jika melewati batas Threshold Jarak Euclidean.
