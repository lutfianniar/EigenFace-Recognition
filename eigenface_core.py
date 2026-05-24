import os
import cv2
import numpy as np

def cari_eigen_utama(matrix, iterasi_maks=100, error=1e-6):
    ukuran = matrix.shape[0]

    vektor = np.random.rand(ukuran)
    vektor = vektor / np.linalg.norm(vektor)

    eigen_lama = 0

    for i in range(iterasi_maks):

        hasil = np.dot(matrix, vektor)

        vektor_baru = hasil / np.linalg.norm(hasil)

        eigen_baru = np.dot(
            vektor_baru.T,
            np.dot(matrix, vektor_baru)
        )

        if abs(eigen_baru - eigen_lama) < error:
            break

        vektor = vektor_baru
        eigen_lama = eigen_baru

    return eigen_baru, vektor_baru

def hitung_beberapa_eigen(matrix, jumlah_eigen):

    semua_eigenvalue = []
    semua_eigenvector = []

    matrix_baru = np.copy(matrix)

    for i in range(jumlah_eigen):

        eigenvalue, eigenvector = cari_eigen_utama(matrix_baru)

        semua_eigenvalue.append(eigenvalue)
        semua_eigenvector.append(eigenvector)

        pengurang = eigenvalue * np.outer(eigenvector, eigenvector)

        matrix_baru = matrix_baru - pengurang

    return np.array(semua_eigenvalue), np.array(semua_eigenvector)

class EigenFaceRecognizer:

    def __init__(self, ukuran=(128, 128), jumlah_komponen=20):

        self.ukuran = ukuran
        self.jumlah_komponen = jumlah_komponen

        self.mean_face = None
        self.eigenfaces = None
        self.projected_data = None

        self.labels = []
        self.image_paths = []

    def load_dataset(self, folder):

        data_wajah = []
        nama_file = []
        lokasi_file = []

        for root, dirs, files in os.walk(folder):

            for file in files:

                if file.lower().endswith((".jpg", ".png", ".jpeg")):

                    path = os.path.join(root, file)

                    # baca gambar grayscale
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

                    if img is not None:

                        # resize gambar
                        img = cv2.resize(img, self.ukuran)

                        # ubah jadi vector 1 dimensi
                        vector = img.flatten()

                        data_wajah.append(vector)

                        nama_file.append(file)
                        lokasi_file.append(path)

        if len(data_wajah) == 0:
            return None, nama_file, lokasi_file

        return np.array(data_wajah).T, nama_file, lokasi_file

    def train(self, folder_dataset):

        matrix_gambar, self.labels, self.image_paths = self.load_dataset(folder_dataset)

        if matrix_gambar is None:
            return False, "Dataset kosong"

        jumlah_data = matrix_gambar.shape[1]

        self.mean_face = np.mean(matrix_gambar, axis=1).reshape(-1, 1)

        A = matrix_gambar - self.mean_face

        C = np.dot(A.T, A)

        jumlah = min(self.jumlah_komponen, jumlah_data)

        eigenvalues, eigenvectors_kecil = hitung_beberapa_eigen(C, jumlah)

        self.eigenfaces = np.dot(A, eigenvectors_kecil.T)

        self.eigenfaces = self.eigenfaces / np.linalg.norm(
            self.eigenfaces,
            axis=0
        )

        self.projected_data = np.dot(self.eigenfaces.T, A)

        return True, "Training berhasil"

    def recognize(self, image_test):

        img = cv2.imread(image_test, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return None, 0, "Gambar tidak ditemukan"

        img = cv2.resize(img, self.ukuran)

        test_vector = img.flatten().reshape(-1, 1)

        test_vector = test_vector - self.mean_face

        projected_test = np.dot(
            self.eigenfaces.T,
            test_vector
        )

        jarak = np.linalg.norm(
            self.projected_data - projected_test,
            axis=0
        )

        index_terbaik = np.argmin(jarak)
        minimum = np.min(jarak)

        maksimum_jarak = np.max(jarak)
        if maksimum_jarak == 0:
            persen_kemiripan = 100.0
        else:
            persen_kemiripan = (1.0 - (minimum / maksimum_jarak)) * 100.0
        persen_kemiripan = max(0.0, min(100.0, persen_kemiripan))

        THRESHOLD_JARAK = 5000.0 
        
        if minimum <= THRESHOLD_JARAK:
            terkenali = True
            label_hasil = self.labels[index_terbaik]
            path_hasil = self.image_paths[index_terbaik]
        else:
            terkenali = False
            label_hasil = "Tidak Dikenal (Wajah Asing)"
            path_hasil = None  

        return (
            path_hasil,
            minimum,
            label_hasil,
            round(persen_kemiripan, 2),
            terkenali
        )