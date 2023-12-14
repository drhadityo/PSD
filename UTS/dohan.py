import streamlit as st
import os
import librosa
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, mode
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle

# Fungsi untuk menghitung statistik audio
def calculate_statistics(audio_path):
    y, sr = librosa.load(audio_path)

    # Menghitung statistik
    mean = np.mean(y)
    std_dev = np.std(y)
    max_value = np.max(y)
    min_value = np.min(y)
    median = np.median(y)
    skewness = skew(y)
    kurt = kurtosis(y)
    q1 = np.percentile(y, 25)
    q3 = np.percentile(y, 75)
    mode_value, _ = mode(y)
    iqr = q3 - q1

    zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y=y))
    zcr_median = np.median(librosa.feature.zero_crossing_rate(y=y))
    zcr_std_dev = np.std(librosa.feature.zero_crossing_rate(y=y))
    zcr_kurtosis = kurtosis(librosa.feature.zero_crossing_rate(y=y)[0])
    zcr_skew = skew(librosa.feature.zero_crossing_rate(y=y)[0])

    rms = np.sum(y**2) / len(y)
    rms_median = np.median(y**2)
    rms_std_dev = np.std(y**2)
    rms_kurtosis = kurtosis(y**2)
    rms_skew = skew(y**2)

    return [mean, std_dev, max_value, min_value, median, skewness, kurt, q1, q3, mode_value[0], iqr, zcr_mean, zcr_median, zcr_std_dev, zcr_kurtosis, zcr_skew, rms, rms_median, rms_std_dev, rms_kurtosis, rms_skew]

# Memuat model dan skalar yang telah dilatih
with open('ini_zscore.pkl', 'rb') as file:
    normalisasi_zscore = pickle.load(file)

# Memuat model KNN untuk kategori emosi dengan normalisasi ZScore
with open('model_zscore_terbaik.pkl', 'rb') as file:
    knn_zscore = pickle.load(file)

#PCA ZSCORE
with open('model_pca_zscore.pkl', 'rb') as pca:
    loadpca= pickle.load(pca)
with open('file_knn_pca_zscore.pkl', 'rb') as modelpca:
    knnpca= pickle.load(modelpca)

# Aplikasi Streamlit
st.title("Deteksi Pengenalan Emosi")
st.markdown("**Nama  : Firlli Yuzia Rahmanu.**")
st.markdown("**NIM   : 210411100163.**")
st.write("Unggah file audio.")

uploaded_file = st.file_uploader("Pilih file audio...", type=["wav","mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    if st.button("Deteksi Audio"):
        # Simpan file audio yang diunggah
        audio_path = "audio_diunggah.wav"
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Hitung statistik untuk file audio yang diunggah
        statistik = calculate_statistics(audio_path)

        # Normalisasi data menggunakan ZScore
        data_ternormalisasi_zscore = normalisasi_zscore.transform([statistik])[0]

        # Prediksi label emosi dengan normalisasi ZScore
        label_emosi_zscore = knn_zscore.predict(data_ternormalisasi_zscore.reshape(1, -1))[0]

        st.write("Emosi Terdeteksi (ZScore):", label_emosi_zscore)

        # Hapus file audio yang diunggah
        os.remove(audio_path)
    
    if st.button("Cek Nilai Statistik"):
        # Simpan file audio yang diunggah
        audio_path = "audio_diunggah.wav"
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Hitung statistik untuk file audio yang diunggah
        statistik = calculate_statistics(audio_path)
        
        # Tampilkan tabel statistik
        st.write("**Statistik Audio:**")
        st.write(f"Mean: {statistik[0]:.4f}")
        st.write(f"Standard Deviation: {statistik[1]:.4f}")
        st.write(f"Max: {statistik[2]:.4f}")
        st.write(f"Min: {statistik[3]:.4f}")
        st.write(f"Median: {statistik[4]:.4f}")
        st.write(f"Skewness: {statistik[5]:.4f}")
        st.write(f"Kurtosis: {statistik[6]:.4f}")
        st.write(f"Q1: {statistik[7]:.4f}")
        st.write(f"Q3: {statistik[8]:.4f}")
        st.write(f"Mode: {statistik[9]:.4f}")
        st.write(f"IQR: {statistik[10]:.4f}")

        st.write("**Zero Crossing Rate (ZCR):**")
        st.write(f"Mean: {statistik[11]:.4f}")
        st.write(f"Median: {statistik[12]:.4f}")
        st.write(f"Standard Deviation: {statistik[13]:.4f}")
        st.write(f"Kurtosis: {statistik[14]:.4f}")
        st.write(f"Skewness: {statistik[15]:.4f}")

        st.write("**Root Mean Square Energy (RMSE):**")
        st.write(f"RMSE: {statistik[16]:.4f}")
        st.write(f"RMSE Median: {statistik[17]:.4f}")
        st.write(f"RMSE Standard Deviation: {statistik[18]:.4f}")
        st.write(f"RMSE Kurtosis: {statistik[19]:.4f}")
        st.write(f"RMSE Skewness: {statistik[20]:.4f}")

        # Hapus file audio yang diunggah
        os.remove(audio_path)


    if st.button("PCA"):
        # Simpan file audio yang diunggah
        audio_path = "audio_diunggah.wav"
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Hitung statistik untuk file audio yang diunggah
        statistik = calculate_statistics(audio_path)
        
        data_ternormalisasi_zscore = normalisasi_zscore.transform([statistik])[0]
        inipca=loadpca.transform(data_ternormalisasi_zscore.reshape(1, -1))[0]
        st.write("CIRI YANG SUDAH DI REDUKSI DIMENSI : ")
        st.write(inipca)
        # Lakukan reduksi dimensi PCA pada data yang telah dinormalisasi
        data_reduksi_pca = loadpca.transform([data_ternormalisasi_zscore])

        # Gunakan model KNN untuk melakukan prediksi emosi pada data yang telah direduksi dimensi
        label_emosi_zscore_pca = knnpca.predict(data_reduksi_pca)
        predicted_emotion = label_emosi_zscore_pca[0]

        st.write("Emosi Terdeteksi :", predicted_emotion)
