# PRD — Obesity Classification

## Ringkasan

Obesity Classification adalah aplikasi web sederhana berbasis Streamlit untuk memperkirakan kategori tingkat obesitas berdasarkan karakteristik fisik dan kebiasaan pengguna. Prediksi dibuat oleh model K-Nearest Neighbors (KNN) yang dimuat dari file model tersimpan. Setelah prediksi, aplikasi dapat meminta Gemini menghasilkan penjelasan risiko dan rekomendasi umum dalam Bahasa Indonesia.

Aplikasi ini ditujukan untuk edukasi dan eksplorasi, bukan untuk mendiagnosis atau menggantikan konsultasi tenaga kesehatan.

## Tujuan

- Menyediakan antarmuka yang mudah digunakan untuk memasukkan data responden.
- Menampilkan kategori hasil klasifikasi dari model KNN.
- Menampilkan probabilitas per kelas jika model mendukung `predict_proba`.
- Menyajikan analisis dan saran umum yang dihasilkan Gemini berdasarkan prediksi dan sebagian data responden.

## Pengguna

Pengguna yang ingin mencoba klasifikasi tingkat obesitas berdasarkan data diri dan kebiasaan yang tersedia di formulir. Hasilnya harus dipahami sebagai informasi edukatif, bukan diagnosis medis.

## Alur Pengguna

1. Pengguna membuka aplikasi.
2. Pengguna mengisi data fisik dan kebiasaan pada formulir.
3. Pengguna memilih **Prediksi Tingkat Obesitas**.
4. Aplikasi menyusun data sesuai urutan fitur model dan menjalankan prediksi KNN.
5. Aplikasi menampilkan kategori prediksi dan, jika didukung model, rincian probabilitas.
6. Aplikasi menghitung BMI dan mengirim prediksi serta sebagian data responden ke Gemini untuk menampilkan analisis singkat.

## Data Masukan

Formulir meminta data berikut:

- Gender, usia, tinggi badan, dan berat badan.
- Riwayat keluarga dengan kelebihan berat badan.
- Konsumsi makanan tinggi kalori dan kebiasaan merokok.
- Frekuensi konsumsi sayur, jumlah makan utama, dan asupan air.
- Frekuensi aktivitas fisik dan waktu menggunakan perangkat teknologi.
- Kebiasaan makan di antara waktu makan dan konsumsi alkohol.
- Moda transportasi.

Nama fitur internal yang digunakan meliputi `Gender`, `Age`, `Height`, `Weight`, `family_history_with_overweight`, `FAVC`, `FCVC`, `NCP`, `CAEC`, `SMOKE`, `CH2O`, `SCC`, `FAF`, `TUE`, `CALC`, dan `MTRANS`.

## Keluaran

- Kategori tingkat obesitas hasil prediksi model.
- Tabel probabilitas kelas jika model menyediakan probabilitas.
- Analisis Gemini berupa risiko kesehatan singkat dan tiga rekomendasi umum berdasarkan prompt aplikasi.
- BMI yang dihitung dari berat dan tinggi untuk disertakan dalam permintaan analisis Gemini.

## Teknologi dan Komponen

- Python dan Streamlit untuk antarmuka aplikasi.
- pandas untuk membentuk data masukan model.
- scikit-learn untuk kompatibilitas model KNN tersimpan.
- `modelKNN_fixed-3.pkl` sebagai artefak model.
- Google Generative AI SDK untuk integrasi Gemini (`gemini-2.5-flash`).

## Menjalankan Aplikasi

Pasang dependensi dari `requirements.txt`, lalu siapkan konfigurasi Streamlit dengan API key Gemini:

```toml
# .streamlit/secrets.toml
API_KEY = "isi-api-key-anda"
```

Pastikan `modelKNN_fixed-3.pkl` tersedia di direktori kerja aplikasi, kemudian jalankan:

```bash
streamlit run app.py
```

Tanpa konfigurasi `API_KEY`, aplikasi tidak dapat membaca pengaturan rahasia saat dimulai. Integrasi Gemini memerlukan API key yang valid dan koneksi ke layanan Google.

## Batasan dan Catatan

- Hasil bergantung pada model dalam `modelKNN_fixed-3.pkl`; dokumentasi ini tidak menyatakan tingkat akurasi atau validasi klinis model.
- Model dan integrasi Gemini dapat gagal dimuat atau dihubungi, misalnya karena artefak model, dependensi, kredensial, atau koneksi tidak tersedia.
- Analisis Gemini dihasilkan oleh AI dan dapat keliru. Jangan gunakan keluaran aplikasi sebagai diagnosis atau dasar tunggal untuk keputusan kesehatan.
- Data yang dikirim ke Gemini mencakup prediksi, BMI, usia, gender, aktivitas fisik, konsumsi sayur, asupan air, kebiasaan merokok, dan riwayat keluarga yang dimasukkan pengguna.

## Kriteria Penerimaan

- Aplikasi menerima masukan melalui formulir dan dapat membentuk satu baris fitur untuk model.
- Saat model berhasil dimuat, pengguna dapat meminta serta melihat kategori prediksi.
- Rincian probabilitas hanya ditampilkan jika model mendukungnya.
- Setelah prediksi, aplikasi menampilkan analisis Gemini apabila konfigurasi dan layanan tersedia.
- Peringatan bahwa aplikasi bersifat edukatif dan bukan pengganti konsultasi dokter tetap terlihat.
