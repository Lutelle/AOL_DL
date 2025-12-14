## Buat Lingkungan Virtual
    Buka terminal dan masukan:

    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    venv\Scripts\activate     # Windows

## Instal Dependensi
    pip install -r requirements.txt

## Persiapan Dataset
1.  Unduh *Anime Face Dataset*. https://www.kaggle.com/datasets/splcher/animefacedataset
2.  Ekstrak gambar ke lokasi yang ditentukan di dalam *notebook* pelatihan (`data/anime_faces`).

### Pelatihan Ulang Model (Opsional)
Jika Anda ingin melatih model dari awal atau melanjutkan pelatihan:
1.  Buka `notebooks/main.ipynb`.
2.  Pastikan `DATASET_PATH` dan `CHECKPOINT_DIR` di *notebook* telah diatur dengan benar.
3.  Jalankan semua sel. Model akan menyimpan *checkpoint* ke folder `outputs/models`.

### Menjalankan Aplikasi Web (Demonstrasi)
Gunakan Streamlit untuk menjalankan antarmuka web yang dapat mendemonstrasikan model Generator yang sudah dimuat.

1.  Pastikan folder `outputs/models` berisi *checkpoint* model yang sudah terlatih.
2.  Jalankan perintah berikut di terminal:

    streamlit run app.py

3.  Aplikasi akan terbuka di *browser* Anda (biasanya di `http://localhost:8501`). Tekan tombol **"Generate Wajah Baru"** untuk melihat hasil sintesis.