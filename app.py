import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT) 

import streamlit as st
import tensorflow as tf
import numpy as np
import yaml

try:
    import app.model_loader as loader
    load_generator_from_checkpoint = loader.load_generator_from_checkpoint
    load_config = loader.load_config
    
except ImportError as e:
    st.error(f"Gagal mengimpor modul ({e}).")

    load_generator_from_checkpoint = lambda: (None, 64)
    load_config = lambda: {'TRAINING_PARAMETERS': {'NOISE_DIM': 100}}

# --- 1. KONFIGURASI ---
config = load_config()
if config:
    NOISE_DIM = config.get('NOISE_DIM', 100)
    
# --- 2. FUNGSI PEMUATAN MODEL (Caching Streamlit) ---

@st.cache_resource
def get_model_and_size():
    return load_generator_from_checkpoint()

# --- 3. FUNGSI GENERASI ---

def generate_anime_face(generator, noise_dim):
    noise = tf.random.normal([1, noise_dim])
    
    # Generasi gambar (training=False)
    generated_image = generator(noise, training=False).numpy()[0]
    
    # De-normalisasi: [-1, 1] -> [0, 1]
    generated_image = (generated_image * 0.5) + 0.5 
    
    # Klip untuk memastikan rentang [0, 1]
    return np.clip(generated_image, 0, 1)

# --- 4. TAMPILAN UTAMA STREAMLIT ---

def main():
    st.set_page_config(
        page_title= "Face Generator",
        layout="centered"
    )

    st.title("Face Generator")
    
    st.markdown("---")

    generator_model, img_size = get_model_and_size()
    
    if generator_model is None:
        st.warning("Aplikasi tidak dapat berjalan. Model Generator gagal dimuat.")
        return

    st.markdown("### Kontrol Generasi")
    
    if st.button("Generate Wajah Baru", key="generate_button", type="primary"):
        with st.spinner('Model sedang menghasilkan gambar...'):
            new_face = generate_anime_face(generator_model, NOISE_DIM)
            st.session_state['last_generated_face'] = new_face
            st.session_state['message'] = "Wajah baru berhasil dibuat!"

    # --- Tampilan Output ---
    st.markdown("---")
    st.markdown("### Hasil Generasi")

    if 'last_generated_face' in st.session_state:
        display_width = img_size * 4 
        st.image(
            st.session_state['last_generated_face'], 
            caption=st.session_state.get('message', 'Wajah Terbaru'), 
            use_container_width=False,
            width=display_width
        )
    else:
        st.info("Tekan tombol 'Generate Wajah Baru' untuk memulai.")
        
    st.markdown("---")
    st.caption(f"Arsitektur Model: DCGAN {img_size}x{img_size} | Loss Function: LSGAN | Powered by TensorFlow")

if __name__ == "__main__":
    if config:
        main()
    else:
        st.error("Gagal menjalankan aplikasi: Konfigurasi proyek tidak dapat dibaca.")