import tensorflow as tf
import os
import yaml

CONFIG_PATH = 'config/config.yaml'

def load_config():
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Gagal memuat konfigurasi: {e}")
        return None

def load_generator_from_checkpoint():
    from src.model import build_generator 
    
    config = load_config()
    if not config:
        return None

    noise_dim = config.get('NOISE_DIM')
    img_size = config.get('IMAGE_SIZE', 64)
    channels = config.get('CHANNELS', 3)
    checkpoint_dir = config.get('CHECKPOINT_PATH', 'outputs/models')
    
    try:
        # 1. Bangun arsitektur model
        generator = build_generator(noise_dim, img_size, channels)
        
        # 2. Inisialisasi Checkpoint
        checkpoint = tf.train.Checkpoint(generator=generator)
        
        # 3. Muat checkpoint terbaru
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        
        if latest_checkpoint:
            checkpoint.restore(latest_checkpoint).expect_partial() 
            return generator, img_size
        else:
            print(f"ERROR: Gagal menemukan checkpoint model di {checkpoint_dir}.")
            return None, img_size
            
    except Exception as e:
        print(f"ERROR saat memuat model dari checkpoint: {e}")
        return None, img_size