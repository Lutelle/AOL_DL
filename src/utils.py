import tensorflow as tf
import os
import numpy as np

# --- KONFIGURASI LOSS LSGAN ---
REAL_TARGET = 1.0
FAKE_TARGET = 0.0
GEN_TARGET = 1.0
mse_loss = tf.keras.losses.MeanSquaredError()
# -----------------------------


def load_data(data_path, img_size, batch_size):
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset path tidak ditemukan: {data_path}")
        
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_path,
        labels=None, 
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True,
        interpolation='bilinear'
    )
    
    def normalize_img(image):
        image = tf.cast(image, tf.float32)
        image = (image - 127.5) / 127.5
        return image

    dataset = dataset.map(normalize_img)
    
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset


def discriminator_loss(real_output, fake_output):
    real_labels = tf.ones_like(real_output) * REAL_TARGET
    real_loss = mse_loss(real_labels, real_output)
    
    fake_labels = tf.ones_like(fake_output) * FAKE_TARGET
    fake_loss = mse_loss(fake_labels, fake_output)
    
    return real_loss + fake_loss


def generator_loss(fake_output):
    gen_labels = tf.ones_like(fake_output) * GEN_TARGET
    return mse_loss(gen_labels, fake_output)


def generate_and_save_images(model, epoch, test_input, img_size, checkpoint_dir):
    
    predictions = model(test_input, training=False)

    predictions = (predictions * 0.5) + 0.5 

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(tf.clip_by_value(predictions[i, :, :, :].numpy(), 0, 1))
        plt.axis('off')

    plt.savefig(os.path.join(checkpoint_dir, f'image_at_epoch_{epoch:04d}.png'))
    plt.close(fig)