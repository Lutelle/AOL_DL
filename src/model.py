import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, BatchNormalization

def build_generator(noise_dim, img_size, channels):
    
    model = Sequential(name='Generator')
    
    # 1. Input: Vektor Z -> Output: 4x4x512
    initial_size = img_size // 16 
    filters = 512
    
    model.add(Dense(initial_size * initial_size * filters, use_bias=False, input_shape=(noise_dim,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Reshape((initial_size, initial_size, filters))) 
    
    # 2. Transpose Convolutions (upsampling)
    model.add(Conv2DTranspose(filters // 2, (5, 5), strides=(2, 2), padding='same', use_bias=False)) # 8x8
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(filters // 4, (5, 5), strides=(2, 2), padding='same', use_bias=False)) # 16x16
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    
    model.add(Conv2DTranspose(filters // 8, (5, 5), strides=(2, 2), padding='same', use_bias=False)) # 32x32
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    
    # 5. Output Layer: 64x64x3
    model.add(Conv2DTranspose(channels, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    
    return model

def build_discriminator(img_size, channels):
    
    input_shape = (img_size, img_size, channels)
    model = Sequential(name='Discriminator')

    # 1. Conv 1: 64x64x3 -> 32x32x64
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    # 2. Conv 2: 32x32 -> 16x16x128
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    # 3. Conv 3: 16x16 -> 8x8x256
    model.add(Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    # 4. Output Layer
    model.add(Flatten())
    model.add(Dense(1)) 

    return model