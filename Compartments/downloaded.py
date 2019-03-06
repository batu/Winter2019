import random, glob, cv2, os

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model
from keras import optimizers
from keras import layers
from keras import regularizers
from keras import metrics
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Concatenate, concatenate, Lambda, Dropout, multiply
from keras.callbacks import TensorBoard

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from sklearn.model_selection import train_test_split


import keras
print(keras.__version__)
print(tf.__version__)

GAME_NAME = "Breakout"
#Due to Keras structuring the data is layout as GameName/Screen/Pixels/All_Images.pngs
IMAGE_LOAD_PATH = f"GameplayData/{GAME_NAME}/Screen"

# Get the files sorted by the order they were modified (created.)

x_mem_train = np.load(f"GameplayData/{GAME_NAME}/Ram/{GAME_NAME}_RAMs.npy")[0:1]
print("The memory has shape:", x_mem_train.shape)

file_names = sorted(glob.glob(f"{IMAGE_LOAD_PATH}/Pixels/*.png"), key=os.path.getmtime)#[0:2000]
x_train = np.array([cv2.imread(file) for file in file_names])
x_train = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in x_train])


# Deepmind DownSampling
x_train = np.array([cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in x_train])
x_train = np.array([cv2.resize(image, (84, 84), interpolation=cv2.INTER_AREA) for image in x_train])
x_train = x_train.reshape(len(x_train), 84, 84, 1)
print("The data has been loaded, with the shape:", x_train.shape)

random.choice(x_train)
random_image = x_train[0]
IMAGE_SHAPE = np.array(random_image).shape
print("Loaded image has shape:",IMAGE_SHAPE )
plt.imshow(random_image.squeeze())


x_train = x_train * (1./255)

assert np.max(x_train) <= 1 and np.min(x_train) >= 0, "The normalization went wrong!"
# assert np.max(x_mem_train) <= 1 and np.min(x_mem_train) >= 0, "The normalization went wrong!"
print("The max and min value of the x_train are:", np.max(x_train),np.min(x_train))

img = random.choice(x_train)
print(img.shape)
plt.imshow(img.squeeze(),  cmap='gray')
print("The training shape:", x_train.shape)
# print("The testing shape:", x_test.shape)

x_len = x_train.shape[0]
print(x_train[ :x_len - 4].shape)
print(x_train[1:x_len - 3].shape)
print(x_train[2:x_len - 2].shape)
print(x_train[3:x_len - 1].shape)

four_vec_stack = [x_train[ :x_len-4].squeeze(), 
                  x_train[1:x_len-3].squeeze(), 
                  x_train[2:x_len-2].squeeze(),
                  x_train[3:x_len-1].squeeze()]
# del x_train

four_vec_stack = np.stack(four_vec_stack, axis=-1)

print(four_vec_stack.shape)

input_image = Input(shape=(84,84,4))  # adapt this if using `channels_first` image data format

x = layers.Conv2D(32, (8, 8), strides= 4, activation='relu',)(input_image)
x = layers.Conv2D(64, (4, 4), strides= 2, activation='relu')(x)
last_conv = layers.Conv2D(64, (3, 3), strides= 1, activation='relu')(x)
feature_extractor = layers.Flatten(name="feature_extractor")(last_conv)

#####################################################################
dense_out = layers.Dense(512, activation="relu")(feature_extractor)
#####################################################################
x = layers.Reshape((8, 8, 8))(dense_out)

x = layers.Conv2DTranspose(64, (3, 3), strides=1, activation='relu',)(x)
x = layers.Conv2DTranspose(64, (4, 4), strides= 2, activation='relu')(x)
x = layers.Conv2DTranspose(1, (8, 8), strides= 4, activation='relu')(x)

decoded = layers.Cropping2D(cropping=(4, 4))(x)

nature_cnn_feature = Model(inputs=input_image, outputs=dense_out)
autopredictor = Model(input_image, decoded)

autopredictor.compile(optimizer='Adam', loss='binary_crossentropy')
autopredictor.summary()

# autopredictor.fit(x=four_vec_stack, y=four_vec_stack[:,:,:,2:3],  validation_split=0.05, epochs=3) 

nature_cnn_feature.compile(optimizer='Adam', loss='binary_crossentropy')
nature_cnn_feature.save("breakout_random_4stack_keras1_8.h5")
nature_cnn_feature.summary()

predict_start_state = 50
decoded_imgs = autopredictor.predict(four_vec_stack)
print(np.array(decoded_imgs).shape)
print(decoded_imgs.shape)
n = 10
plt.figure(figsize=(80, 16))
for i in range(1, n):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(four_vec_stack[:,:,:,3][predict_start_state:predict_start_state+10][0])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(84, 84))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

