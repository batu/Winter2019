import numpy as np
from keras.models import load_model
from keras.backend import clear_session
from keras.applications.inception_v3 import InceptionV3
import matplotlib.pyplot as plt
from scipy.misc import imsave, imread
from skimage.transform import resize
from glob import glob
import subprocess
import os

model_dirs = 'Model/'

original_embedding = np.load(model_dirs + 'embedding.npy')
input_model = load_model(model_dirs + 'input_model.h5')
embedded_model = load_model(model_dirs + 'embedded_model.h5')
inception = InceptionV3(weights='imagenet')
max_embedding = np.amax(original_embedding, axis=0)
min_embedding = np.amin(original_embedding, axis=0)

actions = ['U', 'D', 'L', 'R', 's', 'S', 'Y', 'B', 'X', 'A', 'l', 'r']
with open('../Model/Input Log.txt', 'r') as f:
    for i, line in enumerate(f):
        temp_action = 0
        for i, action in enumerate(actions):
            if line.find(action) != -1:
                temp_action += 2 ** i
        if temp_action == 0 or i < 3:
            continue
        if temp_action in action_dict:
            action_dict[temp_action] += 1
        else:
            action_dict[temp_action] = 1
