import numpy as np
import math
import timeit

from PIL import Image as im
import random as rd
import sys

num = sys.argv[1]

imgArray=[]
RESIZE_SIZE = 32 #Change depending on what input your CNN expects
for j in range(1,10001):
    imgName = '../input/train/' + str(j + 10000*int(num)) +'.jpg'
    img = im.open(imgName).convert("RGB")
    img = img.resize((RESIZE_SIZE, RESIZE_SIZE))
    imgArray.append(np.array(img))

imgArray = np.array(imgArray)

print('data shape: ', imgArray.shape)

import keras
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import scipy
from scipy import misc
import os
import math


# load inceptionV3 model + remove final classification layers
model = InceptionV3(weights='imagenet', include_top=False, input_shape=(139, 139, 3))
print('model loaded')

# obtain bottleneck features (train)
if os.path.exists('inception_features_train_' + num + '.npz'):
    print('bottleneck features detected (train)')
    features = np.load('inception_features_train_' + num + '.npz')['features']
else:
    print('bottleneck features file not detected (train)')
    print('calculating now ...')
    # pre-process the train data
    big_imgArray = np.array([scipy.misc.imresize(imgArray[i], (139, 139, 3))
                            for i in range(0, len(imgArray))]).astype('float32')
    inception_input_train = preprocess_input(big_imgArray)
    print('train data preprocessed')
    # extract, process, and save bottleneck features
    features = model.predict(inception_input_train)
    features = np.squeeze(features)
    np.savez('inception_features_train_' + argument, features=features)

print('bottleneck features saved (train)')
