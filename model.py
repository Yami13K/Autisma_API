import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras

import numpy as np
import pandas as pd
import shutil
import time
import cv2 as cv2

import os

from PIL import Image
from IPython.core.display import display, HTML

from tensorflow import keras



def predictImages(filename):
    model = keras.models.load_model('vgg19.h5')

    test_df = pd.DataFrame({
        'filename': filename
    }, index=[0])
    nb_samples = test_df.shape[0]

    test_gen = ImageDataGenerator(rescale=1. / 255)
    batch_size = 64
    image_size = 224

    test_generator = test_gen.flow_from_dataframe(
        test_df,
        # r"IMAGES",
        x_col='filename',
        y_col=None,
        class_mode=None,
        batch_size=batch_size,
        target_size=(image_size, image_size),
        shuffle=False
    )

    predict = model.predict(test_generator, steps=np.ceil(nb_samples / batch_size))
    return predict
