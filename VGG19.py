from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
import keras, os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, GlobalMaxPooling2D
import numpy as np
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras import optimizers, layers, applications

def loadmodel():
    model = keras.models.load_model('vgg19.h5')
def nothing():
    test_df = pd.DataFrame({
        'filename': '11.jpg'
    }, index=[0])
    nb_samples = test_df.shape[0]

    test_gen = ImageDataGenerator(rescale=1. / 255)
    batch_size = 64
    image_size = 224

    test_generator = test_gen.flow_from_dataframe(
        test_df,
        r"FastAPI",
        x_col='filename',
        y_col=None,
        class_mode=None,
        batch_size=batch_size,
        target_size=(image_size, image_size),
        shuffle=False
    )

    predict = model.predict(test_generator, steps=np.ceil(nb_samples / batch_size))
    print(predict)
