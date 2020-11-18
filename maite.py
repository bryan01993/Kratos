import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.optimizers import SGD
import sklearn
from sklearn import preprocessing
import numpy as np
import pandas as pd

PORTFOLIO = 'AATROX'
LOCATION = '/home/miguel/Proyectos/kratos/atrox/WARWICK_Portfolio.csv'

dataframe = pd.read_csv(LOCATION)
len(list(dataframe))



# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28,28)), # input layer
#     keras.layers.Dense(128,activation="relu"), # hidden layer
#     keras.layers.Dense(10,activation="softmax") # output layer
# ])
# model.compile(optimizer="adam",loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

# model.fit(dataframe, epochs=5)#number of repetitions


