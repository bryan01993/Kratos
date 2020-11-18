import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.optimizers import SGD
import sklearn
from sklearn import preprocessing
import numpy as np
import pandas as pd

LOCATION = './atrox/WARWICK_Portfolio.csv'
dataframe = pd.read_csv(LOCATION)

le = preprocessing.LabelEncoder()
strategy = le.fit_transform(list(dataframe['Strategy name (Global)']))
balance = dataframe['Balance (Global)']
size= dataframe['Size (Global)']

dataframe = pd.DataFrame(list(zip(balance, strategy, size)))
columns = len(dataframe.columns)

model = keras.Sequential([
    keras.layers.Dense(32, activation="relu", input_shape=(columns, ), use_bias=False),
    keras.layers.Dense(4, activation="softmax") # output layer
])
model.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError(), metrics=["accuracy"])
model.summary()

model.fit(dataframe, epochs=5)
