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

dataframe = list(zip(balance, strategy, size))

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(3,)))
model.add(tf.keras.layers.Dense(8))
model.compile(optimizer="adam",loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
model.fit(dataframe, epochs=5)