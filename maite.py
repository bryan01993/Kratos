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
print(len(dataframe.columns))
dataframe = pd.DataFrame(list(zip(balance, strategy, size)))
columns = len(dataframe.columns)
print(columns)
model = keras.Sequential([
    keras.layers.Dense(32, activation="relu", input_shape=(columns, ), use_bias=False,name='Input_Layer'),
    keras.layers.Dense(4, activation="relu",name='Output_Layer') # output layer
])

optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False,name='Optimizer_SGD')
model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(), metrics=["accuracy"])
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('./checkpoints/',monitor = 'val_accuracy',verbose =2 ,save_weights_only=True,save_best_only=True,mode='max')
model.summary()
model.get_weights()
history = model.fit(dataframe,batch_size=1, epochs=5)
