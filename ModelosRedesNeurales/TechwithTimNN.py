import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist


(train_images, train_labels), (test_images, test_labels) = data.load_data()
print(data)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0
# here start model creation
model = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),  # input layer
                          keras.layers.Dense(128,activation="relu"),  # hidden layer
                          keras.layers.Dense(10,activation="softmax")  # output layer
                          ])
model.compile(optimizer="adam",loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

model.fit(train_images,train_labels, epochs=5)  # number of repetitions

prediction = model.predict(test_images)
# Below here validating the model through visual vs prediction
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: "+ class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()
