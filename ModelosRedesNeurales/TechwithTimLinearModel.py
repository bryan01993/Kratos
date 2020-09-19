import unittest
import tensorflow as tf
import keras
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
data = pd.read_csv('student-mat.csv',sep=';')

data = data[['G1','G2','G3','studytime','failures','absences']]

predict = 'G3'

X = np.array(data.drop([predict],1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

best = 0

#Training part, at the end it saves the model that best predicts the result, use for training, then comment
"""for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train,y_train)

    acc = linear.score(x_test,y_test)
    print(acc)

    if acc > best:
        best = acc
        with open ('studentmodel.pickle','wb') as f:    #saves model god knows how
            pickle.dump(linear,f)"""

pickle_in =  open('studentmodel.pickle','rb')
linear = pickle.load(pickle_in)


print('Co: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)


predictions =linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x],y_test[x])
p = 'G1'
style.use('ggplot')
pyplot.scatter(data[p],data['G3'])
pyplot.xlabel(p)
pyplot.ylabel('Final Grade')
pyplot.show()