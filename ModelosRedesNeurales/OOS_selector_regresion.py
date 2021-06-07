import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor

from services.create_timebricks import CreateTimebricks
from services.helpers import movecol
from sklearn import preprocessing
from tensorflow import keras
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
print('Tensorflow Version:',tf.__version__)
# Datos de Prueba Personal
base_dir = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/EA-B1v2/GBPJPY/M15/WF_Report'
save_dir = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/EA-B1v2/GBPJPY/M15/'

# Datos de Prueba en Laptop
#base_dir = 'C:/Users/bryan.aleixo/EA-B1v1Train/GBPUSD/H4/WF_Report'
#save_dir = 'C:/Users/bryan.aleixo/EA-B1v1Train/GBPUSD/H4/'

train_from = '2007.01.01'
test_from = '2015.01.01'
test_finish = '2020.12.01'
test_runs = 30
total_bricks = CreateTimebricks(train_from,1,48,12,0,test_finish)
total_list = total_bricks.run()
train_list = total_list[:len(total_list)-test_runs]
test_list = total_list[-test_runs:]
print(test_list)

def prepare_training(start):
    """Concatenates all the training data and adds the dates from the optimization up to the end of training."""
    for file in os.listdir(base_dir):
        start_date = file.split('-')[5]
        end_date = file.split('-')[6]
        if "Complete" in file and "Filtered" not in file and start_date == start:
            dataframe = pd.read_csv(base_dir + '/' + file)
            dataframe['Range'] = start_date + ' to ' + end_date
            dataframe = movecol(dataframe,['Range'],'Pass',place='Before')
            return dataframe

def prepare_test(start):
    """Concatenates all the training data and adds the dates from the optimization up to the end of testing."""
    for file in os.listdir(base_dir):
        start_date = file.split('-')[5]
        end_date = file.split('-')[6]
        if "Complete" in file and "Filtered" not in file and start_date == start:
            dataframe = pd.read_csv(base_dir + '/' + file)
            dataframe['Range'] = start_date + ' to ' + end_date
            dataframe = movecol(dataframe,['Range'],'Pass',place='Before')
            return dataframe

def get_training():
    training_dataframe = pd.DataFrame()
    for step in train_list:
        training_dataframe = training_dataframe.append(prepare_training(step[0]))
    training_dataframe = training_dataframe.dropna()  # drop nan values
    training_dataframe = training_dataframe.apply(preprocessing.LabelEncoder().fit_transform)
    norm_num_dataframe = training_dataframe.select_dtypes(include= [np.number])
    training_dataframe = (norm_num_dataframe - norm_num_dataframe.min()) / (norm_num_dataframe.max() - norm_num_dataframe.min())
    training_target = training_dataframe.pop('Rank Forward')  # extract targets
    columns_list = list(training_dataframe)
    forward_columns = [c for c in columns_list if 'Forward' in c]
    training_dataframe = training_dataframe.drop(columns=forward_columns)  # drop forward columns
    training_dataframe = training_dataframe.drop(columns='Back Result')  # drop back result is irrelevant
    training_dataframe = training_dataframe.dropna(axis=1, how='all')  # reassure no empty columns
    #training_dataframe.to_csv(save_dir + '/' + 'Training_data.csv')
    #training_target.to_csv(save_dir + '/' + 'Training_target.csv')
    return training_dataframe, training_target

def get_testing():
    testing_dataframe = pd.DataFrame()
    for step in test_list:
        testing_dataframe = testing_dataframe.append(prepare_test(step[0]))
    testing_dataframe = testing_dataframe.dropna()  # drop nan values
    testing_dataframe = testing_dataframe.apply(preprocessing.LabelEncoder().fit_transform)
    norm_num_dataframe = testing_dataframe.select_dtypes(include=[np.number, np.float])
    testing_dataframe = (norm_num_dataframe - norm_num_dataframe.min()) / (norm_num_dataframe.max() - norm_num_dataframe.min())
    testing_target = testing_dataframe.pop('Rank Forward')  # extract targets
    columns_list = list(testing_dataframe)
    forward_columns = [c for c in columns_list if 'Forward' in c]
    testing_dataframe = testing_dataframe.drop(columns=forward_columns)  # drop forward columns
    testing_dataframe = testing_dataframe.drop(columns='Back Result')  # drop back result is irrelevant
    testing_dataframe = testing_dataframe.dropna(axis=1, how='all')  # reassure no empty columns
    #testing_dataframe.to_csv(save_dir + '/' + 'Testing_data.csv')
    #testing_target.to_csv(save_dir + '/' + 'Testing_target.csv')
    return testing_dataframe, testing_target



def create_model(inputtrain, optimizer='adam', firstlayer=200, secondlayer=200, thirdlayer=200, finallayer=1, activation='tanh', init_mode='uniform'):
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(firstlayer, input_shape=(inputtrain.shape[1], ), kernel_initializer=init_mode, activation=activation, name='First_Layer'),
    tf.keras.layers.Dense(secondlayer, activation=activation, kernel_initializer=init_mode, name='Second_Layer'),
    tf.keras.layers.Dense(secondlayer, activation=activation, kernel_initializer=init_mode, name='Third_Layer'),
    tf.keras.layers.Dense(thirdlayer, activation=activation, kernel_initializer=init_mode, name='Fourth_Layer'),
    tf.keras.layers.Dense(finallayer, kernel_initializer=init_mode, name='Fifth_Layer'),
  ])
  model.compile(optimizer=optimizer, loss='mse', metrics=['mse'], learning_rate=0.1)
  return model


print('B4')
def run_model():
    print('Run model')
    training_dataframe = get_training()[0]
    np_training_dataframe = training_dataframe.values
    testing_dataframe = get_testing()[0]
    np_testing_dataframe = testing_dataframe.values
    training_target = get_training()[1]
    np_training_target = training_target.values
    maximo = np.max(np_training_target)
    testing_target = get_testing()[1]
    np_testing_target = testing_target.values
    model = create_model(inputtrain=np_training_dataframe,optimizer='adam')
    model.summary()
    pesos = model.get_weights()
    #print('this is pesos before', pesos)
    #history = model.fit(x=np_training_dataframe,y=np_training_target, batch_size=100, epochs=150, verbose=2, shuffle=False, validation_data=(np_testing_dataframe, np_testing_target))
    #prediction = model.predict()

    def grid_search():
        print('start grid search')
        batch_size_grid = [20, 50, 100]
        epochs_grid = [50, 150]
        optimizer_grid = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
        #learn_rate_grid = [0.001, 0.01, 0.1, 0.2, 0.3]
        #init_mode_grid = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
        #activation_grid = ['relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
        #first_neurons_grid = [50, 100]
        train = np_training_dataframe
        target = np_training_target
        param_grid = dict(batch_size=batch_size_grid)
        model = KerasRegressor(build_fn=create_model(inputtrain=train))
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=2, verbose=2, scoring='neg_mean_squared_error')
        grid_result = grid.fit(X=train, y=target)
        print('Reached Here')
        #print('Best Score:', grid_result.best_score_)
    grid_search()
    plt.figure(1)
    plt.plot(np.sqrt(history.history['loss']))
    plt.plot(np.sqrt(history.history['val_loss']))
    plt.title('Perdidas de Modelo')
    plt.ylabel('Perdidas')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

run_model()
print('Final Print')



#steps_tested = create_steps_list()
#print("This is the Step List:", steps_tested)
#Benchmark = create_benchmark_list()
#training_steps()