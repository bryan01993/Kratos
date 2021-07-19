import numpy as np
import pandas as pd
import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import regularizers
from sklearn.model_selection import GridSearchCV
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from services.create_timebricks import CreateTimebricks
from services.helpers import movecol
from sklearn import preprocessing
import matplotlib.pyplot as plt


### Rutas a los directorios de data, de guardado de resultados y de tensores para TensorBoard
# base_dir = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/EA-B1v2/GBPJPY/M15/WF_Report'
# save_dir = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/EA-B1v2/GBPJPY/M15/'
# tensor_dir = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/EA-B1v2/GBPJPY/M15/Tensorlogs/'
save_dir = '/home/miguel/Proyectos/kratos/Data/GBPJPY/M15/'
base_dir = os.path.join(save_dir, 'WF_Report')
tensor_dir = os.path.join(save_dir, 'Tensorlogs')


### Fechas de Entrenamiento y Validacion, cortes en uso de clase TimeBricks
train_start = '2007.01.01'
test_start = '2015.01.01'
sequest_start = '2020.12.01'
train_steps = 30
#### TO DO
# Create a list separated by bricks to split training and validation (not sequestered) DONE
# Load the Optimization Data for training and validation (not sequestered) DONE
# Add Optimization Range to all csv files from optimization includes validation (not sequestered) DONE
# Concatenate training data in a file, and concatenate validation data in a file (not sequestered) DONE
# Preprocessing and Normalization for training and validation DONE
# Create Tensorflow model
# Train model   OR    Grid search train    (both with validation)
# Graph Results and callbacks
#
class SelectorRegression:
    """DNN to Forecast the best possibility of high walk forward values"""

    def __init__(self,train_start, test_start, sequest_start, train_steps):
        self.train_start = train_start
        self.test_start = test_start
        self.sequest_start = sequest_start
        self.train_steps = train_steps

    def split_train_test_sequest_bricks(self):
        """Creates the timelist that is used to split train, test and sequestered"""
        self.total_bricks = CreateTimebricks(self.train_start, 1, 48, 12, 0, self.sequest_start)
        self.split_train_test_sequest_list = self.total_bricks.run()
        self.train_list = self.split_train_test_sequest_list[:len(self.split_train_test_sequest_list) - self.train_steps]
        self.test_list = self.split_train_test_sequest_list[-self.train_steps:]
        return self.split_train_test_sequest_list

    def add_range(self, look_start):
        """Adds the Range of dates that the optimization used"""
        for file in os.listdir(base_dir):
            self.file_start_date = file.split('-')[5]
            self.file_end_date = file.split('-')[6]
            if 'Complete' in file and 'Filtered' not in file and self.file_start_date == look_start:
                self.dataframe = pd.read_csv(base_dir + '/' + file)
                self.dataframe['Range'] = self.file_start_date + ' to ' + self.file_end_date
                self.dataframe = movecol(self.dataframe, ['Range'], 'Pass', place='Before')
                #self.dataframe.to_csv(save_dir + '/' + '{} {}_data_last_step.csv'.format(look_start,phase_split))   # For debugging purposes
                return self.dataframe

    def concatenate_phase(self, phase_list, Target):
        """Concatenates the phase and pops the target value"""
        concatenated_dataframe = pd.DataFrame()
        for step in phase_list:
            concatenated_dataframe = concatenated_dataframe.append(self.add_range(step[0]))
        concatenated_dataframe = concatenated_dataframe.dropna(axis=1, how='all')    #drop nan values
        try:
            concatenated_target = concatenated_dataframe.pop(Target)
        except:
            print(Target)


        le = preprocessing.LabelEncoder()
        concatenated_dataframe['Range'] = le.fit_transform(concatenated_dataframe['Range'])
        columns_list = list(concatenated_dataframe)
        forward_columns = [c for c in columns_list if 'Forward' in c]
        concatenated_dataframe = concatenated_dataframe.drop(columns=forward_columns) # drop forward columns to prevent look ahead bias
        concatenated_dataframe = concatenated_dataframe.drop(columns='Back Result')  # drop back result is irrelevant
        return concatenated_dataframe, concatenated_target

    def normalize_dataframe(self,raw_dataframe, norm_type):
        """Applies a Normalization to the dataframe to pass to the model"""
        if norm_type == 'Median':
            raw_dataframe = (raw_dataframe-raw_dataframe.mean())/raw_dataframe.std()
        elif norm_type == 'MaxMin':
            raw_dataframe = (raw_dataframe - raw_dataframe.min()) / (raw_dataframe.max() - raw_dataframe.min())
        else:
            print("Select a normalization type between Median and MaxMin")
        return raw_dataframe

    def create_basic_model(self, input_dimension):
        """Here the Model is created"""
        self.optimizer = 'adam'
        self.init_mode = 'uniform'
        self.activation = 'tanh'
        self.dropout_rate = 0.5
        self.wd = 1e-5   # do not remember what this was, weight something
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(1200, input_shape=(input_dimension, ), kernel_regularizer=regularizers.l2(self.wd), activation=self.activation, name='First_Layer'),
            tf.keras.layers.Dense(1000, kernel_regularizer=regularizers.l2(self.wd), activation=self.activation, name='Second_Layer'),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(800, kernel_regularizer=regularizers.l2(self.wd), activation=self.activation, name='Third_Layer'),
            tf.keras.layers.Dense(600, kernel_regularizer=regularizers.l2(self.wd), activation=self.activation, name='a'),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(400, kernel_regularizer=regularizers.l2(self.wd), activation=self.activation, name='b'),
            tf.keras.layers.Dense(200, kernel_regularizer=regularizers.l2(self.wd), activation=self.activation, name='c'),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(100, kernel_regularizer=regularizers.l2(self.wd), activation=self.activation, name='Fourth_Layer'),
            tf.keras.layers.Dense(1, kernel_initializer=self.init_mode, name='Fifth_Layer'),
        ])
        self.model.compile(optimizer=self.optimizer, loss='mae', metrics='mae')
        self.model.optimizer.learning_rate.assign(0.001)

    def run(self):
        test_run = self.split_train_test_sequest_bricks()
        self.train_dataframe = self.concatenate_phase(phase_list = self.train_list, Target="CustomForward")[0]
        #print(len(self.train_dataframe.columns))
        self.train_target = self.concatenate_phase(phase_list = self.train_list, Target="CustomForward")[1]
        #self.train_dataframe[0].to_csv(save_dir + '/ some_concatenated_data.csv')     #  For Debugging
        #self.train_dataframe[1].to_csv(save_dir + '/ some_concatenated_target.csv')   #  For Debugging
        self.norm_train_dataframe = self.normalize_dataframe(self.train_dataframe, 'Median')
        self.norm_train_target = self.normalize_dataframe(self.train_target, 'Median')
        #self.norm_train_dataframe.to_csv(save_dir + '/ some_norm_data.csv')    #  For Debugging
        print('train done')
        validation_dataframe = self.concatenate_phase(phase_list = self.test_list, Target="CustomForward")[0]
        validation_targets = self.concatenate_phase(phase_list = self.test_list, Target="CustomForward")[1]
        #some_other_dataframe[0].to_csv(save_dir + '/ other_concatenated_data.csv')   #  For Debugging
        #some_other_dataframe[1].to_csv(save_dir + '/other_concatenated_target.csv')  #  For Debugging
        # other_norm_dataframe = self.normalize_dataframe(some_other_dataframe[0], 'Median')
        #other_norm_dataframe.to_csv(save_dir + '/ some_other_norm_data.csv')   #  For Debugging
        print('testing done')
        print('Enter basic model')
        self.create_basic_model(input_dimension=len(self.train_dataframe.columns))

        callback_path = os.path.join(save_dir + 'savedmodel.ckpt')

        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensor_dir + 'logs', histogram_freq=1)
        model_callbacks = tf.keras.callbacks.ModelCheckpoint(filepath=callback_path, save_best_only=True, verbose=2)

        history = self.model.fit(
            x=self.train_dataframe,
            y=self.train_target,
            batch_size=5000,
            epochs=15,
            verbose=2,
            shuffle=False,
            validation_data=(validation_dataframe, validation_targets),
            callbacks=[tensorboard, model_callbacks]
        )

        ### Plots the Training and Validation
        plt.figure(1)
        plt.plot(np.sqrt(history.history['loss']))
        plt.plot(np.sqrt(history.history['val_loss']))
        plt.title('Perdidas de Modelo')
        plt.ylabel('Perdidas')
        plt.xlabel('Epochs')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()


smth = SelectorRegression(train_start, test_start, sequest_start, train_steps)
smth.run()











def prepare_training(start):
    """Concatenates all the training data and adds the dates from the optimization up to the end of training."""
    for file in os.listdir(base_dir):
        start_date = file.split('-')[5]
        end_date = file.split('-')[6]
        if "Complete" in file and "Filtered" not in file and start_date == start:
            dataframe = pd.read_csv(base_dir + '/' + file)
            dataframe['Range'] = start_date + ' to ' + end_date
            dataframe = movecol(dataframe,['Range'],'Pass',place='Before')
            dataframe.to_csv(save_dir + '/' + 'Training_data_last_step.csv')
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
            dataframe.to_csv(save_dir + '/' + 'Test_data_last_step.csv')
            return dataframe

def get_training():
    """Data Treatment, Normalization and Separation of Properties and Target for Training"""
    training_dataframe = pd.DataFrame()
    for step in train_list:
        training_dataframe = training_dataframe.append(prepare_training(step[0]))
    training_dataframe = training_dataframe.dropna()  # drop nan values
    le = preprocessing.LabelEncoder()
    training_dataframe['Range'] = le.fit_transform(training_dataframe['Range'])
    training_dataframe.to_csv(save_dir + '/' + '1-{}-Training_data_norm.csv'.format(step))
    training_target = training_dataframe.pop('CustomForward')
    training_target.to_csv(save_dir + '/' + '2-{}-Training_target_norm.csv'.format(step))
    norm_num_dataframe = training_dataframe.select_dtypes(include= [np.number, np.float])
    training_dataframe = (norm_num_dataframe-norm_num_dataframe.mean())/norm_num_dataframe.std()     # Normalize in Standard Deviations
    columns_list = list(training_dataframe)
    forward_columns = [c for c in columns_list if 'Forward' in c]
    training_dataframe = training_dataframe.drop(columns=forward_columns)  # drop forward columns
    training_dataframe = training_dataframe.drop(columns='Back Result')  # drop back result is irrelevant
    training_dataframe = training_dataframe.dropna(axis=1, how='all')  # reassure no empty columns

    return training_dataframe, training_target

def get_testing():
    """Data Treatment, Normalization and Separation of Properties and Target for Validation"""
    testing_dataframe = pd.DataFrame()
    for step in test_list:
        testing_dataframe = testing_dataframe.append(prepare_test(step[0]))
    testing_dataframe = testing_dataframe.dropna()  # drop nan values
    le = preprocessing.LabelEncoder()
    testing_dataframe['Range'] = le.fit_transform(testing_dataframe['Range'])
    testing_dataframe.to_csv(save_dir + '/' + '3-{}-Testing_data_norm.csv'.format(step))
    #testing_dataframe = testing_dataframe.apply(preprocessing.LabelEncoder().fit_transform)
    norm_num_dataframe = testing_dataframe.select_dtypes(include=[np.number, np.float])
    testing_dataframe = (norm_num_dataframe - norm_num_dataframe.mean()) / norm_num_dataframe.std()        # Normalize in Standard Deviations
    testing_target = testing_dataframe.pop('CustomForward')  # extract targets, you select the target column here
    testing_target.to_csv(save_dir + '/' + '4-{}-Testing_target_norm.csv'.format(step))
    columns_list = list(testing_dataframe)
    forward_columns = [c for c in columns_list if 'Forward' in c]
    testing_dataframe = testing_dataframe.drop(columns=forward_columns)  # drop forward columns
    testing_dataframe = testing_dataframe.drop(columns='Back Result')  # drop back result is irrelevant
    testing_dataframe = testing_dataframe.dropna(axis=1, how='all')  # reassure no empty columns

    return testing_dataframe, testing_target

def data_sequester(csvpath):
    """Extraction of Data Outside of Training and Validation for True OOS Testing"""
    sequest_dataframe = pd.read_csv(csvpath)
    print('Testing len before drop:', len(sequest_dataframe))
    sequest_dataframe = sequest_dataframe.dropna()  # drop nan values
    print('Testing len after drop:', len(sequest_dataframe))
    print(sequest_dataframe)
    le = preprocessing.LabelEncoder()
    #sequest_dataframe['Range'] = le.fit_transform(sequest_dataframe['Range'])
    #sequest_dataframe = sequest_dataframe.apply(preprocessing.LabelEncoder().fit_transform)
    norm_num_dataframe = sequest_dataframe.select_dtypes(include=[np.number, np.float])
    sequest_dataframe = (norm_num_dataframe - norm_num_dataframe.min()) / (norm_num_dataframe.max() - norm_num_dataframe.min())  # Normalize in Standard Deviations
    sequest_dataframe['Range'] = 0
    sequest_dataframe = movecol(sequest_dataframe, ['Range'], 'Pass', place='Before')
    sequest_target = sequest_dataframe.pop('CustomForward')  # extract targets
    columns_list = list(sequest_dataframe)
    forward_columns = [c for c in columns_list if 'Forward' in c]
    sequest_dataframe = sequest_dataframe.drop(columns=forward_columns)  # drop forward columns
    sequest_dataframe = sequest_dataframe.drop(columns='Back Result')  # drop back result is irrelevant
    sequest_dataframe = sequest_dataframe.dropna(axis=1, how='all')  # reassure no empty columns
    sequest_dataframe.to_csv(save_dir + '/' + 'Sequest_data.csv')
    sequest_target.to_csv(save_dir + '/' + 'Sequest_target.csv')
    print(sequest_dataframe)
    return sequest_dataframe, sequest_target

def create_model(inputtrain, optimizer='adam', firstlayer=5000, secondlayer=2000, thirdlayer=1000, finallayer=1, activation='tanh', init_mode='uniform'):
  """Creates the Model, to be HyperParametrized and tested"""
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(600, input_shape=(inputtrain.shape[1], ), activation=activation, name='First_Layer'),
    tf.keras.layers.Dense(500, name='Second_Layer'),
    tf.keras.layers.Dense(400, name='Third_Layer'),
    tf.keras.layers.Dense(300, name='a'),
    tf.keras.layers.Dense(200, name='b'),
    tf.keras.layers.Dense(100, name='c'),
    tf.keras.layers.Dense(50, name='Fourth_Layer'),
    tf.keras.layers.Dense(1, kernel_initializer=init_mode, name='Fifth_Layer'),
  ])
  model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
  return model

def create_regulated_model( inputtrain, wd, rate, optimizer='adam', activation='tanh', init_mode='uniform'):    #weight decay and dropout rate
    """Creates the Model with L2 and dropout regulators , to be HyperParametrized and tested"""
    model = tf.keras.Sequential([
    tf.keras.layers.Dense(600, input_shape=(inputtrain.shape[1], ),kernel_regularizer=regularizers.l2(wd), activation=activation, name='First_Layer'),
    tf.keras.layers.Dense(500,kernel_regularizer=regularizers.l2(wd), activation=activation, name='Second_Layer'),
    tf.keras.layers.Dropout(rate),
    tf.keras.layers.Dense(400,kernel_regularizer=regularizers.l2(wd), activation=activation, name='Third_Layer'),
    tf.keras.layers.Dense(300,kernel_regularizer=regularizers.l2(wd), activation=activation, name='a'),
    tf.keras.layers.Dropout(rate),
    tf.keras.layers.Dense(200,kernel_regularizer=regularizers.l2(wd), activation=activation, name='b'),
    tf.keras.layers.Dense(100,kernel_regularizer=regularizers.l2(wd), activation=activation, name='c'),
    tf.keras.layers.Dropout(rate),
    tf.keras.layers.Dense(50,kernel_regularizer=regularizers.l2(wd), activation=activation, name='Fourth_Layer'),
    tf.keras.layers.Dense(1, kernel_initializer=init_mode, name='Fifth_Layer'),
    ])
    model.compile(optimizer=optimizer, loss='mae', metrics='mae')
    model.optimizer.learning_rate.assign(0.001)
    return model

def run_model():
    """Feeds data to the model and fits the model, manages callbacks"""
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
    #model = create_model(inputtrain=np_training_dataframe,optimizer='adam')
    model = create_regulated_model(inputtrain=np_training_dataframe, wd=1e-5, rate=0.5)
    model.summary()
    pesos = model.get_weights()
    callback_path = save_dir + 'savedmodel.ckpt'
    save_model_path = save_dir + 'saved_model.h5'
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensor_dir + 'logs', histogram_freq=1)
    # ejecutar con tensorboard --logdir C:\Users\bryan\AppData\Roaming\MetaQuotes\Terminal\6C3C6A11D1C3791DD4DBF45421BF8028\reports\EA-B1v2\GBPJPY\M15\Tensorlogs\logs en CMD
    model_callbacks = tf.keras.callbacks.ModelCheckpoint(filepath=callback_path, save_best_only=True, verbose=2)
    history = model.fit(x=np_training_dataframe,y=np_training_target, batch_size=5000, epochs=2000, verbose=2, shuffle=False, validation_data=(np_testing_dataframe, np_testing_target), callbacks=[tensorboard, model_callbacks])
    model.save(save_model_path)
    #prediction = model.predict()

    def grid_search():
        """HyperParameter Optimization"""
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


    ### Plots the Training and Validation
    plt.figure(1)
    plt.plot(np.sqrt(history.history['loss']))
    plt.plot(np.sqrt(history.history['val_loss']))
    plt.title('Perdidas de Modelo')
    plt.ylabel('Perdidas')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
def run_trained_model():
    """Shows Predictions for a Validation Slice of data"""
    print('Run Previously Trained Model')
    testing_dataframe = get_testing()[0]
    np_testing_dataframe = testing_dataframe.values
    training_target = get_training()[1]
    np_training_target = training_target.values
    maximo = np.max(np_training_target)
    testing_target = get_testing()[1]
    np_testing_target = testing_target.values
    print('np_testing_target', np_testing_target)
    save_model_path = save_dir + 'saved_model.h5'
    new_model = tf.keras.models.load_model(save_model_path)
    new_model.summary()
    simple_test = new_model.predict(testing_dataframe)
    simple_test_df = pd.DataFrame(simple_test)
    simple_result = testing_target
    save_dataframe = pd.DataFrame(simple_test)
    print('here', simple_result, simple_test_df)
    tempo_path = save_dir + '/' + 'simpleresult.csv'
    simple_result.to_csv(tempo_path)
    tempo_path_test = save_dir + '/' + 'simpletest.csv'
    simple_test_df.to_csv(tempo_path_test)
    simple_result.index = simple_test_df.index
    print('shapes', simple_test_df.index, simple_result.index)
    saveable_dataframe = pd.concat([simple_test_df,simple_result], axis=1, ignore_index=True)
    print(saveable_dataframe.head())
    save_dataframe_predictions = save_dir + '/' + 'predictions.csv'
    saveable_dataframe.to_csv(save_dataframe_predictions)
    print('this is saveable dataframe', saveable_dataframe)


def run_trained_model_sequest(datafile):
    """Runs the model for predictions on Sequestered Data"""
    print('Run Previously Trained Model on sequestered data')
    testing_dataframe = datafile[0]
    np_testing_dataframe = testing_dataframe.values
    testing_target = datafile[1]
    np_testing_target = testing_target.values
    print('np_testing_target', np_testing_target)
    save_model_path = save_dir + 'saved_model.h5'
    new_model = tf.keras.models.load_model(save_model_path)
    new_model.summary()
    simple_test = new_model.predict(testing_dataframe)
    simple_test_df = pd.DataFrame(simple_test)
    simple_result = testing_target
    save_dataframe = pd.DataFrame(simple_test)
    print('here', simple_result, simple_test_df)
    tempo_path = save_dir + '/' + 'simpleresult.csv'
    simple_result.to_csv(tempo_path)
    tempo_path_test = save_dir + '/' + 'simpletest.csv'
    simple_test_df.to_csv(tempo_path_test)
    simple_result.index = simple_test_df.index
    print('shapes', simple_test_df.index, simple_result.index)
    saveable_dataframe = pd.concat([simple_test_df,simple_result], axis=1, ignore_index=True)
    print(saveable_dataframe.head())
    save_dataframe_predictions = save_dir + '/' + 'predictions.csv'
    saveable_dataframe.to_csv(save_dataframe_predictions)
    print('this is saveable dataframe', saveable_dataframe)


#run_model()
#run_trained_model()
#first_sequester = base_dir + '/' + 'OptiWFResults-EA-B1v2-GBPJPY-M15-2016.1.1-2021.1.1-Complete.csv'
#datafile = data_sequester(first_sequester)
#run_trained_model_sequest(datafile)
#print('Final Print')



#steps_tested = create_steps_list()
#print("This is the Step List:", steps_tested)
#Benchmark = create_benchmark_list()
#training_steps()