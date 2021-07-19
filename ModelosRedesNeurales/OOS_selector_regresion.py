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

    def __init__(self, train_start, test_start, sequest_start, train_steps):
        self.train_start = train_start
        self.test_start = test_start
        self.sequest_start = sequest_start
        self.train_steps = train_steps


    def split_train_test_sequest_bricks(self):
        """Creates the timelist that is used to split train, test and sequestered"""
        total_bricks = CreateTimebricks(self.train_start, 1, 48, 12, 0, self.sequest_start)
        self.split_train_test_sequest_list = total_bricks.run()
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