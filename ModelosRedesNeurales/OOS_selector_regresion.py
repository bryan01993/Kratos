import numpy as np
import pandas as pd
import os
import sys
import tensorflow as tf
import kerastuner as kt
from tensorflow.keras import regularizers
from sklearn import preprocessing
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from kerastuner import RandomSearch
from keras_tuner import HyperParameters
from keras_tuner.applications import HyperXception



CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from services.create_timebricks import CreateTimebricks
from services.helpers import movecol
from ModelosRedesNeurales.my_custom_callback import MyCustomCallback





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

    def build_basic_model(self, input_dimension):
        """Here the Model is created"""
        optimizer = 'adam'
        init_mode = 'uniform'
        activation = 'tanh'
        dropout_rate = 0.5
        wd = 1e-5

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1200, input_shape=(input_dimension, ), kernel_regularizer=regularizers.l2(wd), activation=activation, name='First_Layer'),
            tf.keras.layers.Dense(1000, kernel_regularizer=regularizers.l2(wd), activation=activation, name='Second_Layer'),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(800, kernel_regularizer=regularizers.l2(wd), activation=activation, name='Third_Layer'),
            tf.keras.layers.Dense(600, kernel_regularizer=regularizers.l2(wd), activation=activation, name='a'),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(400, kernel_regularizer=regularizers.l2(wd), activation=activation, name='b'),
            tf.keras.layers.Dense(200, kernel_regularizer=regularizers.l2(wd), activation=activation, name='c'),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(100, kernel_regularizer=regularizers.l2(wd), activation=activation, name='Fourth_Layer'),
            tf.keras.layers.Dense(1, kernel_initializer=init_mode, name='Fifth_Layer'),
        ])

        model.compile(optimizer=optimizer, loss='mae', metrics='mae')
        model.optimizer.learning_rate.assign(0.001)

        return model

    def build_model(self, hp):
        test_run = self.split_train_test_sequest_bricks()
        self.train_dataframe = self.concatenate_phase(phase_list = self.train_list, Target="CustomForward")[0]
        self.train_target = self.concatenate_phase(phase_list = self.train_list, Target="CustomForward")[1]
        self.norm_train_dataframe = self.normalize_dataframe(self.train_dataframe, 'Median')
        self.norm_train_target = self.normalize_dataframe(self.train_target, 'Median')
        validation_dataframe = self.concatenate_phase(phase_list = self.test_list, Target="CustomForward")[0]
        validation_targets = self.concatenate_phase(phase_list = self.test_list, Target="CustomForward")[1]

        input_dimension=len(self.train_dataframe.columns)
        model = tf.keras.Sequential()

        activation = 'tanh'
        wd = 1e-5

        model.add(layers.Dense(1200, input_shape=(input_dimension, ), kernel_regularizer=regularizers.l2(wd), activation=activation, name='First_Layer'))

        for i in range(hp.Int("num_layers", 2, 20)):
            print("aaaaaaaaaaaaaaaaaaaaaaa")

            model.add(
                layers.Dense(
                    units=hp.Int("units_" + str(i), min_value=32, max_value=512, step=32),
                    activation="relu",
                )
            )

        optimizer=tf.keras.optimizers.Adam(hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])),
        model.add(layers.Dense(1, activation="tanh"))
        model.compile(optimizer=optimizer, loss='mae', metrics='mae')

        return model

    def run_tuner(self):

        hp = HyperParameters()

        # This will override the `learning_rate` parameter with your own selection of choices
        hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        tuner = RandomSearch(
            self.build_model,
            objective="mae",
            max_trials=3,
            executions_per_trial=2,
            overwrite=True,
            directory="my_dir",
            project_name="helloworld",
        )

        validation_dataframe = self.concatenate_phase(phase_list = self.test_list, Target="CustomForward")[0]
        validation_targets = self.concatenate_phase(phase_list = self.test_list, Target="CustomForward")[1]

        tuner.search(self.train_dataframe, self.train_target, epochs=10, validation_data=(validation_dataframe, validation_targets))


    def run(self):
        test_run = self.split_train_test_sequest_bricks()
        self.train_dataframe = self.concatenate_phase(phase_list = self.train_list, Target="CustomForward")[0]
        self.train_target = self.concatenate_phase(phase_list = self.train_list, Target="CustomForward")[1]
        self.norm_train_dataframe = self.normalize_dataframe(self.train_dataframe, 'Median')
        self.norm_train_target = self.normalize_dataframe(self.train_target, 'Median')
        print('train done')
        validation_dataframe = self.concatenate_phase(phase_list = self.test_list, Target="CustomForward")[0]
        validation_targets = self.concatenate_phase(phase_list = self.test_list, Target="CustomForward")[1]
        print('testing done')
        print('Enter basic model')

        callback_path = os.path.join(save_dir + 'savedmodel.ckpt')

        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensor_dir + 'logs', histogram_freq=1)
        model_callbacks = tf.keras.callbacks.ModelCheckpoint(filepath=callback_path, save_best_only=True, verbose=2)

        model = self.build_model(input_dimension=len(self.train_dataframe.columns))
        history = model.fit(
            x=self.train_dataframe,
            y=self.train_target,
            batch_size=5000,
            epochs=15,
            verbose=2,
            shuffle=False,
            validation_data=(validation_dataframe, validation_targets),
            callbacks=[tensorboard, model_callbacks]
        )

        plt.figure(1)
        plt.plot(np.sqrt(history.history['loss']))
        plt.plot(np.sqrt(history.history['val_loss']))
        plt.title('Perdidas de Modelo')
        plt.ylabel('Perdidas')
        plt.xlabel('Epochs')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()


smth = SelectorRegression(train_start, test_start, sequest_start, train_steps)
# smth.run()
smth.run_tuner()

