import numpy as np
import pandas as pd
import os
import sys
import time
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras import regularizers
from sklearn import preprocessing
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from keras_tuner import RandomSearch
from keras_tuner import HyperParameters

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from services.create_timebricks import CreateTimebricks
from services.helpers import movecol
#from ModelosRedesNeurales.my_custom_callback import MyCustomCallback



### Rutas a los directorios de data, de guardado de resultados y de tensores para TensorBoard
LOG_DIR = f"{int(time.time())}"
base_dir = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/EA-B1v2/GBPJPY/M15/WF_Report'
save_dir = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/EA-B1v2/GBPJPY/M15/'
tensor_dir = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/EA-B1v2/GBPJPY/M15/Tensorlogs/'
# save_dir = '/home/miguel/Proyectos/kratos/Data/GBPJPY/M15/'
# base_dir = os.path.join(save_dir, 'WF_Report')
# tensor_dir = os.path.join(save_dir, 'Tensorlogs')

# tensorboard --logdir /home/miguel/Proyectos/kratos/Data/GBPJPY/M15/Tensorlogslogs

### Fechas de Entrenamiento y Validacion, cortes en uso de clase TimeBricks
train_start = '2007.01.01'
test_start = '2015.01.01'
sequest_start = '2020.12.01'
train_steps = 30
#### TO DO

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
        print("Before cutting: ", len(concatenated_dataframe))
        concatenated_dataframe.drop(concatenated_dataframe[concatenated_dataframe['Result'] <= 0].index, inplace=True) #drops custom values below 0
        print("After cutting: ", len(concatenated_dataframe))
        try:
            concatenated_target = concatenated_dataframe.pop(Target)
        except:
            print(Target)
        concatenated_dataframe.to_csv(save_dir+'/cutlosers.csv')
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
        activation = 'relu'
        dropout_rate = 0.5
        wd = 1e-9

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_shape=(input_dimension, ), kernel_regularizer=regularizers.l2(wd), activation='relu', name='first'),
            tf.keras.layers.Dense(2000, activation=activation, kernel_regularizer=regularizers.l2(wd), name='second'),
            tf.keras.layers.Dense(2000, activation=activation, kernel_regularizer=regularizers.l2(wd), name='third'),
            #tf.keras.layers.Dense(1440, kernel_regularizer=regularizers.l2(wd), activation=activation, name='c'),
            #tf.keras.layers.Dense(1984, kernel_regularizer=regularizers.l2(wd), activation=activation, name='Fourth_Layer'),
            #tf.keras.layers.Dense(928, kernel_regularizer=regularizers.l2(wd), activation=activation),
            #tf.keras.layers.Dense(704, kernel_regularizer=regularizers.l2(wd), activation=activation),
            #tf.keras.layers.Dense(1888, kernel_regularizer=regularizers.l2(wd), activation=activation),
            #tf.keras.layers.Dense(1664, kernel_regularizer=regularizers.l2(wd), activation=activation),
            #tf.keras.layers.Dense(160, kernel_regularizer=regularizers.l2(wd), activation=activation),
            #tf.keras.layers.Dense(128, kernel_regularizer=regularizers.l2(wd), activation=activation),
            #tf.keras.layers.Dense(1152, kernel_regularizer=regularizers.l2(wd), activation=activation),
            #tf.keras.layers.Dense(224, kernel_regularizer=regularizers.l2(wd), activation=activation),
            #tf.keras.layers.Dense(1856, kernel_regularizer=regularizers.l2(wd), activation=activation),
            #tf.keras.layers.Dense(2016, kernel_regularizer=regularizers.l2(wd), activation=activation),
            #tf.keras.layers.Dense(544, kernel_regularizer=regularizers.l2(wd), activation=activation),
            #tf.keras.layers.Dense(1024, kernel_regularizer=regularizers.l2(wd), activation=activation),
            #tf.keras.layers.Dense(288, kernel_regularizer=regularizers.l2(wd), activation=activation),
            #tf.keras.layers.Dense(896, kernel_regularizer=regularizers.l2(wd), activation=activation),
            #tf.keras.layers.Dense(160, kernel_regularizer=regularizers.l2(wd), activation=activation),
            #tf.keras.layers.Dense(1504, kernel_regularizer=regularizers.l2(wd), activation=activation),
            #tf.keras.layers.Dense(352, kernel_regularizer=regularizers.l2(wd), activation=activation),
            #tf.keras.layers.Dense(96, kernel_regularizer=regularizers.l2(wd), activation=activation),
            #tf.keras.layers.Dense(1184, kernel_regularizer=regularizers.l2(wd), activation=activation),
            #tf.keras.layers.Dense(2048, kernel_regularizer=regularizers.l2(wd), activation=activation),
            #tf.keras.layers.Dense(192, kernel_regularizer=regularizers.l2(wd), activation=activation),
            #tf.keras.layers.Dense(192, kernel_regularizer=regularizers.l2(wd), activation=activation),
            #tf.keras.layers.Dense(1632, kernel_regularizer=regularizers.l2(wd), activation=activation),
            #tf.keras.layers.Dense(1888, kernel_regularizer=regularizers.l2(wd), activation=activation),
            #tf.keras.layers.Dense(1088, kernel_regularizer=regularizers.l2(wd), activation=activation),
            #tf.keras.layers.Dense(416, kernel_regularizer=regularizers.l2(wd), activation=activation),
            #tf.keras.layers.Dense(1184, kernel_regularizer=regularizers.l2(wd), activation=activation),
            #tf.keras.layers.Dense(1600, kernel_regularizer=regularizers.l2(wd), activation=activation),
            #tf.keras.layers.Dense(224, kernel_regularizer=regularizers.l2(wd), activation=activation),
            #tf.keras.layers.Dense(1472, kernel_regularizer=regularizers.l2(wd), activation=activation),
            #tf.keras.layers.Dense(1920, kernel_regularizer=regularizers.l2(wd), activation=activation),
            tf.keras.layers.Dense(1, name='output_layer'),
        ])

        model.compile(optimizer=optimizer, loss='mse', metrics='mse')
        model.optimizer.learning_rate.assign(0.001)

        return model

    def build_model(self, hp):
        test_run = self.split_train_test_sequest_bricks()
        self.train_dataframe = self.concatenate_phase(phase_list = self.train_list, Target="CustomForward")[0]
        self.train_target = self.concatenate_phase(phase_list = self.train_list, Target="CustomForward")[1]
        self.norm_train_dataframe = self.normalize_dataframe(self.train_dataframe, 'Median')
        self.norm_train_target = self.normalize_dataframe(self.train_target, 'Median')
        self.validation_dataframe = self.concatenate_phase(phase_list = self.test_list, Target="CustomForward")[0]
        self.validation_targets = self.concatenate_phase(phase_list = self.test_list, Target="CustomForward")[1]
        self.norm_validation_dataframe = self.normalize_dataframe(self.validation_dataframe, 'Median')
        self.norm_validation_targets = self.normalize_dataframe(self.validation_targets, 'Median')
        input_dimension=len(self.train_dataframe.columns)
        model = tf.keras.Sequential()

        hp_activation = ['elu', 'relu']
        hp_activation = hp.Choice('Activation', hp_activation)

        hp_learning_rate = [0.1, 0.01, 0.001, 0.0001]
        hp_learning_rate = hp.Choice('learningRate', hp_learning_rate)

        hp_num_layers = hp.Int('numLayers', 1, 10)

        hp_weight_decay = hp.Float('weightDecay', 0.001, 0.1, 0.001)

        hp_type_regularization = hp.Choice('regularizer', ['l1', 'l2'])
        regularizer = getattr(regularizers, hp_type_regularization)

        model.add(layers.Dense(24, input_shape=(input_dimension, ), kernel_regularizer=regularizer(hp_weight_decay), activation=hp_activation, name='First_Layer'))
        for i in range(hp_num_layers):
            model.add(
                layers.Dense(
                    units=hp.Int("units_" + str(i), min_value=96, max_value=2048, step=32),
                    activation=hp_activation,
                    kernel_regularizer=regularizer(hp_weight_decay)
                )
            )
        model.add(layers.Dense(1))
        optimizer=tf.keras.optimizers.Adam(hp_learning_rate)
        model.compile(optimizer=optimizer, loss='mae', metrics='mae')
        return model

    def run_tuner(self):
        hp = HyperParameters()
        # This will override the `learning_rate` parameter with your own selection of choices
        #hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        tuner = RandomSearch(
            self.build_model,
            objective="val_mae",
            max_trials=5,
            executions_per_trial=1,
            overwrite=False,
            directory="my_dir",
            project_name="helloworld",
        )

        print("Tuner Summary:", tuner.search_space_summary())
        tuner.search(self.train_dataframe, self.train_target, epochs=10, validation_data=(self.validation_dataframe, self.validation_targets))
        print("Tuner Results Summary:", tuner.results_summary())

    def run(self):
        test_run = self.split_train_test_sequest_bricks()
        self.train_dataframe = self.concatenate_phase(phase_list = self.train_list, Target="CustomForward")[0]
        self.train_target = self.concatenate_phase(phase_list = self.train_list, Target="CustomForward")[1]
        self.norm_train_dataframe = self.normalize_dataframe(self.train_dataframe, 'Median')
        self.norm_train_target = self.normalize_dataframe(self.train_target, 'Median')
        self.validation_dataframe = self.concatenate_phase(phase_list = self.test_list, Target="CustomForward")[0]
        self.validation_targets = self.concatenate_phase(phase_list = self.test_list, Target="CustomForward")[1]
        self.norm_validation_dataframe = self.normalize_dataframe(self.validation_dataframe, 'Median')
        self.norm_validation_targets = self.normalize_dataframe(self.validation_targets, 'Median')
        print('train done')
        validation_dataframe = self.concatenate_phase(phase_list = self.test_list, Target="CustomForward")[0]
        validation_targets = self.concatenate_phase(phase_list = self.test_list, Target="CustomForward")[1]
        print('testing done')
        print('Enter basic model')

        callback_path = os.path.join(save_dir + 'savedmodel.ckpt')
        self.save_model_path = save_dir + 'saved_model.h5'
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensor_dir + 'logs', histogram_freq=1)
        model_callbacks = tf.keras.callbacks.ModelCheckpoint(filepath=callback_path, save_best_only=True, verbose=2)

        model = self.build_basic_model(input_dimension=len(self.train_dataframe.columns))
        history = model.fit(
            x=self.norm_train_dataframe,
            y=self.norm_train_target,
            batch_size=50,
            epochs=10,
            verbose=2,
            shuffle=False,
            validation_data=(self.norm_validation_dataframe, self.norm_validation_targets),
            callbacks=[tensorboard, model_callbacks]
        )
        model.save(self.save_model_path)
        plt.figure(1)
        plt.plot(np.sqrt(history.history['loss']))
        plt.plot(np.sqrt(history.history['val_loss']))
        plt.title('Perdidas de Modelo')
        plt.ylabel('Perdidas')
        plt.xlabel('Epochs')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    def run_sequestered_model(self):
        self.save_model_path = save_dir + 'saved_model.h5'
        self.sequest_list = [['2015.9.1', '2019.9.1', '2020.9.1'], ['2016.1.1', '2020.1.1', '2021.1.1']]
        self.split_train_test_sequest_bricks()
        print('train: ', self.train_list)
        print('test: ', self.test_list)
        print("Start: ", self.sequest_start)
        sequest_data = self.concatenate_phase(phase_list=self.sequest_list, Target="CustomForward")[0]
        sequest_target = self.concatenate_phase(phase_list=self.sequest_list, Target="CustomForward")[1]
        new_model = tf.keras.models.load_model(self.save_model_path)
        predictions = new_model.predict(sequest_data)
        sequest_data.to_csv(save_dir + "sequestered_data.csv")
        sequest_target.to_csv(save_dir + '/sequestered_target.csv')
        dfPredictions = pd.DataFrame(predictions)
        dfPredictions.to_csv(save_dir + '/predictions.csv')
        print(predictions)

smth = SelectorRegression(train_start, test_start, sequest_start, train_steps)
# smth.run()
smth.run_tuner()
# smth.run_sequestered_model()

