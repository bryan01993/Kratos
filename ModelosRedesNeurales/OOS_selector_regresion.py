import numpy as np
import pandas as pd
import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
import seaborn as sns
from tensorflow.keras import regularizers
from sklearn import preprocessing as skpre
import matplotlib.pyplot as plt
from tensorflow.keras import layers, preprocessing
from tensorflow.keras.layers.experimental import preprocessing
from keras_tuner import RandomSearch, BayesianOptimization
from keras_tuner import HyperParameters
###Magic Block
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


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
        #print("Before cutting: ", len(concatenated_dataframe))
        concatenated_dataframe.drop(concatenated_dataframe[concatenated_dataframe['Result'] <= 0].index, inplace=True) #drops custom values below 0
        #print("After cutting: ", len(concatenated_dataframe))
        try:
            concatenated_target = concatenated_dataframe.pop(Target)
        except:
            print(Target)
        #concatenated_dataframe.to_csv(save_dir+'/cutlosers.csv')
        le = skpre.LabelEncoder()
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
        wd = 1e-6

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_shape=(input_dimension, )),
            tf.keras.layers.Dense(2000, activation=activation, kernel_regularizer=regularizers.l2(wd), name='second'),
            tf.keras.layers.Dense(2000, activation=activation, kernel_regularizer=regularizers.l2(wd), name='third'),
            #tf.keras.layers.Dense(1440, kernel_regularizer=regularizers.l2(wd), activation=activation, name='c'),
            #tf.keras.layers.Dense(1984, kernel_regularizer=regularizers.l2(wd), activation=activation, name='Fourth_Layer'),
            tf.keras.layers.Dense(1, name='output_layer'),
        ])

        model.compile(optimizer=optimizer, loss='mae', metrics='mae')
        model.optimizer.learning_rate.assign(0.01)

        return model

    def build_model(self, hp):

        model = tf.keras.Sequential()
        hp_activation = ['elu', 'relu']  #, 'relu'
        hp_activation = hp.Choice('Activation', hp_activation)
        hp_learning_rate = [0.01, 0.001, 0.0001]            #  0.001, 0.0001
        hp_learning_rate = hp.Choice('learningRate', hp_learning_rate)
        hp_num_layers = hp.Int('numLayers', 8, 12)
        hp_weight_decay = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01] #0.0001, 0.001, 0.01
        hp_weight_decay = hp.Choice('weightDecay', hp_weight_decay)
        hp_type_regularization = hp.Choice('regularizer', ['l1'])  #, 'l2'
        regularizer = getattr(regularizers, hp_type_regularization)
        weight_initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=1.0)
        data = np.array(self.train_dataframe)
        data_normalizer = preprocessing.Normalization( axis=-1)
        data_normalizer.adapt(data)
        self.tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensor_dir + 'logs/{}-{}-{}-{}-{}'.format(hp_activation ,hp_learning_rate ,hp_num_layers ,hp_weight_decay, time.time()), histogram_freq=1)
        model.add(layers.Dense(24, input_shape=(len(self.norm_train_dataframe.columns), )))
        for i in range(hp_num_layers):
            model.add(
                layers.Dense(
                    units=hp.Int("units_" + str(i), min_value=400, max_value=1024, step=24),
                    activation=hp_activation,
                    kernel_regularizer=regularizer(hp_weight_decay),
                    #kernel_initializer= weight_initializer
                )
            )
        model.add(layers.Dense(1))
        optimizer=tf.keras.optimizers.Adam(hp_learning_rate)
        model.compile(optimizer=optimizer, loss='mae', metrics='mae')
        #model.summary()
        return model

    def run_tuner(self):
        hp = HyperParameters()
        test_run = self.split_train_test_sequest_bricks()
        self.train_dataframe = self.concatenate_phase(phase_list = self.train_list, Target="CustomForward")[0]
        self.train_target = self.concatenate_phase(phase_list = self.train_list, Target="CustomForward")[1]
        self.norm_train_dataframe = self.normalize_dataframe(self.train_dataframe, 'Median')
        #print("This is Training dataframe normalized", '\n', self.norm_train_dataframe.head())
        training_data_description = self.norm_train_dataframe.describe()
        #print("This is Training Data: ", "\n", training_data_description)
        #training_data_description.to_csv(save_dir + 'training_data_description.csv')
        self.norm_train_target = self.normalize_dataframe(self.train_target, 'Median')
        training_target_description = self.norm_train_target.describe()
        #print("This is Training Target : ", "\n",training_target_description)
        #training_target_description.to_csv(save_dir + 'training_target_description.csv')
        #print("This is Training target normalized", '\n', self.norm_train_target.head())
        self.validation_dataframe = self.concatenate_phase(phase_list = self.test_list, Target="CustomForward")[0]
        self.validation_targets = self.concatenate_phase(phase_list = self.test_list, Target="CustomForward")[1]
        self.norm_validation_dataframe = self.normalize_dataframe(self.validation_dataframe, 'Median')
        self.norm_validation_targets = self.normalize_dataframe(self.validation_targets, 'Median')
        validation_data_description = self.norm_validation_dataframe.describe()
        #print("This is Validation Data: ", "\n", validation_data_description)
        #validation_data_description.to_csv(save_dir + 'validation_data_description.csv')
        validation_target_description = self.norm_validation_targets.describe()
        #print("This is Validation Target : ", "\n", validation_target_description)
        #validation_target_description.to_csv(save_dir + 'validation_target_description.csv')
        self.input_dimension=len(self.norm_train_dataframe.columns)
        # This will override the `learning_rate` parameter with your own selection of choices
        self.tuner = BayesianOptimization(
            self.build_model,
            objective="val_loss",
            max_trials=10,
            executions_per_trial=1,
            overwrite=True,
            directory="my_dir",
            project_name="Kratos",
        )

        print("Tuner Summary:", self.tuner.search_space_summary())
        self.tuner.search(self.norm_train_dataframe, self.train_target, epochs=30, validation_data=(self.norm_validation_dataframe, self.validation_targets), shuffle=True, callbacks = [self.tensorboard])
        print("Tuner Results Summary:", self.tuner.results_summary())

    def single_input_model(self):
        """Creates a Linear Regression Model based only on 1 feature"""
        print("Starting Single Input Model")
        test_run = self.split_train_test_sequest_bricks()
        self.train_dataframe = self.concatenate_phase(phase_list = self.train_list, Target="CustomForward")[0]
        self.train_target = self.concatenate_phase(phase_list = self.train_list, Target="CustomForward")[1]
        normalizer = preprocessing.Normalization(axis=-1)
        normalizer.adapt(np.array(self.train_dataframe))
        profit = np.array(self.train_dataframe)
        profit_normalizer = preprocessing.Normalization( axis=-1)
        profit_normalizer.adapt(profit)
        profit_model = tf.keras.Sequential([ profit_normalizer,layers.Dense(units=10), layers.Dense(units=1)])
        profit_model.summary()
        print(profit_model.predict(profit[:10]))

        profit_model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=10),
            loss='mean_absolute_error')

        self.history = profit_model.fit(
            self.train_dataframe, self.train_target,
            epochs=20,
            # suppress logging
            verbose=1,
            # Calculate validation results on 20% of the training data
            validation_split=0.2)
        print(normalizer.mean.numpy())
        hist = pd.DataFrame(self.history.history)
        hist['epoch'] = self.history.epoch
        hist.tail()

        test_results = {}

        test_results['profit_model'] = profit_model.evaluate(
            self.train_dataframe,
            self.train_target, verbose=0)
        #self.plot_loss()

        x = tf.linspace(0.0, 250, 251)
        y = profit_model.predict(x)

        def plot_result(x, y):
            plt.scatter(self.train_dataframe['Trades'], self.train_target, label='Data')
            plt.plot(x, y, color='k', label='Predictions')
            plt.xlabel('Trades')
            plt.ylabel('Fwd Result')
            plt.legend()

        plot_result(x, y)
        plt.show()

    def plot_loss(self):
        plt.plot(self.history.history['loss'], label='loss')
        plt.plot(self.history.history['val_loss'], label='val_loss')
        plt.ylim([0, 50])
        plt.xlabel('Epoch')
        plt.ylabel('Error [CustFwd]')
        plt.legend()
        plt.grid(True)

    def run(self):
        """Simple 1 Time DNN Model, along with graphics on Loss and Predictions and saving the Model."""
        print("Start Simple Run")
        test_run = self.split_train_test_sequest_bricks()
        self.train_dataframe = self.concatenate_phase(phase_list = self.train_list, Target="CustomForward")[0]
        self.train_target = self.concatenate_phase(phase_list = self.train_list, Target="CustomForward")[1]
        self.norm_train_dataframe = self.normalize_dataframe(self.train_dataframe, 'Median')
        self.norm_train_target = self.normalize_dataframe(self.train_target, 'Median')
        self.validation_dataframe = self.concatenate_phase(phase_list = self.test_list, Target="CustomForward")[0]
        self.validation_targets = self.concatenate_phase(phase_list = self.test_list, Target="CustomForward")[1]
        self.norm_validation_dataframe = self.normalize_dataframe(self.validation_dataframe, 'Median')
        self.norm_validation_targets = self.normalize_dataframe(self.validation_targets, 'Median')
        validation_dataframe = self.concatenate_phase(phase_list = self.test_list, Target="CustomForward")[0]
        validation_targets = self.concatenate_phase(phase_list = self.test_list, Target="CustomForward")[1]
        callback_path = os.path.join(save_dir + 'savedmodel.ckpt')
        self.save_model_path = save_dir + 'saved_model.h5'
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensor_dir + 'logs', histogram_freq=1)
        model_callbacks = tf.keras.callbacks.ModelCheckpoint(filepath=callback_path, save_best_only=True, verbose=2)
        model = self.build_basic_model(input_dimension=len(self.train_dataframe.columns))
        history = model.fit(
            x=self.norm_train_dataframe,
            y=self.train_target,
            batch_size=50,
            epochs=100,
            verbose=2,
            shuffle=False,
            validation_data=(self.norm_validation_dataframe, self.validation_targets),
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
        """Runs the Model (from tuner, single Column, or Simple Basic Model) with data straight from the optimization"""
        self.save_model_path = save_dir + 'saved_model.h5'
        self.sequest_list = [['2015.9.1', '2019.9.1', '2020.9.1'], ['2016.1.1', '2020.1.1', '2021.1.1']]
        self.split_train_test_sequest_bricks()
        #print('train: ', self.train_list)
        #print('test: ', self.test_list)
        #print("Start: ", self.sequest_start)
        sequest_data = self.concatenate_phase(phase_list=self.sequest_list, Target="CustomForward")[0]
        norm_sequest_data = self.normalize_dataframe(sequest_data, 'Median')
        sequest_target = self.concatenate_phase(phase_list=self.sequest_list, Target="CustomForward")[1]
        norm_sequest_target = self.normalize_dataframe(sequest_target, 'Median')
        new_model = tf.keras.models.load_model(self.save_model_path)
        #i = 00
        for i in range(0,10):
            new_model = self.tuner.get_best_models(num_models=20)[i]
            self.predictions = new_model.predict(norm_sequest_data)
            norm_sequest_data.to_csv(save_dir + "sequestered_data.csv")
            norm_sequest_target.to_csv(save_dir + '/sequestered_target.csv')
            dfPredictions = pd.DataFrame(self.predictions)
            dfPredictions.to_csv(save_dir + '/predictions{}.csv'.format(i))
            print("iteration", i)
            print("Three prediction examples: ", self.predictions[:3])
        self.plot_predictions_vs_actual(target= norm_sequest_target)

    def plot_predictions_vs_actual(self, target):
        """Plots Predictions against actual values from the sequestered test"""
        print("plotting predictions")
        a = plt.axes(aspect='equal')
        plt.scatter(target, self.predictions)
        plt.xlabel('Actual Results')
        plt.ylabel('Predicted Results')
        lims = [0, 20]
        plt.xlim(lims)
        plt.ylim(lims)
        _ = plt.plot(lims, lims)
        plt.show()


smth = SelectorRegression(train_start, test_start, sequest_start, train_steps)
#smth.single_input_model()
smth.run()
#smth.run_tuner()
smth.run_sequestered_model()