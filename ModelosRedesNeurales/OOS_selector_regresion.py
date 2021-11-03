import numpy as np
import pandas as pd
import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
from tensorflow.keras import regularizers
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing as skpre
import matplotlib.pyplot as plt
from tensorflow.keras import layers, preprocessing
from tensorflow.keras.layers.experimental import preprocessing
from keras_tuner import RandomSearch, BayesianOptimization, Hyperband
from keras_tuner import HyperParameters
import shutil
###Magic Block
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
np.set_printoptions(suppress=True)  # set numpy prints value to not scientific
from services.create_timebricks import CreateTimebricks
from services.helpers import movecol

### Rutas a los directorios de data, de guardado de resultados y de tensores para TensorBoard
LOG_DIR = f"{int(time.time())}"
base_dir = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/EA-B1v2/GBPJPY/M15/WF_Report'
save_dir = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/EA-B1v2/GBPJPY/M15/'
tensor_dir = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/EA-B1v2/GBPJPY/M15/Tensorlogs/'

# tensorboard --logdir /home/miguel/Proyectos/kratos/Data/GBPJPY/M15/Tensorlogslogs

### Fechas de Entrenamiento y Validacion, cortes en uso de clase TimeBricks
train_start = '2007.01.01'
test_start = '2016.01.01'
sequest_start = '2020.12.01'
train_steps = 30
#### TO DO

# Graph Results and callbacks
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
        #concatenated_dataframe.to_csv(save_dir+'/only_positive_dataframe.csv')
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
            processed_dataframe = (raw_dataframe-raw_dataframe.mean())/raw_dataframe.std()
        elif norm_type == 'MaxMin':
            processed_dataframe = (raw_dataframe - raw_dataframe.min()) / (raw_dataframe.max() - raw_dataframe.min())
        else:
            print("Select a normalization type between Median and MaxMin")
        return processed_dataframe

    def get_optimizer(self, dataframe_cut, batch_size):
        """Returns Optimizer with LR decay"""
        STEPS_PER_EPOCH = len(dataframe_cut) // batch_size
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.001, decay_steps=STEPS_PER_EPOCH * 1000,decay_rate=1, staircase=False)
        return tf.keras.optimizers.Adam(lr_schedule)

    def get_callbacks(self, name="other_stupid_model"):
        """returns callbacks both for EarlyStopping and for TensorBoard feed"""
        callback_path = os.path.join(save_dir + 'savedmodel.ckpt')
        tf.keras.callbacks.ModelCheckpoint(filepath=callback_path, save_best_only=True, verbose=2)
        return [
            tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=150),
            tf.keras.callbacks.TensorBoard(log_dir=tensor_dir + 'logs/{}'.format(name), histogram_freq=1)]

    def delete_previous_logs(self):
        """Clears Logs from previous runs"""
        for filename in os.listdir(tensor_dir):
            file_path = os.path.join(tensor_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as exception:
                print('Failed to delete %s. Reason: %s' % (file_path, exception))
            print("previous files removed!")
    def build_basic_model(self, input_dimension):
        """Here the Model is created used for a single run"""
        activation = 'relu'
        dropout_rate = 0.5
        wd = 1e-5
        weight_initializer = tf.keras.initializers.GlorotNormal()
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_dimension, name="Input_Layer"),
            tf.keras.layers.Dense(5000,kernel_regularizer = tf.keras.regularizers.l2(0.01), kernel_initializer=weight_initializer, activation=activation, name="A_Layer"),
            tf.keras.layers.Dense(3000,kernel_regularizer = tf.keras.regularizers.l2(0.01), kernel_initializer=weight_initializer, activation=activation,name="B_Layer"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(2000,kernel_regularizer = tf.keras.regularizers.l2(0.01),kernel_initializer=weight_initializer, activation=activation, name="C_Layer"),
            tf.keras.layers.Dense(1000,kernel_regularizer = tf.keras.regularizers.l2(0.01),kernel_initializer=weight_initializer, activation=activation, name="D_Layer"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(500,kernel_regularizer = tf.keras.regularizers.l2(0.01),kernel_initializer=weight_initializer, activation=activation, name="E_Layer"),
            tf.keras.layers.Dense(200,kernel_regularizer = tf.keras.regularizers.l2(0.01),kernel_initializer=weight_initializer, activation=activation, name="F_Layer"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, name='output_layer')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mae', metrics='mae')
        return model

    def build_model(self, hp):
        """Builds the model used for hyperparameter optimization only"""
        model = tf.keras.Sequential()
        hp_activation = ['elu', 'relu']  #, 'relu'
        hp_activation = hp.Choice('Activation', hp_activation)
        #hp_learning_rate = [0.001, 0.0001, 0.01, 0.1]            #  0.001, 0.0001, 0.01, 0.1, 0.15, 0.2
        #hp_learning_rate = hp.Choice('learningRate', hp_learning_rate)
        hp_num_layers = hp.Int('numLayers', 1, 5)
        hp_weight_decay = [0.000000001,0.00000001,0.0000001, 0.000001, 0.00001, 0.0001] #0.0001, 0.001, 0.01
        hp_weight_decay = hp.Choice('weightDecay', hp_weight_decay)
        hp_type_regularization = hp.Choice('regularizer', ['l2'])  #, 'l2'
        hp_dropout_rate = [0.5, 0.3, 0.2, 0.1]
        hp_dropout_rate = hp.Choice('DropoutRate', hp_dropout_rate)
        regularizer = getattr(regularizers, hp_type_regularization)
        weight_initializer = tf.keras.initializers.GlorotNormal()
        data = np.array(self.train_dataframe)
        data_normalizer = preprocessing.Normalization( axis=-1)
        data_normalizer.adapt(data)
        self.tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensor_dir + 'logs/{}-{}-{}-{}'.format(hp_activation ,hp_learning_rate ,hp_num_layers, time.time()), histogram_freq=1)
        model.add(layers.Dense(24, input_shape=(len(self.train_dataframe.columns), )))
        for i in range(hp_num_layers):
            model.add(
                layers.Dense(
                    units=hp.Int("units_" + str(i), min_value=500, max_value=5000, step=50),
                    activation=hp_activation,
                    kernel_regularizer=regularizer(hp_weight_decay),
                    kernel_initializer= weight_initializer
                )
            )
            if i %2 == 0:
                model.add(layers.Dropout(hp_dropout_rate))
        model.add(layers.Dense(1))
        optimizer=tf.keras.optimizers.Adam(learning_rate= hp_learning_rate)                          # learning_rate= hp_learning_rate
        model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])
        #model.summary()
        return model

    def run_tuner(self):
        """Starts the hyperparameter optimization process, along with saving results"""
        print("Start Tuner Run")
        self.delete_previous_logs()
        hp = HyperParameters()
        test_run = self.split_train_test_sequest_bricks()
        self.train_dataframe = pd.read_csv("C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/Optimizaciones Listas/Data encadenada/EA-B1v2 on GBPJPY on M15.csv")
        self.train_dataframe = self.train_dataframe.dropna(axis=1, how='all')
        self.train_dataframe.drop(self.train_dataframe[self.train_dataframe['Result'] <= 0].index,inplace=True)  # drops custom values below 0
        self.train_target = self.train_dataframe.pop("Forward Result")
        self.train_dataframe = self.normalize_dataframe(self.train_dataframe, 'Median')
        columns_list = list(self.train_dataframe)
        forward_columns = [c for c in columns_list if 'Forward' in c]
        self.train_dataframe = self.train_dataframe.drop(columns=forward_columns)  # drop forward columns to prevent look ahead bias
        self.train_dataframe = self.train_dataframe.drop(columns='Back Result')  # drop back result is irrelevant
        #self.train_dataframe = self.concatenate_phase(phase_list = self.train_list, Target="CustomForward")[0]
        #self.train_target = self.concatenate_phase(phase_list = self.train_list, Target="CustomForward")[1]
        #self.norm_train_dataframe = self.normalize_dataframe(self.train_dataframe, 'Median')
        #self.norm_train_dataframe.to_csv(save_dir + "train_dataframe_normalized.csv")
        #self.train_target.to_csv(save_dir + "train_target.csv")
        #self.norm_train_target = self.normalize_dataframe(self.train_target, 'Median')
        #self.validation_dataframe = self.concatenate_phase(phase_list = self.test_list, Target="CustomForward")[0]
        #self.validation_targets = self.concatenate_phase(phase_list = self.test_list, Target="CustomForward")[1]
        #self.norm_validation_dataframe = self.normalize_dataframe(self.validation_dataframe, 'Median')
        #self.norm_validation_dataframe.to_csv(save_dir + "validation_dataframe_normalized.csv")
        #self.validation_targets.to_csv(save_dir + "validation_target.csv")
        #print(self.train_target.shape)
        #self.norm_validation_targets = self.normalize_dataframe(self.validation_targets, 'Median')
        #validation_dataframe = self.concatenate_phase(phase_list = self.test_list, Target="CustomForward")[0]
        #validation_targets = self.concatenate_phase(phase_list = self.test_list, Target="CustomForward")[1]
        #print("Columns of norm train dataframe: \n", self.norm_train_dataframe.columns)
        #print("Columns of norm train target: \n", self.train_target.columns)
        X = self.train_dataframe.to_numpy()
        y = self.train_target.to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        self.input_dimension=len(self.train_dataframe.columns)
        self.tuner = BayesianOptimization(
            self.build_model,
            objective="val_mae",
            max_trials=100,
            executions_per_trial=1,
            overwrite=True,
            directory="my_dir",
            project_name="Kratos",
        )

        print("Tuner Summary:", self.tuner.search_space_summary())
        self.tuner.search(X_train, y_train, epochs=50,validation_data=(X_test, y_test), shuffle=False, verbose=1 ,callbacks = [self.tensorboard])
        print("Tuner Results Summary:", self.tuner.results_summary())
        #self.run_sequestered_model_from_tuner()

    def run_sequestered_model_from_tuner(self):
        """Runs the Model from tuner, with data straight from the optimization"""
        #self.save_model_path = save_dir + 'saved_model.h5'
        self.sequest_list = [['2015.9.1', '2019.9.1', '2020.9.1'], ['2016.1.1', '2020.1.1', '2021.1.1']]
        self.split_train_test_sequest_bricks()
        sequest_data = self.concatenate_phase(phase_list=self.sequest_list, Target="CustomForward")[0]
        norm_sequest_data = self.normalize_dataframe(sequest_data, 'Median')
        sequest_target = self.concatenate_phase(phase_list=self.sequest_list, Target="CustomForward")[1]
        for i in range(0,30):
            new_model = self.tuner.get_best_models(num_models= 35)[i]
            self.predictions = new_model.predict(norm_sequest_data)
            norm_sequest_data.to_csv(save_dir + "sequestered_data.csv")
            sequest_target.to_csv(save_dir + '/sequestered_target.csv')
            dfPredictions = pd.DataFrame(self.predictions)
            dfPredictions.to_csv(save_dir + '/predictions{}_from_tuner.csv'.format(i))
            print("iteration", i)
            print("Three prediction examples: ", self.predictions[:3])
            self.plot_predictions_vs_actual(target=sequest_target, tuner_mode=True, model_number=i)

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
        self.delete_previous_logs()
        test_run = self.split_train_test_sequest_bricks()
        self.train_dataframe = pd.read_csv("C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/Optimizaciones Listas/Data encadenada/EA-B1v2 on GBPJPY on M15.csv")
        self.train_dataframe = self.train_dataframe.dropna(axis=1, how='all')
        self.train_dataframe.drop(self.train_dataframe[self.train_dataframe['Result'] <= 0].index,inplace=True)  # drops custom values below 0
        self.train_target = self.train_dataframe.pop("Forward Result")
        self.train_dataframe = self.normalize_dataframe(self.train_dataframe, 'Median')
        columns_list = list(self.train_dataframe)
        forward_columns = [c for c in columns_list if 'Forward' in c]
        self.train_dataframe = self.train_dataframe.drop(columns=forward_columns)  # drop forward columns to prevent look ahead bias
        self.train_dataframe = self.train_dataframe.drop(columns='Back Result')  # drop back result is irrelevant
        #self.train_dataframe = self.concatenate_phase(phase_list = self.train_list, Target="CustomForward")[0]
        #self.train_target = self.concatenate_phase(phase_list = self.train_list, Target="CustomForward")[1]
        #self.norm_train_dataframe = self.normalize_dataframe(self.train_dataframe, 'Median')
        #self.norm_train_dataframe.to_csv(save_dir + "train_dataframe_normalized.csv")
        #self.train_target.to_csv(save_dir + "train_target.csv")
        #self.norm_train_target = self.normalize_dataframe(self.train_target, 'Median')
        #self.validation_dataframe = self.concatenate_phase(phase_list = self.test_list, Target="CustomForward")[0]
        #self.validation_targets = self.concatenate_phase(phase_list = self.test_list, Target="CustomForward")[1]
        #self.norm_validation_dataframe = self.normalize_dataframe(self.validation_dataframe, 'Median')
        #self.norm_validation_dataframe.to_csv(save_dir + "validation_dataframe_normalized.csv")
        #self.validation_targets.to_csv(save_dir + "validation_target.csv")
        #print(self.train_target.shape)
        #self.norm_validation_targets = self.normalize_dataframe(self.validation_targets, 'Median')
        #validation_dataframe = self.concatenate_phase(phase_list = self.test_list, Target="CustomForward")[0]
        #validation_targets = self.concatenate_phase(phase_list = self.test_list, Target="CustomForward")[1]
        #print("Columns of norm train dataframe: \n", self.norm_train_dataframe.columns)
        #print("Columns of norm train target: \n", self.train_target.columns)
        X = self.train_dataframe.to_numpy()
        y = self.train_target.to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        self.save_model_path = save_dir + 'saved_model.h5'
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensor_dir + 'logs/logs-{}'.format(time.time()), histogram_freq=1)
        model = self.build_basic_model(input_dimension=len(self.train_dataframe.columns))
        print("Model Summary: /n", model.summary())
        print("Model Weights before: \n", model.weights)
        history = model.fit(
            X_train,
            y_train,
            epochs=500,
            verbose=2,
            shuffle=False,
            validation_data=(X_test, y_test),
            callbacks=[tensorboard, model_callbacks]
        )
        model.save(self.save_model_path)
        print("Model Weights after: \n", model.weights)
        print("Evaluate on test data")
        #results = model.evaluate(self.validation_dataframe, self.validation_targets, batch_size=128)
        #print("test loss, test acc:", results)
        plt.figure(1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Perdidas de Modelo')
        plt.ylabel('Perdidas')
        plt.xlabel('Epochs')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        pred = model.predict(X_test)
        # print(y_test, pred)
        df = pd.DataFrame()
        df['Y val'] = y_test
        df['pred'] = pred
        df['Error'] = abs(df['Y val'] - df['pred'])

        def plot_predictions_vs_actual(y_test):
            """Plots Predictions against actual values from the sequestered test"""
            print("plotting predictions vs observations")
            a = plt.axes(aspect='equal')
            plt.scatter(y_test, y_test, color='blue')
            plt.scatter(y_test, pred, color='red')
            plt.xlabel('Actual Results')
            plt.ylabel('Predicted Results')
            lims = [-70, 70]
            plt.xlim(lims)
            plt.ylim(lims)
            plot_object = plt.plot(lims, lims)
            plt.show()

        plot_predictions_vs_actual(y_test)
        #self.run_sequestered_model()

    def run_sequestered_model(self):
        """Runs the Model for Simple Basic Model with data straight from the optimization"""
        self.save_model_path = save_dir + 'saved_model.h5'
        self.sequest_list = [['2015.9.1', '2019.9.1', '2020.9.1'], ['2016.1.1', '2020.1.1', '2021.1.1']]
        self.split_train_test_sequest_bricks()
        sequest_data = self.concatenate_phase(phase_list=self.sequest_list, Target="CustomForward")[0]
        norm_sequest_data = self.normalize_dataframe(sequest_data, 'Median')
        sequest_target = self.concatenate_phase(phase_list=self.sequest_list, Target="CustomForward")[1]
        new_model = tf.keras.models.load_model(self.save_model_path)
        self.predictions = new_model.predict(norm_sequest_data)
        norm_sequest_data.to_csv(save_dir + "sequestered_data.csv")
        sequest_target.to_csv(save_dir + '/sequestered_target.csv')
        dfPredictions = pd.DataFrame(self.predictions, columns = ["Predictions"])
        dfPredictions.to_csv(save_dir + '/predictions_from_single_run.csv')
        comparison_dataframe = dfPredictions.join(sequest_target, how="right")
        comparison_dataframe.to_csv(save_dir + 'comparison_single_run.csv')
        print(comparison_dataframe.describe())
        print("Some prediction examples: \n", self.predictions[:10])
        self.plot_predictions_vs_actual(target= sequest_target)

    def plot_predictions_vs_actual(self, target, tuner_mode=False, model_number=0):
        """Plots Predictions against actual values from the sequestered test"""
        print("plotting predictions vs observations")
        a = plt.axes(aspect='equal')
        plt.scatter(target, target, color='blue')
        plt.scatter(target, self.predictions, color='red')
        plt.xlabel('Actual Results')
        plt.ylabel('Predicted Results')
        lims = [-70, 70]
        plt.xlim(lims)
        plt.ylim(lims)
        plot_object = plt.plot(lims, lims)
        if tuner_mode == True:
            save_image = save_dir + 'predictions_vs_actual_model_{}.png'.format(model_number)
            plt.savefig(save_image, bbox_inches='tight')
            plt.clf()
            #plt.show()
            #plt.close()
        if tuner_mode == False:
            plt.show()


smth = SelectorRegression(train_start, test_start, sequest_start, train_steps)
# smth.single_input_model()
smth.run()
#smth.run_tuner()
