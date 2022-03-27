import numpy as np
import pandas as pd
import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
from scipy import stats
from tensorflow.keras import regularizers
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing as skpre
import matplotlib.pyplot as plt
from tensorflow.keras import layers, preprocessing
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.callbacks import Callback
from keras_tuner import RandomSearch, BayesianOptimization, Hyperband
from keras_tuner import HyperParameters
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
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

### Fechas de Entrenamiento y Validacion, cortes en uso de clase TimeBricks
train_start = '2007.01.01'
test_start = '2016.01.01'
sequest_start = '2020.12.01'
train_steps = 30

class TrainingCallback(Callback):

    def __init__(self,model_name):
        self.model_name = model_name

    def on_epoch_begin(self, epoch, logs=None):
        """Try to plot the predictions vs actual data here"""
        if epoch%10 == 0 and epoch != 0:
            self.save_model_path = os.path.join(save_dir + "Models/" + '{}.ckpt'.format(self.model_name))
            sequest_data = pd.read_csv("C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/EA-B1v2/GBPJPY/M15/sequestered_data.csv")
            correlated_columns = ['Range', 'Result', 'Profit', 'Expected Payoff', 'Profit Factor',
                                  'Recovery Factor', 'Sharpe Ratio', 'Average Loss', 'Equity DD %', 'Absolute DD']
            sequest_data = sequest_data.drop(columns=correlated_columns)  # drop correlated columns

            sequest_data = sequest_data.to_numpy()
            y_test = pd.read_csv("C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/EA-B1v2/GBPJPY/M15/sequestered_target.csv")
            y_test = y_test.drop(y_test.columns[0], axis=1)
            y_test = y_test.to_numpy()
            model_that_predicts = tf.keras.models.load_model(self.save_model_path)   # , custom_objects={'custom_loss_function':'loss'} paste to load custom loss function
            pred = model_that_predicts.predict(sequest_data)
            plt = SelectorRegression.plot_predictions_vs_actual_simple(self, y_test=y_test,pred=pred)
            save_image = os.path.join(save_dir + "Models/{}/pred_vs_actual_{}_{}.png".format(self.model_name, self.model_name, epoch))
            plt.savefig(save_image, bbox_inches='tight')
            df_predictions_and_error = pd.DataFrame(data=y_test[:,0], dtype='int32')
            df_predictions_and_error['true_values'] = y_test
            df_predictions_and_error['predictions'] = pred
            df_predictions_and_error = df_predictions_and_error.drop(columns=0)
            df_predictions_and_error['ABSError'] = np.abs(df_predictions_and_error['true_values'] - df_predictions_and_error['predictions'])
            df_predictions_and_error['MSE'] = pow((df_predictions_and_error['true_values'] - df_predictions_and_error['predictions']), 2)
            df_predictions_and_error['Custom_Error'] = pow((df_predictions_and_error['true_values'] - df_predictions_and_error['predictions']),4)
            save_predictions_and_error_df = os.path.join(save_dir + "Models/{}/{}_predictions_and_error.csv".format(self.model_name, self.model_name))
            df_predictions_and_error.to_csv(save_predictions_and_error_df)
            plt.cla()
            plt.clf()
            del plt
            print("epoch {}, execute function".format(epoch))

    def on_train_end(self, logs=None):
        """Save model with correct Name"""
        print("On train end works well and now saves model {}".format(self.model_name))
        pass

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
        concatenated_dataframe['Result Difference'] = concatenated_dataframe['Result'] - concatenated_dataframe['Forward Result']
        #print("Before cutting: ", len(concatenated_dataframe))
        concatenated_dataframe.drop(concatenated_dataframe[concatenated_dataframe['Result'] <= 0].index, inplace=True) #drops custom values below 0
        #concatenated_dataframe = concatenated_dataframe.drop(concatenated_dataframe["Lots"], axis=1)
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

    def get_optimizer(self, batch_size):
        """Returns Optimizer with LR decay"""
        STEPS_PER_EPOCH = len(self.X_train) // batch_size
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.005, decay_steps=STEPS_PER_EPOCH * 1000, decay_rate=1, staircase=False)
        return tf.keras.optimizers.Adam(lr_schedule)

    def custom_loss_function(self, y_true, y_pred):
        """Returns Custom Loss function"""
        #squared_difference = tf.square(y_true - y_pred)
        squared_difference = pow(y_true - y_pred, 4)
        return tf.reduce_mean(squared_difference, axis=-1)

    def get_callbacks(self, name="other_stupid_model"):
        """returns callbacks both for EarlyStopping and for TensorBoard feed"""
        self.save_model_path = os.path.join(save_dir+ "Models/" + '{}.ckpt'.format(name))
        return [
            tf.keras.callbacks.ModelCheckpoint(filepath=self.save_model_path, save_best_only=True, epochs= 5, verbose=2),
            tf.keras.callbacks.EarlyStopping(monitor='val_mse', patience=150),
            tf.keras.callbacks.TensorBoard(log_dir=tensor_dir + 'logs/{}'.format(name), histogram_freq=1, write_grads=True)]

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

    def select_predefined_model(self, model_name, input_shape):
        """Select from a pre-defined model"""
        weight_initializer = tf.keras.initializers.LecunNormal()
        # weight_initializer = tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5)
        if model_name == "Small":
            small_model = tf.keras.Sequential([
                # `input_shape` is only required here so that `.summary` works.
                tf.keras.layers.Dense(16, activation='selu', input_shape=(input_shape,)),
                tf.keras.layers.Dense(16, activation='selu', kernel_initializer=weight_initializer),
                tf.keras.layers.Dense(1)])
            return small_model

        if model_name == "Medium":
            medium_model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='selu', input_shape=(input_shape,)),
                tf.keras.layers.Dense(64, activation='selu', kernel_initializer=weight_initializer),
                tf.keras.layers.Dense(64, activation='selu', kernel_initializer=weight_initializer),
                tf.keras.layers.Dense(1)])
            return medium_model

        if model_name == "Large":
            large_model = tf.keras.Sequential([
                tf.keras.layers.Dense(512, activation='selu', input_shape=(input_shape,)),
                tf.keras.layers.Dense(512, activation='selu', kernel_initializer=weight_initializer),
                tf.keras.layers.Dense(512, activation='selu', kernel_initializer=weight_initializer),
                tf.keras.layers.Dense(512, activation='selu', kernel_initializer=weight_initializer),
                tf.keras.layers.Dense(1)])
            return large_model

        if model_name == "Larger":
            Larger_model = tf.keras.Sequential([
                tf.keras.layers.Dense(1024, activation='selu', input_shape=(input_shape,)),
                tf.keras.layers.Dense(1024, activation='selu', kernel_initializer=weight_initializer),
                tf.keras.layers.Dense(1024, activation='selu', kernel_initializer=weight_initializer),
                tf.keras.layers.Dense(1024, activation='selu', kernel_initializer=weight_initializer),
                tf.keras.layers.Dense(1)])
            return Larger_model

        if model_name == "Giant":
            Giant_model = tf.keras.Sequential([
                tf.keras.layers.Dense(2048, activation='selu', input_shape=(input_shape,)),
                tf.keras.layers.Dense(2048, activation='selu', kernel_initializer=weight_initializer),
                tf.keras.layers.Dense(2048, activation='selu', kernel_initializer=weight_initializer),
                tf.keras.layers.Dense(2048, activation='selu', kernel_initializer=weight_initializer),
                tf.keras.layers.Dense(1)])
            return Giant_model

        if model_name == "l2_model_lowest":
            l2_model_lowest = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='selu', kernel_initializer=weight_initializer, kernel_regularizer=tf.keras.regularizers.l2(0.00001), input_shape=(input_shape,)),
                tf.keras.layers.Dense(64, activation='selu', kernel_initializer=weight_initializer, kernel_regularizer=tf.keras.regularizers.l2(0.00001)),
                tf.keras.layers.Dense(64, activation='selu', kernel_initializer=weight_initializer, kernel_regularizer=tf.keras.regularizers.l2(0.00001)),
                tf.keras.layers.Dense(64, activation='selu', kernel_initializer=weight_initializer, kernel_regularizer=tf.keras.regularizers.l2(0.00001)),
                tf.keras.layers.Dense(1)])
            return l2_model_lowest

        if model_name == "l2_model_low":
            l2_model_low = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='selu', kernel_initializer=weight_initializer, kernel_regularizer=tf.keras.regularizers.l2(0.0001), input_shape=(input_shape,)),
                tf.keras.layers.Dense(64, activation='selu', kernel_initializer=weight_initializer, kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
                tf.keras.layers.Dense(64, activation='selu', kernel_initializer=weight_initializer, kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
                tf.keras.layers.Dense(64, activation='selu', kernel_initializer=weight_initializer, kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
                tf.keras.layers.Dense(1)])
            return l2_model_low

        if model_name == "l2_model_mid":
            l2_model_mid = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='selu', kernel_initializer=weight_initializer, kernel_regularizer=tf.keras.regularizers.l2(0.001), input_shape=(input_shape,)),
                tf.keras.layers.Dense(64, activation='selu', kernel_initializer=weight_initializer, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                tf.keras.layers.Dense(64, activation='selu', kernel_initializer=weight_initializer, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                tf.keras.layers.Dense(64, activation='selu', kernel_initializer=weight_initializer, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                tf.keras.layers.Dense(1)])
            return l2_model_mid

        if model_name == "l2_model_high":
            l2_model_high = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='selu', kernel_initializer=weight_initializer, kernel_regularizer=tf.keras.regularizers.l2(0.01), input_shape=(input_shape,)),
                tf.keras.layers.Dense(64, activation='selu', kernel_initializer=weight_initializer, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
                tf.keras.layers.Dense(64, activation='selu', kernel_initializer=weight_initializer, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
                tf.keras.layers.Dense(64, activation='selu', kernel_initializer=weight_initializer, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
                tf.keras.layers.Dense(1)])
            return l2_model_high

        if model_name == "Dropout_model_ten":
            Dropout_model_ten = tf.keras.Sequential([
                tf.keras.layers.Dense(512, activation='selu', input_shape=(input_shape,)),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(512, activation='selu', kernel_initializer=weight_initializer),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(512, activation='selu', kernel_initializer=weight_initializer),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(512, activation='selu', kernel_initializer=weight_initializer),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(1)])
            return Dropout_model_ten

        if model_name == "Dropout_model_twenty":
            Dropout_model_twenty = tf.keras.Sequential([
                tf.keras.layers.Dense(512, activation='selu', input_shape=(input_shape,)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(512, activation='selu', kernel_initializer=weight_initializer),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(512, activation='selu', kernel_initializer=weight_initializer),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(512, activation='selu', kernel_initializer=weight_initializer),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1)])
            return Dropout_model_twenty

        if model_name == "Dropout_model_thirty":
            Dropout_model_thirty = tf.keras.Sequential([
                tf.keras.layers.Dense(512, activation='selu', input_shape=(input_shape,)),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(512, activation='selu', kernel_initializer=weight_initializer),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(512, activation='selu', kernel_initializer=weight_initializer),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(512, activation='selu', kernel_initializer=weight_initializer),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(1)])
            return Dropout_model_thirty

        if model_name == "Dropout_model_forty":
            Dropout_model_forty = tf.keras.Sequential([
                tf.keras.layers.Dense(512, activation='selu', input_shape=(input_shape,)),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.Dense(512, activation='selu', kernel_initializer=weight_initializer),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.Dense(512, activation='selu', kernel_initializer=weight_initializer),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.Dense(512, activation='selu', kernel_initializer=weight_initializer),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.Dense(1)])
            return Dropout_model_forty

        if model_name == "Dropout_model_fifty":
            Dropout_model_fifty = tf.keras.Sequential([
                tf.keras.layers.Dense(512, activation='selu', input_shape=(input_shape,)),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(512, activation='selu', kernel_initializer=weight_initializer),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(512, activation='selu', kernel_initializer=weight_initializer),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(512, activation='selu', kernel_initializer=weight_initializer),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1)])
            return Dropout_model_fifty

        if model_name == "Dropout_model_sixty":
            Dropout_model_sixty = tf.keras.Sequential([
                tf.keras.layers.Dense(512, activation='selu', input_shape=(input_shape,)),
                tf.keras.layers.Dropout(0.6),
                tf.keras.layers.Dense(512, activation='selu', kernel_initializer=weight_initializer),
                tf.keras.layers.Dropout(0.6),
                tf.keras.layers.Dense(512, activation='selu', kernel_initializer=weight_initializer),
                tf.keras.layers.Dropout(0.6),
                tf.keras.layers.Dense(512, activation='selu', kernel_initializer=weight_initializer),
                tf.keras.layers.Dropout(0.6),
                tf.keras.layers.Dense(1)])
            return Dropout_model_sixty

        if model_name == "Combined":
            combined_model = tf.keras.Sequential([
                tf.keras.layers.Dense(256, activation='selu',  input_shape=(input_shape,)), #kernel_regularizer=tf.keras.regularizers.l2(0.001), bias_regularizer=tf.keras.regularizers.l2(0.001), kernel_initializer=weight_initializer,
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(128, activation='selu'), #,kernel_regularizer=tf.keras.regularizers.l2(0.0001), bias_regularizer=tf.keras.regularizers.l2(0.0001), kernel_initializer=weight_initializer
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(164,  activation='selu'), #,kernel_regularizer=tf.keras.regularizers.l2(0.0001), bias_regularizer=tf.keras.regularizers.l2(0.0001), kernel_initializer=weight_initializer
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(32, activation='selu'), #,kernel_regularizer=tf.keras.regularizers.l2(0.0001), bias_regularizer=tf.keras.regularizers.l2(0.0001), kernel_initializer=weight_initializer
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(1)]) # , bias_regularizer=tf.keras.regularizers.l2(0.01)
            return combined_model


    def compile_and_fit(self, model_name, name, optimizer=None, max_epochs=1200):
        """Compiles and fit a Tensorflow Model Object"""  #self.get_callbacks(name)
        if optimizer is None:
            optimizer = self.get_optimizer(batch_size=self.batch_size)
        model = self.select_predefined_model(model_name, input_shape=len(self.train_dataframe.columns))
        model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
        model.summary()
        history = model.fit(self.X_train, self.y_train, epochs=max_epochs, callbacks=[TrainingCallback(model_name=name), self.get_callbacks(name)],
                            validation_data=(self.X_test, self.y_test), shuffle=False, batch_size=self.batch_size)
        self.model_weights = model.get_weights()
        save_weights_csv = os.path.join(save_dir + "Models/{}/{}_weights.csv".format(model_name, model_name))
        np.savetxt(save_weights_csv, self.model_weights, fmt='%s', delimiter=',')
        #print("These are the {} weights: \n".format(model_name), self.model_weights)
        return history


    def build_model(self, hp):
        """Builds the model used for hyperparameter optimization only"""
        model = tf.keras.Sequential()
        hp_activation = ['elu', 'relu']  #, 'relu'
        hp_activation = hp.Choice('Activation', hp_activation)
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
        model.add(layers.Dense(24, input_shape=(len(self.train_dataframe.columns), )))
        self.try_dropout = False
        for i in range(hp_num_layers):
            model.add(
                layers.Dense(
                    units=hp.Int("units_" + str(i), min_value=500, max_value=5000, step=50),
                    activation=hp_activation,
                    kernel_regularizer=regularizer(hp_weight_decay),
                    kernel_initializer= weight_initializer
                )
            )
            if self.try_dropout == True:
                model.add(layers.Dropout(hp_dropout_rate))
        model.add(layers.Dense(1))

        model.compile(optimizer=self.get_optimizer(batch_size=50), loss='mae', metrics=['mae'])
        self.model_tuner_name = "tuner-{}-{}-{}-{}".format(hp_num_layers, hp_weight_decay, str(self.try_dropout), hp_dropout_rate )
        self.get_callbacks(name=self.model_tuner_name)
        print("Model name:", self.model_tuner_name)
        model.summary()
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
        self.X = self.train_dataframe.to_numpy()
        self.y = self.train_target.to_numpy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3)
        self.tuner = BayesianOptimization(
            self.build_model,
            objective="val_mae",
            max_trials=50,
            executions_per_trial=1,
            overwrite=True,
            directory="my_dir",
            project_name="Kratos",
        )

        print("Tuner Summary:", self.tuner.search_space_summary())
        self.tuner.search(self.X_train, self.y_train, epochs=15,validation_data=(self.X_test, self.y_test), shuffle=False, verbose=1, callbacks=[tf.keras.callbacks.TensorBoard(tensor_dir)])
        print("Tuner Results Summary:", self.tuner.results_summary())


    def run_sequestered_model_from_tuner(self):
        """Runs the Model from tuner, with data straight from the optimization"""
        #self.save_model_path = save_dir + 'saved_model.h5'
        self.sequest_list = [['2015.9.1', '2019.9.1', '2020.9.1'], ['2016.1.1', '2020.1.1', '2021.1.1']]
        self.split_train_test_sequest_bricks()
        sequest_data = self.concatenate_phase(phase_list=self.sequest_list, Target='Result Difference')[0]
        sequest_data = sequest_data.drop(columns='Lots')
        sequest_data = sequest_data.drop(sequest_data.columns[0], axis=1)
        #sequest_data.drop(sequest_data[sequest_data["Result"] <= 0].index, inplace=True)
        sequest_data.to_csv(save_dir + "sequestered_data_not_normalized_only_positive.csv", index=False)
        norm_sequest_data = self.normalize_dataframe(sequest_data, 'Median')
        norm_sequest_data.to_csv(save_dir + "sequestered_data.csv", index=False)
        sequest_target = self.concatenate_phase(phase_list=self.sequest_list, Target='Result Difference')[1]
        sequest_target.to_csv(save_dir + "sequestered_target_not_normalized_full.csv", index=False)
        #for i in range(0,30):
        #new_model = self.tuner.get_best_models(num_models= 35)[i]
        #self.predictions = new_model.predict(norm_sequest_data)
        sequest_target.to_csv(save_dir + '/sequestered_target.csv')
        #dfPredictions = pd.DataFrame(self.predictions)
        #dfPredictions.to_csv(save_dir + '/predictions{}_from_tuner.csv'.format(i))
        #print("iteration", i)
        #print("Three prediction examples: ", self.predictions[:3])
        #self.plot_predictions_vs_actual(target=sequest_target, tuner_mode=True, model_number=i)

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
        self.train_dataframe = pd.read_csv("C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/Optimizaciones Listas/Data encadenada/EA-B1v2 on GBPJPY on M15.csv")
        self.train_dataframe = self.train_dataframe.dropna(axis=1, how='all')
        #self.train_dataframe.drop(self.train_dataframe[self.train_dataframe['Result'] <= 0].index,inplace=True)  # drops custom values below 0
        self.train_dataframe['Result Difference'] =self.train_dataframe['Result'] - self.train_dataframe['Forward Result']
        self.train_target = self.train_dataframe.pop("Result Difference")
        self.train_dataframe = self.normalize_dataframe(self.train_dataframe, 'Median')
        self.train_dataframe =self.train_dataframe.drop(columns='Lots')
        columns_list = list(self.train_dataframe)
        forward_columns = [c for c in columns_list if 'Forward' in c]
        self.train_dataframe = self.train_dataframe.drop(columns=forward_columns)  # drop forward columns to prevent look ahead bias
        self.train_dataframe = self.train_dataframe.drop(columns='Back Result')  # drop back result is irrelevant
        self.train_dataframe = self.train_dataframe.drop(self.train_dataframe.columns[0], axis=1)
        correlated_columns = ['index', 'Result', 'Profit', 'Expected Payoff', 'Profit Factor', 'Recovery Factor', 'Sharpe Ratio', 'Average Loss', 'Equity DD %', 'Absolute DD']
        self.train_dataframe = self.train_dataframe.drop(columns=correlated_columns)  # drop correlated columns
        self.train_dataframe.to_csv(save_dir + "LookHERE_train_data.csv", index=False)
        self.train_target.to_csv(save_dir + "LookHERE_target_data.csv", index=False)
        self.X = self.train_dataframe.to_numpy()
        self.y = self.train_target.to_numpy()
        print("X: \n", self.X,"y: \n", self.y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3)
        self.batch_size = 200
        self.save_model_path = save_dir + 'saved_model.h5'
        self.run_sequestered_model_from_tuner()
        #model = self.compile_and_fit(model_name="Large", name="Large")
        size_histories = {}
        #size_histories['Small'] = self.compile_and_fit(model_name="Small", name='Small')
        #size_histories['Medium'] = self.compile_and_fit(model_name="Medium", name='Medium')
        #size_histories['Large'] = self.compile_and_fit(model_name="Large", name='Large')
        #size_histories['Larger'] = self.compile_and_fit(model_name="Larger", name='Larger')
        #size_histories['Giant'] = self.compile_and_fit(model_name="Giant", name='Giant')
        #size_histories['l2_model_lowest'] = self.compile_and_fit(model_name="l2_model_lowest", name='l2_model_lowest')
        #size_histories['l2_model_low'] = self.compile_and_fit(model_name="l2_model_low", name='l2_model_low')
        #size_histories['l2_model_mid'] = self.compile_and_fit(model_name="l2_model_mid", name='l2_model_mid')
        #size_histories['l2_model_high'] = self.compile_and_fit(model_name="l2_model_high", name='l2_model_high')
        #size_histories['Dropout_model_ten'] = self.compile_and_fit(model_name="Dropout_model_ten", name='Dropout_model_ten')
        #size_histories['Dropout_model_twenty'] = self.compile_and_fit(model_name="Dropout_model_twenty", name='Dropout_model_twenty')
        #size_histories['Dropout_model_thirty'] = self.compile_and_fit(model_name="Dropout_model_thirty", name='Dropout_model_thirty')
        #size_histories['Dropout_model_forty'] = self.compile_and_fit(model_name="Dropout_model_forty", name='Dropout_model_forty')
        size_histories['Combined'] = self.compile_and_fit(model_name="Combined",name='Combined')
        #size_histories['Dropout_model_fifty'] = self.compile_and_fit(model_name="Dropout_model_fifty", name='Dropout_model_fifty')
        #size_histories['Dropout_model_sixty'] = self.compile_and_fit(model_name="Dropout_model_sixty", name='Dropout_model_sixty')
        ##size_histories['Combined'] = self.compile_and_fit(model_name="Combined", name='Combined')
        #plotter = tfdocs.plots.HistoryPlotter(metric='mse', smoothing_std=10)
        #plotter.plot(size_histories)
        #plt.ylim([5, 60])
        #plt.show()
        #save_image = save_dir + 'All_models_all_data.png'
        ##plotter.savefig(save_image, bbox_inches='tight')
        ##size_histories['Medium'].save(self.save_model_path)
        #plt.figure(1)
        #plt.plot(size_histories['Medium'].history['loss'])
        #plt.plot(size_histories['Medium'].history['val_loss'])
        #plt.title('Perdidas de Modelo')
        #plt.ylabel('Perdidas')
        #plt.xlabel('Epochs')
        #plt.legend(['Train', 'Test'], loc='upper left')
        #plt.show()
        #model_that_predicts = tf.keras.models.load_model(self.save_model_path)
        #pred = model_that_predicts.predict(self.y_test)
        #print(self.y_test, pred)
        #self.plot_predictions_vs_actual_simple(self.y_test, pred)
        #df = pd.DataFrame()
        #df['Y val'] = y_test
        #df['pred'] = pred
        #df['Error'] = abs(df['Y val'] - df['pred'])

    def plot_predictions_vs_actual_simple(self,y_test, pred):
        """Plots Predictions against actual values from the sequestered test"""
        print("plotting predictions vs observations")
        a = plt.axes(aspect='equal')
        plt.scatter(y_test, y_test, color='blue')
        plt.scatter(y_test, pred, color='red')
        plt.xlabel('Actual Results')
        plt.ylabel('Predicted Results')
        lims = [-200, 200]
        plt.xlim(lims)
        plt.ylim(lims)
        plot_object = plt.plot(lims, lims)
        #plt.show()
        return plt
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
        norm_sequest_data.to_csv(save_dir + "sequestered_data.csv", index=False)
        sequest_target.to_csv(save_dir + '/sequestered_target.csv', index=False)
        dfPredictions = pd.DataFrame(self.predictions, columns = ["Predictions"])
        dfPredictions.to_csv(save_dir + '/predictions_from_single_run.csv', index=False)
        comparison_dataframe = dfPredictions.join(sequest_target, how="right")
        comparison_dataframe.to_csv(save_dir + 'comparison_single_run.csv', index=False)
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
        lims = [-200, 200]
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
