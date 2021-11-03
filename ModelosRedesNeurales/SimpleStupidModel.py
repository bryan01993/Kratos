import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
batch_size = 50
epochs = 35
load_dir = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/EA-B1v2/GBPJPY/M15/'
tensor_dir = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/EA-B1v2/GBPJPY/M15/Tensorlogs/'
save_dir = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/EA-B1v2/GBPJPY/M15/'
np.set_printoptions(suppress=True)  # set numpy prints value to not scientific

def simple_stupid_model():

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensor_dir + 'logs/stupid_model')
    train_dataframe = pd.read_csv(load_dir + "train_dataframe_normalized.csv")
    train_target = pd.read_csv(load_dir + "train_target.csv")
    validation_dataframe = pd.read_csv(load_dir + "validation_dataframe_normalized.csv")
    validation_target = pd.read_csv(load_dir + "validation_target.csv")
    train_dataframe = train_dataframe.drop(['Unnamed: 0'], axis=1)
    train_target = train_target.drop(['Unnamed: 0'], axis=1)
    validation_dataframe = validation_dataframe.drop(['Unnamed: 0'], axis=1)
    validation_target = validation_target.drop(['Unnamed: 0'], axis=1)
    np_train_dataframe = train_dataframe.to_numpy()
    np_train_target = train_target.to_numpy()
    np_validation_dataframe = validation_dataframe.to_numpy()
    np_validation_target = validation_target.to_numpy()
    print(np_train_dataframe.shape)
    print(np_train_target.shape)
    weight_initializer = tf.keras.initializers.GlorotNormal()
    Entradas = tf.keras.Input(shape=(24, ), name="Entry")
    x = tf.keras.layers.Dense(1200, activation='elu',kernel_initializer=weight_initializer, kernel_regularizer=tf.keras.regularizers.l1(1e-8),  name="Deep_1")(Entradas)
    #x = tf.keras.layers.Dense(64, activation="elu")(x)
    x = tf.keras.layers.Dense(1, activation = "linear", name="Output")(x)
    modelo = tf.keras.models.Model(inputs=Entradas, outputs=x)
    Adam = tf.keras.optimizers.Adam(learning_rate=2, beta_1=0.9, beta_2=0.9)
    modelo.compile(loss=tf.keras.losses.mae, optimizer=Adam, metrics=["mae"])
    print("Model Weights before: \n", modelo.weights)
    history = modelo.fit(np_train_dataframe, np_train_target, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(np_validation_dataframe, np_validation_target), callbacks=[tensorboard])
    puntuacion = modelo.evaluate(validation_dataframe, validation_target, verbose=1)
    print("Model Weights after: \n", modelo.weights)
    print(puntuacion)
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Perdidas de Modelo')
    plt.ylabel('Perdidas')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def other_stupid_model():

    other_stupid_model_dataframe = pd.read_csv("C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/Optimizaciones Listas/Data encadenada/EA-B1v2 on GBPJPY on M15.csv")
    """Concatenates the phase and pops the target value"""
    other_stupid_model_dataframe = other_stupid_model_dataframe.dropna(axis=1, how='all')  # drop nan values
    other_stupid_model_dataframe.drop(other_stupid_model_dataframe[other_stupid_model_dataframe['Result'] <= 0].index, inplace=True)  # drops custom values below 0
    try:
        other_stupid_model_target = other_stupid_model_dataframe.pop("Forward Result")
    except:
        print("Forward Result")

    def normalize_dataframe(raw_dataframe, norm_type):
        """Applies a Normalization to the dataframe to pass to the model"""
        if norm_type == 'Median':
            processed_dataframe = (raw_dataframe-raw_dataframe.mean())/raw_dataframe.std()
        elif norm_type == 'MaxMin':
            processed_dataframe = (raw_dataframe - raw_dataframe.min()) / (raw_dataframe.max() - raw_dataframe.min())
        else:
            print("Select a normalization type between Median and MaxMin")
        return processed_dataframe

    def get_callbacks(name="other_stupid_model"):
        return [
            tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=150),
            tf.keras.callbacks.TensorBoard(log_dir=tensor_dir + 'logs/{}'.format(name), histogram_freq=1)
        ]

    def get_optimizer():
        STEPS_PER_EPOCH = len(X) // batch_size
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.001, decay_steps=STEPS_PER_EPOCH * 1000,decay_rate=1, staircase=False)
        return tf.keras.optimizers.Adam(lr_schedule)

    other_stupid_model_dataframe = normalize_dataframe(other_stupid_model_dataframe, "MaxMin")
    other_stupid_model_dataframe = other_stupid_model_dataframe.dropna(axis=1, how='all')
    other_stupid_model_dataframe.to_csv(save_dir+'/MaxMin.csv')
    columns_list = list(other_stupid_model_dataframe)
    forward_columns = [c for c in columns_list if 'Forward' in c]
    other_stupid_model_dataframe = other_stupid_model_dataframe.drop(columns=forward_columns)  # drop forward columns to prevent look ahead bias
    other_stupid_model_dataframe = other_stupid_model_dataframe.drop(columns='Back Result')  # drop back result is irrelevant
    X = other_stupid_model_dataframe.to_numpy()
    y = other_stupid_model_target.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    def compile_and_fit(model, name, optimizer=None, max_epochs= 1500):
        if optimizer is None:
            optimizer = get_optimizer()
        model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])
        model.summary()
        history = model.fit(X_train, y_train, epochs=max_epochs, callbacks=[get_callbacks(name)], validation_data=(X_test, y_test), shuffle=True, batch_size=batch_size)
        return history

    tiny_model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='elu', input_shape=(23,)),
        tf.keras.layers.Dense(1)
    ])
    small_model = tf.keras.Sequential([
        # `input_shape` is only required here so that `.summary` works.
        tf.keras.layers.Dense(16, activation='elu', input_shape=(23,)),
        tf.keras.layers.Dense(16, activation='elu'),
        tf.keras.layers.Dense(1)
    ])
    medium_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='elu', input_shape=(23,)),
        tf.keras.layers.Dense(64, activation='elu'),
        tf.keras.layers.Dense(64, activation='elu'),
        tf.keras.layers.Dense(1)
    ])
    large_model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='elu', input_shape=(23,)),
        tf.keras.layers.Dense(512, activation='elu'),
        tf.keras.layers.Dense(512, activation='elu'),
        tf.keras.layers.Dense(512, activation='elu'),
        tf.keras.layers.Dense(1)
    ])

    l2_model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='elu',kernel_regularizer=tf.keras.regularizers.l2(0.001),input_shape=(23,)),
        tf.keras.layers.Dense(512, activation='elu',kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dense(512, activation='elu',kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dense(512, activation='elu',kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dense(1)
    ])

    dropout_model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='elu', input_shape=(23,)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='elu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='elu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='elu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1)
    ])

    combined_model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.0001),activation='elu', input_shape=(23,)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.0001),activation='elu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.0001),activation='elu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.0001), activation='elu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1)
    ])
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(500, input_shape=(24,), activation= "relu"))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    model.summary()



    size_histories = {}
    print("starting Tiny")
    size_histories['Tiny'] = compile_and_fit(tiny_model, 'sizes/Tiny')
    size_histories['Small'] = compile_and_fit(small_model, 'sizes/Small')
    size_histories['Medium'] = compile_and_fit(medium_model, "sizes/Medium")
    print("starting Large")
    size_histories['large'] = compile_and_fit(large_model, "sizes/large")
    size_histories['l2'] = compile_and_fit(l2_model, "regularizers/l2")
    print("starting dropout")
    size_histories['dropout'] = compile_and_fit(dropout_model, "regularizers/dropout")
    size_histories['combined'] = compile_and_fit(combined_model, "regularizers/combined")
    plotter = tfdocs.plots.HistoryPlotter(metric = 'mae', smoothing_std=10)
    plotter.plot(size_histories)
    plt.ylim([15, 25])
    plt.show()
    pred_train = model.predict(X_train)
    #print(y_train, pred_train)

    pred = model.predict(X_test)
    #print(y_test, pred)
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
    df.to_csv( 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/EA-B1v2/GBPJPY/M15/other_stupid.csv')
other_stupid_model()