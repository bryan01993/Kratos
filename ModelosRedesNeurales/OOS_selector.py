import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
print('Tensorflow Version:',tf.__version__)
# Datos de Prueba
base_dir = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/EA-B1v1Train/GBPUSD/H4/WF_Report'
save_dir = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/EA-B1v1Train/GBPUSD/H4/'


def create_steps_list(directory=base_dir):
    steps_list = []
    # Produces Steps List to use as index in benchmark
    for file in os.listdir(directory):
        if "Complete-Filtered" in file:
            file_name = directory + '/' + file
            file_start_date = file_name[-37:-35]
            file_end_date = file_name[-28:-26]
            file_to_list = file_start_date + ' to ' + file_end_date
            steps_list.append(file_to_list)
    return steps_list


def create_benchmark_list(directory=base_dir, steps_list=create_steps_list()):
    dataframe_list = []

    # Create the dataframes to concatenate and join the benchmark
    for file in os.listdir(directory):
        if "Complete-Filtered" in file:
            file_name = directory + '/' + file
            file_dataframe = pd.read_csv(file_name)
            dataframe_list.append(file_dataframe)

    benchmark_list = pd.concat(dataframe_list, keys=steps_list)
    benchmark_list.to_csv(save_dir + 'Benchmark.csv')
    return benchmark_list


def prepare_training_file(stepstart,stepend):
    for file in os.listdir(base_dir):
        file_start_date = file[-19:-17]
        file_end_date = file[-10:-8]
        if '.xml' not in file and 'Complete' not in file and 'forward' not in file:
            if stepstart == file_start_date and stepend == file_end_date:
                train_dataframe = pd.read_csv(base_dir + '/' + file)
                train_dataframe = train_dataframe.sort_values('Result',ascending=False)
                train_dataframe  = (train_dataframe-train_dataframe.min())/(train_dataframe.max()-train_dataframe.min())
                train_dataframe = train_dataframe.drop(['Win Ratio','Lots'], axis= 1)
                return train_dataframe

def prepare_answers_file(stepstart,stepend):
    for file in os.listdir(base_dir):
        file_start_date = file[-27:-25]
        file_end_date = file[-18:-16]
        file_dates_benchmark_search = stepstart + ' to ' + stepend
        if '.xml' not in file and 'Complete' not in file and 'forward' in file:
            if stepstart == file_start_date and stepend == file_end_date:
                answers_dataframe = pd.read_csv(base_dir + '/' + file)
                answers_dataframe = answers_dataframe.sort_values('Back Result',ascending=False)
                answers_dataframe = (answers_dataframe-answers_dataframe.min())/(answers_dataframe.max()-answers_dataframe.min())
                answers_dataframe['Target'] = answers_dataframe['Forward Result'].map(lambda x: x > Benchmark.loc[file_dates_benchmark_search]['Forward Result'])
                answers_dataframe['Target'] = answers_dataframe['Target'].astype(int)
                return answers_dataframe

def prepare_target_file(stepstart,stepend):
    for file in os.listdir(base_dir):
        file_start_date = file[-27:-25]
        file_end_date = file[-18:-16]
        file_dates_benchmark_search = stepstart + ' to ' + stepend
        if '.xml' not in file and 'Complete' not in file and 'forward' in file:
            if stepstart == file_start_date and stepend == file_end_date:
                target_dataframe = pd.read_csv(base_dir + '/' + file)
                target_dataframe = target_dataframe.sort_values('Back Result',ascending=False)
                target_dataframe['Target'] = target_dataframe['Forward Result'].map(lambda x: x > Benchmark.loc[file_dates_benchmark_search]['Forward Result'])
                target_dataframe['Target'] = target_dataframe['Target'].astype(int)
                return target_dataframe

def get_compiled_model(inputtrain,inputtarget,firstlayer=60,secondlayer=10,thirdlayer=1):
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(firstlayer,input_shape=(inputtrain.shape[1], ),name='First_Layer'),
    tf.keras.layers.Dense(secondlayer, activation='sigmoid',name='Second_Layer'),
    tf.keras.layers.Dense(secondlayer, activation='sigmoid', name='Third_Layer'),
    tf.keras.layers.Dense(secondlayer, activation='sigmoid', name='Fourth_Layer'),
    tf.keras.layers.Dense(secondlayer, activation='sigmoid', name='Fifth_Layer'),
    tf.keras.layers.Dense(secondlayer, activation='sigmoid', name='Sixth_Layer'),
    tf.keras.layers.Dense(secondlayer, activation='sigmoid', name='Seventh_Layer'),
    tf.keras.layers.Dense(secondlayer, activation='sigmoid', name='Eighth_Layer'),
    tf.keras.layers.Dense(secondlayer, activation='sigmoid', name='Nineth_Layer'),
    tf.keras.layers.Dense(thirdlayer, activation = 'sigmoid',name='Output_Layer')
  ])
  model.compile(optimizer='sgd',
                loss='binary_crossentropy',
                metrics=['accuracy'])
  return model

def training_steps():
    Fitting_Train = pd.DataFrame()
    Fitting_Target = pd.DataFrame()
    for steps in steps_tested:
        step_start_date = steps[0:2]
        step_end_date = steps[-2:]
        Train_Dataframe = prepare_training_file(stepstart =step_start_date, stepend=step_end_date)
        Fitting_Train = Fitting_Train.append(Train_Dataframe)
        Target_Dataframe = prepare_target_file(stepstart =step_start_date, stepend=step_end_date)
        Fitting_Target = Fitting_Target.append(Target_Dataframe)
    #Fitting_Target.to_csv(save_dir + '/' + 'FitTarget.csv')
    target = Fitting_Target.pop('Target')
    nptrain = Fitting_Train.values
    nptarget = target.values
    #nptrain = nptrain.reshape((21,1))
    model = get_compiled_model(inputtrain=nptrain,inputtarget=nptarget)
    model.fit(x=nptrain,y=nptarget,batch_size= 10 , epochs=3, verbose =1,shuffle=False)
    #model.summary()
    #pesos = model.get_weights()
    #print('Train shape', nptrain.shape)
    #print('Train shape', nptrain.shape)
    #print('Target shape', nptarget.shape)
    #print(pesos)

steps_tested = create_steps_list()
print("This is the Step List:", steps_tested)
Benchmark = create_benchmark_list()
training_steps()