import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

import numpy as np
# Data in candles directory
data_dir = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/CSV MODELOS'
store_dir = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/CSV Output'
# Date Selection for graphics and comparison
start_date = '2015.01.01'
end_date = '2020.01.01'
# Transform .csv data into a DataFrame

def plot_returns():
    for file in os.listdir(data_dir):
        file_dataframe = pd.read_csv(data_dir + '/' + file, sep='\t', index_col='<DATE>')
        file_dataframe['<RETURN>'] = (file_dataframe['<CLOSE>'] - file_dataframe['<CLOSE>'].shift(1,fill_value=0))/file_dataframe['<CLOSE>'].shift(1, fill_value=0)
        snipped_dataframe = file_dataframe[start_date:end_date]
        snipped_dataframe.iat[0,-1] = 0
        #snipped_dataframe.to_csv('C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/CSV MODELOS/checkinf.csv')
        #snipped_dataframe['<RETURN>'].plot(figsize=(16, 6))
        snipped_dataframe['<RETURN>'].plot.hist(bins = 100)
        kurt = snipped_dataframe.kurt(axis=0)
        print('File Name:', file)
        print(kurt)
        plt.show()


def create_dataframe(tf='H4'):

    for file in os.listdir(data_dir):
        file_pair = file[0:6]
        file_timeframe = file[10:13]
        if 'EURUSD' in file_pair and tf in file_timeframe:
            analysis_dataframe = pd.read_csv(data_dir + '/' + file, sep= '\t', index_col=['<DATE>','<TIME>'])
            analysis_dataframe = analysis_dataframe.drop(columns=['<OPEN>','<HIGH>','<LOW>','<CLOSE>','<VOL>','<SPREAD>','<TICKVOL>'])

    for file in os.listdir(data_dir):
        file_pair = file[0:6]
        file_timeframe = file[10:13]
        if tf in file_timeframe:
            file_dataframe = pd.read_csv(data_dir + '/' + file, sep='\t', index_col=['<DATE>','<TIME>'])
            file_dataframe = file_dataframe.rename(columns={'<CLOSE>': file_pair})
            analysis_dataframe = analysis_dataframe.join(file_dataframe[file_pair], how='left', on=['<DATE>','<TIME>'])
            #analysis_dataframe.to_csv(store_dir + '/' + tf + ' JointDataframe.csv')
    return analysis_dataframe

    print('Head:', '\n',analysis_dataframe.head())
    print('Head:', '\n',analysis_dataframe.tail())
    print('Finished')

data_to_plot = create_dataframe(tf='H4')
snipped_data = data_to_plot[start_date:end_date]
data_correlations = snipped_data.corr()
euro_corred = data_correlations.loc[(data_correlations['EURUSD']>0.7)]
print(data_correlations)
tf='H4'
data_correlations.to_csv(store_dir + '/' + tf + ' Correlations.csv')
def plot_correlations(df):
    sns.heatmap(df, cmap='rocket')
    plt.show()

create_dataframe(tf='H4')
plot_correlations(data_correlations)