import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

folder_name = 'EA-B1v2'
folder_pair = 'GBPJPY'
folder_timeframe = 'M15'
data_dir = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/'
store_dir = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/Optimizaciones Listas/Data encadenada/'

def create_dataframe_encadenado(folder_name=folder_name,folder_pair=folder_pair,folder_timeframe=folder_timeframe):
    """Crea un dataframe con todos los datos de walk forward junto con sus resultados de OOS """

    final_dir = data_dir + folder_name + '/' + folder_pair + '/' + folder_timeframe + '/' + 'WF_Report'
    dataframe_encadenado = pd.DataFrame()
    for file in os.listdir(final_dir):
        if 'Complete' in file and 'Filtered' not in file:
            file_df = final_dir + '/' + file
            df = pd.read_csv(file_df)
            dataframe_encadenado = dataframe_encadenado.append(df)

    print(dataframe_encadenado)
    dataframe_encadenado.to_csv(store_dir + "{} on {} on {}.csv".format(folder_name,folder_pair,folder_timeframe))
    dataframe_encadenado.drop(dataframe_encadenado[dataframe_encadenado['Result'] <= 0].index, inplace=True)
    print("Optimization saved For {} on {} on {} in Data encadenada".format(folder_name,folder_pair,folder_timeframe))
    sns.pairplot(dataframe_encadenado[['Result', 'Profit', 'Profit Factor', 'Trades', 'Forward Result']], diag_kind='kde')
    sns.pairplot(dataframe_encadenado[['ATRMultiplierTP', 'ATRMultiplierSL', 'WilliamsPeriod', 'KeltnerPeriod','ATRPeriod', 'Forward Result']], diag_kind='kde')
    plt.show()


def create_dataframe_encadenado_normalizado(folder_name=folder_name,folder_pair=folder_pair,folder_timeframe=folder_timeframe):
    """Crea un dataframe con todos los datos de walk forward normalizados junto con sus resultados de OOS """

    final_dir = data_dir + folder_name + '/' + folder_pair + '/' + folder_timeframe + '/' + 'WF_Report'
    dataframe_encadenado = pd.DataFrame()
    for file in os.listdir(final_dir):
        if 'Complete' in file and 'Filtered' not in file:
            file_df = final_dir + '/' + file
            df = pd.read_csv(file_df)
            dataframe_encadenado = dataframe_encadenado.append(df)

    dataframe_encadenado = (dataframe_encadenado - dataframe_encadenado.min()) / (dataframe_encadenado.max() - dataframe_encadenado.min())
    print(dataframe_encadenado)
    dataframe_encadenado.to_csv(store_dir + "{} on {} on {} normalizado.csv".format(folder_name,folder_pair,folder_timeframe))
    print("Optimization saved For {} on {} on {} in Data encadenada normalizado".format(folder_name,folder_pair,folder_timeframe))

create_dataframe_encadenado()
create_dataframe_encadenado_normalizado()