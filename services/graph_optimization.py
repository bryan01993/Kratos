import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

# Data in candles directory
base_dir = 'C:/Users/bryan/OneDrive/Desktop/Prueba de Seleccion/WF_Report'
data_dir = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/Optimizaciones Listas/Data encadenada/EA-B1v1 on GBPJPY on H4.csv'
store_dir = 'C:/Users/bryan/OneDrive/Desktop/Prueba de Seleccion'
# Date Selection for graphics and comparison
start_date = '2010.01.01'
end_date = '2020.01.01'
# Transform .csv data into a DataFrame

def open_optimization_file(path=data_dir):
    Opti_graph = pd.read_csv(path)
    n_points = 1000
    n_bins = 100
    back_profit = Opti_graph['Profit']
    forward_profit = Opti_graph['ProfitForward']
    Stats = Opti_graph.describe()
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    axs[0].axvline(back_profit.mean(), color='k', linestyle='dashed', linewidth=1)
    plt.axvline(forward_profit.mean(), color='k', linestyle='dashed', linewidth=1)
    file_start_date = path[-28:-26]
    file_end_date = path[-19:-17]
    #Stats.to_csv(store_dir + '/Describe-' + file_start_date + 'to' + file_end_date + '.csv')
    print(Stats)
    print('Back Profit mean is:', back_profit.mean())
    print('Forward Profit mean is:', forward_profit.mean())
    axs[0].hist(back_profit, bins=n_bins)
    axs[1].hist(forward_profit, bins=n_bins)
    axs[0].set_title('Profit from {} to {}'.format(file_start_date,file_end_date))
    #axs[1].set_title('Profit Forward from {} to {}'.format(file_start_date,file_end_date))
    return axs[0], axs[1]

open_optimization_file()
plt.show()
#iterate_complete_folder()