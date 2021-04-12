import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

# TEST AND DEBUGGING DATA
#base_dir = 'C:/Users/bryan/OneDrive/Desktop/Prueba de Seleccion/WF_Report'
#data_dir = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/Optimizaciones Listas/Data encadenada/EA-TS1v2 on GBPUSD on H1.csv'
#store_dir = 'C:/Users/bryan/OneDrive/Desktop/Prueba de Seleccion'
# Date Selection for graphics and comparison
#start_date = '2010.01.01'
#end_date = '2020.01.01'
class GraphOpti:
    """Graph based on all results of an optimization"""
    def __init__(self, base_dir, data_dir, store_dir, start_date, end_date): #Attributes
        self.base_dir = base_dir
        self.data_dir = data_dir
        self.store_dir = store_dir
        self.start_date = start_date
        self.end_date = end_date

    def open_opti_file(self): # Method
        """Transforms a .csv file data and returns a DataFrame"""
        file_dataframe = pd.read_csv(self.data_dir,index_col = 'index')
        return file_dataframe

    def graph_opti_file_histogram(self, dataframe, data_cut):
        """Shows a matplotlib histogram from back and forward data cuts"""
        n_points = 1000
        n_bins = 100
        back_graph = dataframe[data_cut]
        forward_data_cut = data_cut + 'Forward'
        forward_graph = dataframe[forward_data_cut]
        fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
        axs[0].axvline(back_graph.mean(), color='k', linestyle='dashed', linewidth=1) # Mean Dashed Line 1st graph
        plt.axvline(forward_graph.mean(), color='k', linestyle='dashed', linewidth=1) # Mean Dashed Line 2nd graph
        axs[0].hist(back_graph, bins=n_bins)
        axs[1].hist(forward_graph, bins=n_bins)
        axs[0].set_title('{}'.format(data_cut))
        axs[1].set_title('{}'.format(forward_data_cut))
        return axs[0], axs[1]

    def opti_stats(self, dataframe):
        """Gives Statistical values for every column on the dataframe"""
        return dataframe.describe()

    def run(self): # Method
        """Basic Test Run""" #Delete when class is finished
        object_print = self.open_opti_file()
        self.graph_opti_file_histogram(object_print,'Profit')
        plt.show()

if __name__ == '__main__':
    thisdata ='C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/Optimizaciones Listas/Data encadenada/EA-TS1v2 on GBPUSD on H1.csv'
    data1 = GraphOpti('base',thisdata,'store','start','end')
    print(data1.run())
