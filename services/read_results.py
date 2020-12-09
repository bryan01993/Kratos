import time
import os
import subprocess
import shutil
import pandas as pd
from Dto import Dto
from IPython.display import display_html
FOLDER_PATH = "C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028"
REPORT_PATH = os.path.join(FOLDER_PATH, 'reports')
CACHE_PATH = os.path.join(FOLDER_PATH, 'Tester', 'cache')
bot = 'EA-B1v1'
pair = 'GBPUSD'
time_frame = 'H4'
MT5_PATH = "C:/Program Files/Darwinex MetaTrader 5/terminal64.exe"
"""
class ReadResults:
    Read results from HTML files for manipulation or analysis
    def __init__(self, dto, pair, time_frame):
        self.dto = dto
        self.bot = dto.bot
        self.pairs = dto.pairs
        self.time_frames = dto.time_frames
        self.phase = phase
"""

def run():
    # ir al directorio donde estan los resultados, e iterar el proceso
    path = os.path.join(REPORT_PATH, bot, pair, time_frame, 'WF_Results')
    #print(path)
    for file in os.listdir(path):
        if 'htm' in file:
            file_name = file
            file = os.path.join(path,file)
            #print('this one should go',file)
            file_trades = pd.read_html(file)
            #print(file_trades)
            report_stats = file_trades[0]
            report_results = report_stats[-30:-1]
            report_results = report_results.drop([1,2,4,5,8,9,12], axis=1)
            trade_record = file_trades[1]

            #trade_record.drop(trade_record.tail(2).index, inplace=True)


        else:
            pass
            #print('Not this one',file)
    #print(list(report_stats))
    print(type(report_stats))
    print(type(report_results))

    report_stats.to_csv('C:\\Users\\bryan\\OneDrive\\Desktop\\Prueba de Seleccion\\{}reportStats.csv'.format(file_name))
    trade_record.to_csv('C:\\Users\\bryan\\OneDrive\\Desktop\\Prueba de Seleccion\\{}trackRecord.csv'.format(file_name))
    report_results.to_csv('C:\\Users\\bryan\\OneDrive\\Desktop\\Prueba de Seleccion\\{}reportResults.csv'.format(file_name))

    print(report_stats[30:-1])
    #print(file_trades[1])
    print('Done')
run()