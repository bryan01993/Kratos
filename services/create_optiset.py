import pandas as pd
import os
import shutil
from Dto import Dto

FOLDER_PATH = "C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028"
REPORT_PATH = os.path.join(FOLDER_PATH, 'reports')
TESTER_PATH = os.path.join(FOLDER_PATH, 'MQL5', 'Profiles', 'Tester')
OPTIMIZED_VARIABLES = 4
INITIAL_DEPOSIT = 10000
DEPOSIT_CURRENCY = "USD"

BOT_MAGIC_NUMBER_SERIES = '01'
USER_SERIES = '01'

class CreateOptiset:
    """Generates the optiset(s) used in optimizations and/or backtests"""
    def __init__(self, bot, pair, time_frame, phase, data_path):
        #self.dto = dto
        self.bot = bot
        self.pair = pair
        self.time_frame = time_frame
        self.phase = phase
        self.data_path = data_path
        self.start_optiset = os.path.join(TESTER_PATH, 'Phase1-{}.set'.format(self.bot))
        print('This is start Optiset:', self.start_optiset)

    def create_dataframe(self):
        """DATAFRAME DE PRUEBA """
        dataframe_path = self.data_path
        dataframe = pd.read_csv(dataframe_path)
        return dataframe

    def accotate_optiset(self): # must add a dataframe (filtered or not should work the same way)
        """Obtains an optiset from a .csv file"""
        self.output_optiset =  os.path.join(TESTER_PATH, self.bot, 'Phase2-{}-{}-{}.set'.format(self.bot, self.pair, self.time_frame))
        with open(self.start_optiset, 'r', encoding='utf-16') as self.base_file:
            with open(self.output_optiset, 'w', encoding='utf-16') as self.final_file:
                for line in self.base_file:
                    try:
                        self.copy_row(line)
                    except IndexError: #did not find something???
                        pass

    def copy_row(self, line):
        """Copy the original optiset analizes the range based on the filters and exports an new optiset"""
        first_letter = line[0]
        if first_letter in (';', '\n', ''):
            self.final_file.write(line)
            return
        try:
            df_opti = pd.read_csv(self.data_path)
            if (len(df_opti) < 1):
                return
            column_name = line.split('=') # split the line
            column_max = df_opti['{}'.format(column_name[0])].max()
            column_min = df_opti['{}'.format(column_name[0])].min()
            if column_max == column_min:
                if column_max <= 1:
                    column_min = 0
                    column_max = 1
                elif column_max >= 2:
                    column_min -= 1
                    column_max += 1
            if isinstance(column_max, str):               #checks if the max value from a column is a str, will depend on the bot
                pass
            column_base_value = column_name[1].split('|')
            column_steps = column_base_value[4]
            opti_format = "{}={}||{}||{}||{}||Y \n".format(column_name[0], column_base_value[0], column_min, column_steps, column_max)

            if column_name[0] == 'Lots':
                opti_format = "{}={}||{}||{}||{}||N \n".format(column_name[0], column_base_value[0], column_min, column_steps, column_max)
            self.final_file.write('\n' + opti_format)

        except KeyError:
            self.final_file.write(line)
        except FileNotFoundError:
            pass

thisobject = CreateOptiset('EA-TS1v2','GBPJPY', 'M5', 1, 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/Optimizaciones Listas/Data encadenada/EA-TS1v2 on GBPJPY on M5 - TEST.csv').accotate_optiset()
    #defs
    #accotate_optiset(recibe un dataframe y busca maximos y minimos en cada columna basados en unos filtros)
    #generar set (recibe UNA LINEA, filtrada o no, y genera un .set en la carpeta de Profile/tester)
    #produce_all_sets(recibe un dataframe y crea un optiset por cada linea)

