import time
import os
import pandas as pd
import shutil
from .create_init import CreateInit

FOLDER_PATH = "C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028"
REPORT_PATH = os.path.join(FOLDER_PATH, 'reports')
TESTER_PATH = os.path.join(FOLDER_PATH, 'MQL5', 'Profiles', 'Tester')
OPTIMIZED_VARIABLES = 4
INITIAL_DEPOSIT = 10000
DEPOSIT_CURRENCY = "USD"

BOT_MAGIC_NUMBER_SERIES = '01'
USER_SERIES = '01'

class BTSetsForwardWalk:
    def __init__(self, dto, pair, time_frame):
        self.bot = dto.bot
        self.pair = pair
        self.time_frame = time_frame
        self.dto = dto

    def run(self):
        total_sets = 0
        file_name = 'OptiWFResults-{}-{}-{}-{}-{}-Complete-Filtered.csv'.format(self.bot, self.pair, self.time_frame,self.dto.opti_start_date,self.dto.opti_end_date)
        file_name = os.path.join(REPORT_PATH, self.bot, self.pair, self.time_frame, 'WF_Report', file_name)
        self.create_ini_by_dataframe(self.pair, self.time_frame)
        total_sets += 1


    def create_ini_by_dataframe(self, pair, time_frame):
    
        file_name = 'OptiWFResults-{}-{}-{}-{}-{}-Complete-Filtered.csv'.format(self.bot, pair, time_frame,self.dto.opti_start_date,self.dto.opti_end_date)
        file_name = os.path.join(REPORT_PATH, self.bot, pair, time_frame, 'WF_Report', file_name)
        dataframe = pd.read_csv(file_name)

        file_a = os.path.join(FOLDER_PATH, 'MQL5/Profiles/Tester/Phase1-{}.set'.format(self.bot))
        total_sets = 0
        if (len(dataframe) < 1):
            return

        with open(file_a, 'r', encoding='utf-16') as file:
            var_name_list = []
            var_value_list= []
            for index, row in dataframe.iterrows():
                file_b = os.path.join(TESTER_PATH, self.bot, 'WF-{}-{}-{}.set'.format(self.bot, pair, time_frame))
                shutil.copyfile(file_a, file_b)
                total_sets += 1
                with open(file_b, 'w', encoding='utf-16') as file1:
                    for line in file:
                        if line[0] == ';' or line[0] == '\n':
                            continue
                        column_name = line.split('=')
                        var_name_list.append(column_name[0])
                        var_value_list.append(column_name[1])
                    for x,z in zip(var_name_list, var_value_list):
                        try:
                            value_spot = dataframe.iloc[index][x]
                            opti_format = "{}={}||1000||1000||2000||N \n".format(x,value_spot)  # a esta altura es donde ocurren los calculos
                        except KeyError:
                            opti_format = "{}={}\n".format(x, z)
                        file1.write('{} \n'.format(opti_format))
                    magic_start_line = 'MagicStart={}{}{}{}{}||1000||1000||2000||N \n'.format(BOT_MAGIC_NUMBER_SERIES, USER_SERIES, self.pair, self.time_frame, index)
                    file1.write(magic_start_line)

                print('INIT FILE for :', pair, time_frame, index, 'created')

