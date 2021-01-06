import time
import os
import pandas as pd
import shutil

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
        #start = time.time()
        #ini_created = self.create_ini_by_dataframe(self.pair,self.time_frame)
        #try:
        file_name = 'OptiWFResults-{}-{}-{}-{}-{}-Complete-Filtered.csv'.format(self.bot, self.pair, self.time_frame,self.dto.opti_start_date,self.dto.opti_end_date)
        file_name = os.path.join(REPORT_PATH, self.bot, self.pair, self.time_frame, 'WF_Report', file_name)
        self.create_ini_by_dataframe(self.pair, self.time_frame)
        total_sets += 1
        #except FileNotFoundError:
        #print('this is filename outside def',file_name)
        print('This Pair and Timeframe has no File', self.pair, self.time_frame)


        #end = time.time()
        #time = end - start
        print('Done All Pairs and TimeFrames BT Sets Created',total_sets)

    def create_ini_by_dataframe(self, pair, time_frame):
    
        file_name = 'OptiWFResults-{}-{}-{}-{}-{}-Complete-Filtered.csv'.format(self.bot, pair, time_frame,self.dto.opti_start_date,self.dto.opti_end_date)
        file_name = os.path.join(REPORT_PATH, self.bot, pair, time_frame, 'WF_Report', file_name)
        print('this is file_name',file_name)
        dataframe = pd.read_csv(file_name)

        file_a = os.path.join(FOLDER_PATH, 'MQL5/Profiles/Tester/Phase1-{}.set'.format(self.bot))
        total_sets = 0
        if (len(dataframe) < 1):
            return

        with open(file_a, 'r', encoding='utf-16') as file:
            var_name_list = []
            for index, row in dataframe.iterrows():
                file_b = os.path.join(TESTER_PATH, self.bot, 'WF-{}-{}-{}-{}.set'.format(self.bot, pair, time_frame, index))
                shutil.copyfile(file_a, file_b)
                total_sets += 1
                with open(file_b, 'w', encoding='utf-16') as file1:
                    for line in file:
                        if line[0] == ';' or line[0] == '\n':
                            continue
                        column_name = line.split('=')
                        var_name_list.append(column_name[0])
                    for x in var_name_list:
                        try:
                            value_spot = dataframe.iloc[index][x]
                        except KeyError:
                            continue
                        opti_format = "{}={}||1000||1000||2000||N \n".format(x, value_spot)  # a esta altura es donde ocurren los calculos
                        file1.write('{} \n'.format(opti_format))
                    magic_start_line = 'MagicStart={}{}{}{}{}||1000||1000||2000||N \n'.format(BOT_MAGIC_NUMBER_SERIES, USER_SERIES, self.pair, self.time_frame, index)
                    file1.write(magic_start_line)
                self.create_ini(pair, time_frame, tail_number=index)
                print('INIT FILE for :', pair, time_frame, index, 'created')


    def create_ini(self, pair, time_frame, optimization_criterion=0, model=2, optimization=0, shutdown=1, visual=0, leverage=33, replace_report=1, use_local=1, forward_mode=0, execution_mode=28, phase=3, tail_number=0):
        file_name = 'INIT-BT-{}-{}-{}-Phase{}-{}.ini'.format(self.bot, pair, time_frame, phase, tail_number)
        path = os.path.join(REPORT_PATH, self.bot, self.pair, self.time_frame, 'WF_Inits', file_name)
        file = open(path, "w")

        file.write(';[Common]' + "\n" \
        ';Login=40539843' + "\n" \
        ';Password=jPHIWVnmZUFn' + "\n"  \
        ';[Charts]' + "\n" \
        ';[Experts]' + "\n" \
        'AllowLiveTrading=1' + "\n" \
        'AllowDllImport=1' + "\n" \
        'Enabled=1' + "\n" \
        '\n' \
        '[Tester]' + "\n" \
        'Expert=Advisors\{}'.format(self.bot) + "\n" \
        'ExpertParameters=\{}\WF-{}-{}-{}-{}.set'.format(self.bot, self.bot, pair, time_frame, tail_number) + "\n" \
        'Symbol={}'.format(pair) + 'MT5' + "\n" \
        'Period={}'.format(time_frame) + "\n" \
        ';Login=XXXXXX' + "\n" \
        'Model={}'.format(model) + "\n" \
        'ExecutionMode={}'.format(str(execution_mode)) + "\n" \
        'Optimization={}'.format(optimization) + "\n" \
        'OptimizationCriterion={}'.format(optimization_criterion) + "\n" \
        'FromDate={}'.format(self.dto.forward_date) + "\n" \
        'ToDate={}'.format(self.dto.opti_end_date) + "\n" \
        ';ForwardMode={}'.format(forward_mode) + "\n" \
        ';ForwardDate={}'.format(self.dto.forward_date) + "\n" \
        'Report=reports\{}\{}\{}\WF_Results\WF-Phase3-{}-{}-{}-{}-{}'.format(self.bot, pair, time_frame, self.bot, pair, time_frame,self.dto.opti_end_date,self.dto.real_date) + "\n" \
        ';--- If the specified report already exists, it will be overwritten' + "\n" \
        'ReplaceReport={}'.format(replace_report) + "\n" \
        ';--- Set automatic platform shutdown upon completion of testing/optimization' + "\n" \
        'ShutdownTerminal={}'.format(shutdown) + "\n" \
        'Deposit={}'.format(INITIAL_DEPOSIT) + "\n" \
        'Currency={}'.format(DEPOSIT_CURRENCY) + "\n" \
        ';Uses or refuses local network resources' + "\n" \
        'UseLocal={}'.format(use_local) + "\n" \
        ';Uses Visual test Mode' + "\n" \
        ';Visual={}'.format(visual) + "\n" \
        'ProfitInPips=0' + "\n" \
        'Leverage={}'.format(str(leverage)) + "\n")

        file.close()
