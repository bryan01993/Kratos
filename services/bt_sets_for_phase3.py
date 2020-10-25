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

class BTSetsForPhase3:
    def __init__(self, dto):
        self.bot = dto.bot
        self.pairs = dto.pairs
        self.time_frames = dto.time_frames
        self.dto = dto

    def run(self):
        total_sets = 0
        start = time.time()

        for pair in self.pairs:
            for time_frame in self.time_frames:
                try:
                    self.create_ini_by_dataframe(pair, time_frame)
                except FileNotFoundError:
                    print('This Pair and Timeframe has no File', pair, time_frame)

        end = time.time()
        time = end - start
        print('A Total of ', total_sets, 'were created.')
        print('Done All Pairs and TimeFrames BT Sets Created in:', round(time, ndigits=2), 'seconds')

    def create_ini_by_dataframe(self, pair, time_frame):
        file_a = os.path.join(TESTER_PATH, 'Phase1-{}-set'.format(self.bot))
        csv_path = os.path.join(REPORT_PATH, self.bot, pair, time_frame, 'OptiResults-{}-{}-{}-Phase1.Complete-Filtered.csv'.format(self.bot, pair, time_frame))
        dataframe = pd.read_csv(csv_path)

        if (len(dataframe) < 1):
            return

        with open(file_a, 'r', encoding='utf-16') as file:
            var_name_list = []
            for g, row in dataframe.iterrows():
                file_b = os.path.join(TESTER_PATH, self.bot, 'Phase3-{}-{}-{}-{}.set'.format(self.bot, pair, time_frame, g))
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
                            value_spot = dataframe.iloc[g][x]
                        except KeyError:
                            continue
                        opti_format = "{}={}||1000||1000||2000||N \n".format(x, value_spot)  # a esta altura es donde ocurren los calculos
                        file1.write('{} \n'.format(opti_format))
                    magic_start_line = 'MagicStart={}{}{}{}{}||1000||1000||2000||N \n'.format(BOT_MAGIC_NUMBER_SERIES, USER_SERIES, self.pairs[pair], self.time_frames[time_frame], g)
                    file1.write(magic_start_line)
                self.create_ini(pair, time_frame, tail_number=g)
                print('INIT FILE for :', pair, time_frame, g, 'created')


    def create_ini(self, pair, time_frame, optimization_criterion=0, model=2, optimization=0, shutdown=1, visual=0, leverage=33, replace_report=1, use_local=1, forward_mode=0, execution_mode=28, phase=3, tail_number=0):
        file_name = 'INIT-BT-{}-{}-{}-Phase{}-{}.ini'.format(self.bot, pair, time_frame, phase, tail_number)
        path = os.path.join(REPORT_PATH, self.bot, 'INITS', 'Phase3', file_name)
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
        'ExpertParameters=\{}\Phase3-{}-{}-{}-{}.set'.format(self.bot, self.bot, pair, time_frame, tail_number) + "\n" \
        'Symbol={}'.format(pair) + 'MT5' + "\n" \
        'Period={}'.format(time_frame) + "\n" \
        ';Login=XXXXXX' + "\n" \
        'Model={}'.format(model) + "\n" \
        'ExecutionMode={}'.format(str(execution_mode)) + "\n" \
        'Optimization={}'.format(optimization) + "\n" \
        'OptimizationCriterion={}'.format(optimization_criterion) + "\n" \
        'FromDate={}'.format(self.dto.opti_start_date) + "\n" \
        'ToDate={}'.format(self.dto.opti_end_date) + "\n" \
        ';ForwardMode={}'.format(forward_mode) + "\n" \
        ';ForwardDate={}'.format(self.dto.forward_date) + "\n" \
        'Report=reports\{}\SETS\Phase3-{}-{}-{}-{}'.format(self.bot, self.bot, pair, time_frame, tail_number) + "\n" \
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