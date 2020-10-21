import time
import os
import pandas as pd
import shutil
import xml.etree.ElementTree as et

FOLDER_PATH = "C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028"
REPORT_PATH = os.path.join(FOLDER_PATH, 'reports')
TESTER_PATH = os.path.join(FOLDER_PATH, 'MQL5', 'Profiles', 'Tester')
OPTIMIZED_VARIABLES = 4
MT5_PATH = "C:/Program Files/Darwinex MetaTrader 5/terminal64.exe"
OPTIMIZED_VARIABLES = 4 #Number of variables to be optimized


BOT_MAGIC_NUMBER_SERIES = '01'
USER_SERIES = '01'


# Hill Climbing Step
# user must define the number of laps (loops) that will go through the Optiset EG: if it has 7 Optimizable variables then 2 laps should be 14 separate Opti
# needs to read previous Phase 2 Results UNFILTERED     EG:    OptiResults-EA-S1v2-EURCHF-H1-Phase2.Complete
# needs to read the number of rows that finish in "Y" in the Optiset    EG:     Phase1-EA-S1v2
# start a loop that: 1.- Creates a INIT optimizing only "H" variable using a previously created HOptiset.
#                    2.- Save that result with a OptiResults-EA-S1v2-EURCHF-H1-Phase2.Complete     H name avoid rewriting, only annexes values
#                    3.- Set Highest value from Opti as Base Value for next Opti       BaseValue |  xmin  | xstep  | xmax     Y
class HillClimbPhase2:
    def __init__(self, dto):
        self.bot = dto.bot
        self.pairs = dto.pairs
        self.time_frames = dto.time_frames
        self.dto = dto

    def run(self):
        count = 0
        laps = 3
        start = time.time()

        base_optiset = os.path.join(TESTER_PATH, 'HCB-Phase1-{}.set'.format(self.bot))
        self.fill_base_optiset(base_optiset)
        while count < laps:
            with open(base_optiset, 'r') as f2:
                HCtext = f2.readlines()
                f2.close()
                print('top', HCtext)

            self.create_ini()
            print('INIT File Created for {}'.format(v))

            path = os.path.join(REPORT_PATH, self.bot, 'INITS', 'HC')
            for file in os.listdir(path):
                start = time.time()
                print(str((MT5_PATH + " /config:" + "{}/".format(FOLDER_PATH) + "reports/{}/INITS/HC/{}".format(self.bot, file))))
                process = subprocess.call(MT5_PATH + " /config:C:\\Users\\bryan\\AppData\\Roaming\\MetaQuotes\\Terminal\\6C3C6A11D1C3791DD4DBF45421BF8028\\reports\\{}\\INITS\\HC\{}".format(self.bot, file))
                end = time.time()
                print('Duration for HC on {} was of'.format(v), (end - start), 'seconds')

                null_values_index = 9 + OPTIMIZED_VARIABLES
                null_values_columns = 8 + OPTIMIZED_VARIABLES

                csv_file_name = os.path.join(REPORT_PATH, self.bot, 'INITS', 'HC-Phase1-{}-EURUSD-H1.csv'.format(self.bot))
                csv_list_hc = self.get_csv_list()

                dfHC = pd.DataFrame(data=csv_list_hc)
                dfHC = dfHC.drop(dfHC.index[:null_values_index])
                dfHC_columns_name = csv_list_hc[null_values_columns]
                dfHC.columns = dfHC_columns_name
                dfHC = dfHC.apply(pd.to_numeric)
                dfHC['Absolute DD'] = dfHC['Profit'] / dfHC['Recovery Factor']
                dfHC = dfHC.apply(pd.to_numeric)
                dfHC.to_csv(csv_file_name, sep=',', index=False)
                dfHC.reset_index(inplace=True)
                print('analizo resultados para', v)
                best_result = dfHC.loc[0, v]
                print('best result is:', best_result)

                self.write_best_value(best_result)
                print('Finished {} on Lap {}/{}'.format(v, count, laps))

            count += 1
            end = time.time()
            print('Finished Lap {} of {}'.format(count, laps))

    def create_ini(self, pair='EURUSD', time_frame='H1', optimization_criterion=0, model=2, optimization=1, shutdown=1, visual=0, leverage=33, replace_report=1, use_local=1, forward_mode=0, execution_mode=28):
        path = os.path.join(REPORT_PATH, self.bot, 'INITS', 'HC', 'INIT-HC-{}-{}-{}-Phase1.ini'.format(self.bot, pair, time_frame))
        file = open(path, 'w')

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
        'ExpertParameters= HCB-Phase1-{}.set'.format(self.bot) + "\n" \
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
        'Report=reports\{}\INITS\HC-Phase1-{}-{}-{}'.format(self.bot, self.bot, pair, time_frame) + "\n" \
        ';--- If the specified report already exists, it will be overwritten' + "\n" \
        'ReplaceReport={}'.format(replace_report) + "\n" \
        ';--- Set automatic platform shutdown upon completion of testing/optimization' + "\n" \
        'ShutdownTerminal={}'.format(shutdown) + "\n" \
        'Deposit={}'.format(self.dto.initial_deposit) + "\n" \
        'Currency={}'.format(self.dto.deposit_currency) + "\n" \
        ';Uses or refuses local network resources' + "\n" \
        'UseLocal={}'.format(use_local) + "\n" \
        ';Uses Visual test Mode' + "\n" \
        ';Visual={}'.format(visual) + "\n" \
        'ProfitInPips=0' + "\n" \
        'Leverage={}'.format(str(leverage)) + "\n")
        file.close()

    def fill_base_optiset(self, base_optiset):
        original_set = os.path.join(TESTER_PATH, 'Phase-{}.set'.format(self.bot))

        with open(original_set, 'r', encoding='utf-16') as f:
            lines = f.readlines()
            optimizablevars = []
            with open(base_optiset, "w") as f1:
                for line in lines:
                    if (line[0] != ';' or line[0] == ' ') and line[-2] == 'Y':
                        varname = line.split('=')
                        optimizablevars.append(varname[0])
                        varline = varname[1].split('||')
                        varline[-1] = varline[-1].replace("Y", "N")
                        line = '{}={}||{}||{}||{}||N\n'.format(varname[0], varline[0], varline[1], varline[2], varline[3])
                        f1.writelines(line)
                    else:
                        pass
                print('These variables are going to be optimized', optimizablevars)
                print('base_optiset for {} Created'.format(self.bot))

            f1.close()

    def write_best_value(self, best_result):
        """ Write Best Result Value from previous loop in Optiset text file"""
        optiset = os.path.join(TESTER_PATH, 'HC-Phase1-{}.set'.format(self.bot))

        with open(optiset, "r+") as file:
            text = file.readlines()
            print('this is HC text before loop line:', text)
            for line in text:

                if (line[0] != ';' or line[0] == ' ') and line[-2] == 'Y':
                    varname = line.split('=')
                    varline = varname[1].split('||')
                    if varname[0] == v:
                        line = '{}={}||{}||{}||{}||N\n'.format(varname[0], best_result, varline[1], varline[2], varline[3])
                        text.append(line)
                        print('this is a line:', line)
                    else:
                        line = '{}={}||{}||{}||{}||N\n'.format(varname[0], varline[0], varline[1], varline[2], varline[3])
                        text.append(line)
                        print('this is a line:', line)
            file.writelines(text)
            print('this is text down: ', text)
            file.close()

    def get_csv_list(self):
        path = os.path.join(REPORT_PATH, self.bot, 'INITS', 'HC-Phase1-{}-EURUSD-H1.xml'.format(self.bot))
        tree = et.parse(path)
        rootback = tree.getroot()
        csv_list = []
        for child in rootback:
            for section in child:
                for row in section:
                    row_list = []
                    csv_list.append(row_list)
                    for cell in row:
                        for data in cell:
                            row_list.append(data.text)

        return csv_list















