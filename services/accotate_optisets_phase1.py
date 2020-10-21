import time
import os
import pandas as pd

FOLDER_PATH = "C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028"
REPORT_PATH = os.path.join(FOLDER_PATH, 'reports')
TESTER_PATH = os.path.join(FOLDER_PATH, 'MQL5', 'Profiles', 'Tester')
OPTIMIZED_VARIABLES = 4

class AccotateOptisetsPhase1:
    """Generates the Optiset for the Results that passed the previous filter"""

    def __init__(self, dto):
        self.bot = dto.bot
        self.pairs = dto.pairs
        self.time_frames = dto.time_frames
        self.dto = dto

    def run(self):
        full_start = time.time()
        for pair in self.pairs:
            for time_frame in self.time_frames:
                try:
                    self.create_optisets(pair, time_frame)
                except FileNotFoundError:
                    print('This Pair and Timeframe has no File', pair, time_frame)

        print('All Pairs and TimeFrames Optisets Created')
        full_end = time.time()
        print('Phase 1 Optisets Accotated in :', (full_end - full_start), ' seconds.')

    def create_optisets(self, pair, time_frame):
        path = os.path.join(TESTER_PATH, 'Phase1-{}.set'.format(self.bot))
        path2 = os.path.join(TESTER_PATH, self.bot, 'Phase2-{}-{}-{}.set'.format(self.bot, pair, time_frame))
        with open(path, 'r', encoding='utf-16') as file:
            with open(path2, 'w', encoding='utf-16') as file1:
                for line in file:
                    try:
                        self.create_optiset(line, file1, pair, time_frame)
                    except IndexError:
                        pass
        print('This Pair', pair, ' and TimeFrame ', time_frame, ' was not Launched')

    def create_optiset(self, line, file, pair, time_frame):
        first_letter = line[0]
        if first_letter in (';', '\n', ''):
            file.write(line)
            return

        try:
            path_csv = os.path.join(REPORT_PATH, self.bot, pair, time_frame, 'OptiResults-{}-{}-{}-Phase1-Filtered.csv'.format(self.bot, pair, time_frame))
            dfOpti = pd.read_csv(path_csv)
            if (len(dfOpti)) < 1:
                return

            column_name = line.split('=')
            column_max = dfOpti['{}'.format(column_name[0])].max()
            column_min = dfOpti['{}'.format(column_name[0])].min()
            if  column_max == column_min:
                if column_max <= 1:
                    column_min = 0
                    column_max = 1
                elif column_max >= 2:
                    column_min -= 1
                    column_max += 1
            if isinstance(column_max, str):
                pass
            column_base_value = column_name[1].split('|')
            column_steps = column_base_value[4]
            opti_format = "{}={}||{}||{}||{}||Y \n".format(column_name[0], column_base_value[0], column_min, column_steps, column_max)
            if  column_name[0] == 'Lots':
                opti_format = "{}={}||{}||{}||{}||N \n".format(column_name[0], column_base_value[0], column_min, column_steps, column_max)
            file.write('\n'+ opti_format)
        except KeyError:
            file.write(line)
        except FileNotFoundError:
            pass

