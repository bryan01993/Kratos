import os
import pandas as pd

FOLDER_PATH = "C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028"
REPORT_PATH = os.path.join(FOLDER_PATH, 'reports')
TESTER_PATH = os.path.join(FOLDER_PATH, 'MQL5', 'Profiles', 'Tester')
OPTIMIZED_VARIABLES = 4

class AccotateOptisetsPhase2:
    """Generates the Optiset for the Results that passed the previous filter"""

    def __init__(self, dto):
        self.bot = dto.bot
        self.pairs = dto.pairs
        self.time_frames = dto.time_frames
        self.dto = dto

    def run(self):
        for pair in self.pairs:
            for time_frame in self.time_frames:
                try:
                    self.create_optisets(pair, time_frame)
                except FileNotFoundError:
                    print('This Pair and Timeframe has no File', pair, time_frame)
        print('All Pairs and TimeFrames Optisets Created')

    def create_optisets(self, pair, time_frame):
        path = os.path.join(TESTER_PATH, 'Phase1-{}.set'.format(self.bot))
        path2 = os.path.join(TESTER_PATH, self.bot, 'Phase3-{}-{}-{}.set'.format(self.bot, pair, time_frame))
        with open(path, 'r', encoding='utf-16') as file:
            with open(path2, 'w', encoding='utf-16') as file1:
                for line in file:
                    try:
                        self.create_optiset(line, file1, pair, time_frame)
                    except IndexError:
                        pass
            print('This Pair', pair, ' and TimeFrame ', time_frame, ' did not Pass Phase 2')

    def create_optiset(self, line, file, pair, time_frame):
        first_letter = line[0]
        if first_letter in (';', '\n', ''):
            file.write(line)
            return

        try:
            path_csv = 'OptiResults-{}-{}-{}-Phase2.Complete-Filtered.csv'.format(self.bot, pair, time_frame)
            path_csv = os.path.join(REPORT_PATH, self.bot, pair, time_frame, path_csv)
            dfOpti = pd.read_csv(path_csv)
            if (len(dfOpti)) < 1:
                return

            column_name = line.split('=')
            column_max = dfOpti['{}'.format(column_name[0])].max()
            column_min = dfOpti['{}'.format(column_name[0])].min()
            min_steps = 50
            if  column_max == column_min:
                if column_max <= 1:
                    column_min = 0
                    column_steps = 1
                    column_max = 1
                elif column_max >= 2:
                    column_min -= 1
                    column_steps = 1
                    column_max += 1
            elif (column_max - column_min) < 30:
                column_steps = 1
            else:
                column_steps = (dfOpti['{}'.format(column_name[0])].max() - dfOpti['{}'.format(column_name[0])].min())/min_steps
            if isinstance(column_max, str):
                pass
            column_base_value = column_name[1].split('|')
            format = "{}={}||{}||{}||{}||Y \n".format(column_name[0], column_base_value[0], column_min, column_steps, column_max)
            file.write('\n'+ format)
        except KeyError:
            file.write(line)
        except FileNotFoundError:
            return

