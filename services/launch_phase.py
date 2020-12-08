import time
import os
import subprocess

FOLDER_PATH = "C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028"
REPORT_PATH = os.path.join(FOLDER_PATH, 'reports')
MT5_PATH = "C:/Program Files/Darwinex MetaTrader 5/terminal64.exe"

class LaunchPhase:
    def __init__(self, dto, phase):
        self.bot = dto.bot
        self.dto = dto
        self.phase = phase

    def run(self):
        """Executes in CMD the INIT file on MT5 for every pair and timeframe selected for Phase 1"""

        phase = 'Phase{}'.format(self.phase)
        folder_launch = os.path.join(REPORT_PATH, self.bot, 'INITS', phase)

        start = time.time()
        for file in os.listdir(folder_launch):
            start = time.time()
            report_file_name = "reports\\{}\\INITS\\Phase{}\\{}".format(self.bot, self.phase, file)
            print(MT5_PATH + " /config:" + "{}/".format(FOLDER_PATH) + report_file_name)
            subprocess.call(MT5_PATH + " /config:C:\\Users\\bryan\\AppData\\Roaming\\MetaQuotes\\Terminal\\6C3C6A11D1C3791DD4DBF45421BF8028\\" + report_file_name)
            end = time.time()
            duration = (end - start) / 60
            print('Duration for Phase {} on'.format(self.phase), self.bot, 'was of', duration, 'minutes')
        end = time.time()
        print('Launch from Phase {} Ended during a total'.format(self.phase), duration, 'minutes')
