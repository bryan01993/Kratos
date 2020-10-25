import time
import os

FOLDER_PATH = "C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028"
REPORT_PATH = os.path.join(FOLDER_PATH, 'reports')
MT5_PATH = "C:/Program Files/Darwinex MetaTrader 5/terminal64.exe"

class LaunchPhase:
    def __init__(self, dto, phase):
        self.bot = dto.bot
        self.phase = phase

    def launch(self):
        """Executes in CMD the INIT file on MT5 for every pair and timeframe selected for Phase 1"""

        folder_launch = os.path.join(REPORT_PATH, self.bot, 'INITS', 'Phase{}'.format(self.bot))

        start = time.time()
        for file in os.listdir(folder_launch):
            start = time.time()
            print(str((MT5_PATH + " /config:" + "{}/".format(FOLDER_PATH) + "reports/{}/INITS/Phase{}/{}".format(self.bot, self.phase, file))))
            process = subprocess.call(MT5_PATH + " /config:C:\\Users\\bryan\\AppData\\Roaming\\MetaQuotes\\Terminal"
                                                "\\6C3C6A11D1C3791DD4DBF45421BF8028\\reports\\{}\\INITS\\Phase{}\{}"
                                    .format(self.bot, self.phase, file))
            end = time.time()
            duration = (end - start) / 60
            print('Duration for Phase {} on'.format(self.phase), self.bot, 'was of', duration, 'minutes')
        end = time.time()
        print('Launch from Phase {} Ended during a total'.format(self.phase), duration, 'minutes')
