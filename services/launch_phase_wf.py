import time
import os
import subprocess
import shutil

FOLDER_PATH = "C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028"
REPORT_PATH = os.path.join(FOLDER_PATH, 'reports')
CACHE_PATH = os.path.join(FOLDER_PATH, 'Tester', 'cache')
MT5_PATH = "C:/Program Files/Darwinex MetaTrader 5/terminal64.exe"

class LaunchPhaseWF:
    """ LaunchPhaseWF """
    def __init__(self, dto, pair, time_frame):
        self.bot = dto.bot
        self.pair = pair
        self.time_frame = time_frame

    def run(self):
        """Executes in CMD the INIT file on MT5 for a pair/time_frame"""

        folder_launch = os.path.join(REPORT_PATH, self.bot, self.pair, self.time_frame,  'WF_Inits')
        start = time.time()

        for file in os.listdir(folder_launch):
            start = time.time()

            report_file_name = "reports\\{}\\{}\\{}\\WF_Inits\\{}".format(self.bot, self.pair, self.time_frame, file)
            print(MT5_PATH + "/config:" + "{}/".format(FOLDER_PATH) + report_file_name)

            file_name = " /config:C:\\Users\\bryan\\AppData\\Roaming\\MetaQuotes\\Terminal\\6C3C6A11D1C3791DD4DBF45421BF8028\\"+report_file_name
            subprocess.call(MT5_PATH + file_name)

            file_path = os.path.join(REPORT_PATH, self.bot, self.pair, self.time_frame, 'WF_Inits', file)
            os.unlink(file_path)

        folder = CACHE_PATH
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as exception:
                print('Failed to delete %s. Reason: %s' % (file_path, exception))


        end = time.time()
        duration = (end - start) / 60
        print('Duration for WF_{}_{}_{} on'.format(self.bot, self.pair, self.time_frame) + 'was of', duration, 'minutes')
