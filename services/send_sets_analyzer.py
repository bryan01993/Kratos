import os
import shutil
BOT_LIST = ['EA-B1v1', 'EA-T1v2', 'TendencialNuevo', 'EA-TS1v2']
PAIR_LIST = ['GBPUSD', 'EURUSD', 'USDCAD', 'USDCHF', 'USDJPY', 'GBPJPY', 'EURAUD', 'EURGBP', 'EURJPY', 'EURCHF']
TIME_FRAMES = ['H4', 'H1', 'M30', 'M15', 'M5', 'M1']

FOLDER_PATH = "C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028"

for bot in BOT_LIST:
    for pair in PAIR_LIST:
        for time_frame in TIME_FRAMES:
            try:
                REPORT_PATH = os.path.join(FOLDER_PATH, 'reports', bot, pair, time_frame,'WF_Results' )
                src_files = os.listdir(REPORT_PATH)
                for file_name in src_files:
                    full_file_name = os.path.join(REPORT_PATH, file_name)
                    if os.path.isfile(full_file_name):
                        shutil.copy(full_file_name, 'C:/Users/bryan/OneDrive/Desktop/liquadora')
                        print('Copy Done for: {}-{}-{}-{}'.format(file_name, bot, pair, time_frame))
            except FileNotFoundError:
                print('No data to copy on: {}-{}-{}-{}'.format(REPORT_PATH, bot, pair, time_frame))