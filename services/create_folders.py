import os

FOLDER_PATH = "C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028"
REPORT_PATH = os.path.join(FOLDER_PATH, 'reports')
TESTER_PATH = os.path.join(FOLDER_PATH, 'MQL5', 'Profiles', 'Tester')

PAIRS = {'GBPUSD':'01', 'EURUSD':'02', 'USDCAD':'03', 'USDCHF':'04', 'USDJPY':'05', 'GBPJPY':'06', 'EURAUD':'07', 'EURGBP':'08', 'EURJPY':'09', 'EURCHF':'10'}
TIME_FRAMES = {'H4':'96', 'H1':'95', 'M30':'94', 'M15':'93', 'M5':'92', 'M1':'91'}

class CreateFolders:
    def __init__(self, dto):
        self.bot = dto.bot
        self.pairs = dto.pairs
        self.time_frames = dto.time_frames

    def run(self):
        """Creates Folders for Results, Optisets and INIT files"""
        self.create_folder(os.path.join(REPORT_PATH, self.bot, 'INITS', 'Phase1'))
        self.create_folder(os.path.join(TESTER_PATH, self.bot))
        self.create_folder(os.path.join(REPORT_PATH, self.bot, 'INITS', 'HC'))
        self.create_folder(os.path.join(REPORT_PATH, self.bot, 'INITS', 'Phase2'))
        self.create_folder(os.path.join(REPORT_PATH, self.bot, 'INITS', 'Phase3'))
        self.create_folder(os.path.join(REPORT_PATH, self.bot, 'SETS'))

        for pair in self.pairs:
            for time_frame in self.time_frames:
                self.create_folder(os.path.join(REPORT_PATH, self.bot, pair, time_frame))
                self.create_folder(os.path.join(REPORT_PATH, self.bot, pair, time_frame, 'WF_Inits'))
                self.create_folder(os.path.join(REPORT_PATH, self.bot, pair, time_frame, 'WF_Results'))
        print("Path Folders for All Pairs and TimeFrames Have Been Created")

    def create_folder(self, path):
        """Creates Folders for Results, Optisets and INIT files"""
        try:
            os.makedirs(path)
            print('path created: ', path)
        except FileExistsError:
            print('path already exist: ', path)