import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import path

class CreateFolders:
    """ Create all the folders"""
    def __init__(self, dto):
        self.bot = dto.bot
        self.pairs = dto.pairs
        self.time_frames = dto.time_frames

    def run(self):
        """Creates Folders for Results, Optisets and INIT files"""
        self.create_folder(os.path.join(path.REPORT_PATH, self.bot, 'INITS', 'Phase1'))
        self.create_folder(os.path.join(path.TESTER_PATH, self.bot))
        self.create_folder(os.path.join(path.REPORT_PATH, self.bot, 'INITS', 'HC'))
        self.create_folder(os.path.join(path.REPORT_PATH, self.bot, 'INITS', 'Phase2'))
        self.create_folder(os.path.join(path.REPORT_PATH, self.bot, 'INITS', 'Phase3'))
        self.create_folder(os.path.join(path.REPORT_PATH, self.bot, 'SETS'))

        for pair in self.pairs:
            for time_frame in self.time_frames:
                self.create_folder(os.path.join(path.REPORT_PATH, self.bot, pair, time_frame))
                self.create_folder(os.path.join(path.REPORT_PATH, self.bot, pair, time_frame, 'WF_Inits'))
                self.create_folder(os.path.join(path.REPORT_PATH, self.bot, pair, time_frame, 'WF_Results'))
        print("Path Folders for All Pairs and TimeFrames Have Been Created")

    def create_folder(self, file_path):
        """Creates Folders for Results, Optisets and INIT files"""
        try:
            os.makedirs(file_path)
            print('path created: ', file_path)
        except FileExistsError:
            print('path already exist: ', file_path)