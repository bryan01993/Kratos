import time
import datetime
import os
from .helpers import add_months
from .create_ini_fw import CreateIniWF
from .launch_phase_wf import LaunchPhaseWF
from .accotate_results_fw import AccotateResultsFw

FOLDER_PATH = "C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028"
REPORT_PATH = os.path.join(FOLDER_PATH, 'reports')

class ForwardWalk:
    def __init__(self, dto):
        self.opti_start_date = dto.opti_start_date
        self.opti_end_date = dto.opti_end_date
        self.in_sample_end_date = dto.forward_date
        self.dto = dto

        # In month
        self.time_brick = 6

        # 5 In Sample time brick, 1 time brick outSample, and 1 time brick for real
        self.ratio = 5

        self.pairs = ['GBPUSD', 'EURUSD', 'USDCAD', 'USDCHF', 'USDJPY', 'GBPJPY', 'EURAUD', 'EURGBP', 'EURJPY', 'EURCHF']
        self.time_frames = ['H4', 'H1', 'M30', 'M15', 'M5', 'M1']

    def run(self):
        for pair in self.pairs:
            for time_frame in self.time_frames:
                self.iteration(self.in_sample_end_date, pair, time_frame)
    
    def iteration(self, in_sample_start_date, pair, time_frame):
        in_sample_end_date = add_months(in_sample_start_date, self.time_brick * self.ratio)
        forward_date = add_months(in_sample_end_date, 1)

        dto = self.dto
        dto.opti_start_date = in_sample_start_date
        dto.opti_end_date = in_sample_end_date
        dto.forward_date = forward_date
        dto.pair = pair
        dto.time_frame = time_frame

        CreateIniWF(self.dto, 1).create_init_file(pair, time_frame)
        LaunchPhaseWF(self.dto, pair, time_frame).run()
        AccotateResultsFw(self.dto).run()
