import time
import datetime
import os
from .helpers import add_months
from .create_ini_fw import CreateIniWF
from .launch_phase_wf import LaunchPhaseWF
from .accotate_results_fw import AccotateResultsFw
from .create_timebricks import add_init_cuts
from .bt_sets_forward_walk import BTSetsForwardWalk

FOLDER_PATH = "C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028"
REPORT_PATH = os.path.join(FOLDER_PATH, 'reports')

class ForwardWalk:
    def __init__(self, dto):
        self.opti_start_date = dto.opti_start_date
        self.opti_end_date = dto.opti_end_date
        self.in_sample_end_date = dto.forward_date
        self.dto = dto

        # In month
        self.time_brick = 12

        # 5 In Sample time brick, 1 time brick outSample, and 1 time brick for real
        self.ratio = 4

        self.pairs = dto.pairs
        self.time_frames = dto.time_frames

    def run(self):
        list_bricks = add_init_cuts()
        for pair in self.pairs:
            for time_frame in self.time_frames:
                for bricks in list_bricks:
                    self.iteration(self.in_sample_end_date, pair, time_frame, bricks)
    
    def iteration(self, in_sample_start_date, pair, time_frame, bricks):

        in_sample_end_date = add_months(in_sample_start_date, self.time_brick * self.ratio)
        forward_date = add_months(in_sample_end_date, 1)

        dto = self.dto
        dto.opti_start_date = bricks[0]
        dto.opti_end_date = bricks[2]
        dto.forward_date = bricks[1]
        dto.real_date = bricks[3]
        dto.pair = pair
        dto.time_frame = time_frame

        CreateIniWF(self.dto, 1).create_init_file(pair, time_frame)
        LaunchPhaseWF(self.dto, pair, time_frame).run()
        AccotateResultsFw(self.dto).run()
        BTSetsForwardWalk(self.dto,pair,time_frame).run()
        LaunchPhaseWF(self.dto,pair,time_frame).run()

