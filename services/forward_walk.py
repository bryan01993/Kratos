import time
import datetime
import os
from .helpers import add_months
from .create_init import CreateInit
from .launch_phase_wf import LaunchPhaseWF
from .accotate_results_fw import AccotateResultsFw
from .create_timebricks import CreateTimebricks
from .bt_sets_forward_walk import BTSetsForwardWalk

FOLDER_PATH = "C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028"
REPORT_PATH = os.path.join(FOLDER_PATH, 'reports')

class ForwardWalk:
    def __init__(self, dto):
        self.opti_start_date = dto.opti_start_date
        self.opti_end_date = dto.opti_end_date
        self.in_sample_end_date = dto.forward_date
        self.dto = dto

        # In months These variables are NOT INSIDE THE DTO, must add them
        self.time_step = 12
        self.is_ratio = 4
        self.oos_ratio = 1
        self.real_ratio = 1
        self.is_steps = self.time_step * self.is_ratio
        self.oos_steps = self.time_step * self.oos_ratio
        self.real_steps = self.time_step * self.real_ratio

        self.pairs = dto.pairs
        self.time_frames = dto.time_frames

    def run(self):
        list_bricks = CreateTimebricks(self.opti_start_date, self.time_step, self.is_steps, self.oos_steps, self.real_steps, self.opti_end_date)
        list_bricks = list_bricks.run()
        for pair in self.pairs:
            for time_frame in self.time_frames:
                for bricks in list_bricks:
                    self.iteration(pair, time_frame, bricks)
    
    def iteration(self, pair, time_frame, bricks):

        dto = self.dto
        dto.opti_start_date = bricks[0]
        dto.opti_end_date = bricks[2]
        dto.forward_date = bricks[1]
        dto.pair = pair
        dto.time_frame = time_frame

        CreateInit(self.dto, pair, time_frame, 'WF', 'opti', 1).create_init()
        LaunchPhaseWF(self.dto, pair, time_frame).run()
        AccotateResultsFw(self.dto).run()
        BTSetsForwardWalk(self.dto,pair,time_frame).run()
        LaunchPhaseWF(self.dto,pair,time_frame).run()

