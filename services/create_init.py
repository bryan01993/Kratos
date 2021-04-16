import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import path
FOLDER_PATH = "C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028"
REPORT_PATH = os.path.join(FOLDER_PATH, 'reports')

class CreateInit:
    """Creates INIT files for optimization and/or backtesting"""
    def __init__(self, dto, pair, time_frame, phase):
        self.dto = dto
        self.bot = dto.bot
        self.pair = pair
        self.time_frame = time_frame
        self.phase = phase


    def create_init_simple_opti(self):
        self.init_path = self.init_path = 'INIT-{}-{}-{}-Phase{}.ini'.format(self.bot, self.pair, self.time_frame, self.phase)
        self.init_path = os.path.join(path.REPORT_PATH, self.bot, 'INITS', self.phase, self.init_path)
        self.optimization = 2
        self.optiset = 'Phase{}-{}.set'.format(self.phase, self.bot)
        self.results = 'reports\\{}\\{}\\{}\OptiResults-{}-{}-{}-Phase{}'.format(self.bot, self.pair, self.time_frame,self.bot, self.pair, self.time_frame,self.phase)
        return self.create_init()

    def create_init_simple_set(self):
        self.init_path = 'INIT-{}-{}-{}-Phase{}.ini'.format(self.bot, self.pair, self.time_frame, self.phase)
        self.init_path = os.path.join(path.REPORT_PATH, self.bot, 'INITS', self.phase, self.init_path)
        self.optimization = 0
        self.optiset = '\{}\Phase3-{}-{}-{}-{}.set'.format(self.bot, self.bot, self.pair, self.time_frame, tail_number)
        self.results = 'reports\{}\SETS\Phase3-{}-{}-{}-{}'.format(self.bot, self.bot, self.pair, self.time_frame, tail_number)
        return self.create_init()

    def create_init_wf_opti(self):
        self.init_path = 'WF-INIT-{}-{}-{}.ini'.format(self.bot, self.pair, self.time_frame)
        self.init_path = os.path.join(REPORT_PATH, self.bot, self.pair, self.time_frame, 'WF_Inits', self.init_path)
        self.optimization = 2
        self.optiset = 'Phase{}-{}.set'.format(self.phase, self.bot)
        self.results = 'reports\\{}\\{}\\{}\\{}\OptiWFResults-{}-{}-{}-{}-{}'.format(self.bot, self.pair, self.time_frame, 'WF_Report', self.bot, self.pair, self.time_frame, self.dto.opti_start_date, self.dto.opti_end_date)
        return self.create_init()

    def create_init_wf_set(self):
        self.init_path = 'WF-INIT-{}-{}-{}.ini'.format(self.bot, self.pair, self.time_frame)
        self.init_path = os.path.join(REPORT_PATH, self.bot, self.pair, self.time_frame, 'WF_Inits', self.init_path)
        self.optimization = 0
        self.optiset = '\{}\WF-{}-{}-{}.set'.format(self.bot, self.bot, self.pair, self.time_frame)
        self.results = 'reports\{}\{}\{}\WF_Results\WF-Phase3-{}-{}-{}-{}-{}'.format(self.bot, self.pair,self.time_frame, self.bot,self.pair, self.time_frame,self.dto.forward_date,self.dto.opti_end_date)
        return self.create_init()


    def create_init(self, optimization_criterion=6, model=2, shutdown_terminal=1, visual=0, leverage_value=33, replace_report=1, use_local=1, forward_mode=4, execution_mode=28):
        """Creates the INIT file specific for a Walk Forward Optimization"""
        path = os.path.join(REPORT_PATH, self.bot, self.pair, self.time_frame, 'WF_Inits', self.init_path)
        file = open(path, "w")
        file.write(';[Common]' + "\n" \
        ';Login=40539843' + "\n" \
        ';Password=jPHIWVnmZUFn' + "\n"  \
        ';[Charts]' + "\n" \
        ';[Experts]' + "\n" \
        'AllowLiveTrading=1' + "\n" \
        'AllowDllImport=1' + "\n" \
        'Enabled=1' + "\n" \
        '\n' \
        '[Tester]' + "\n" \
        'Expert=Advisors\\{}'.format(self.bot) + "\n" \
        'ExpertParameters={}'.format(self.optiset) + "\n" \
        'Symbol={}'.format(self.pair) + 'MT5' + "\n" \
        'Period={}'.format(self.time_frame) + "\n" \
        ';Login=XXXXXX' + "\n" \
        'Model={}'.format(model) + "\n" \
        'ExecutionMode={}'.format(str(execution_mode)) + "\n" \
        'Optimization={}'.format(self.optimization) + "\n" \
        'OptimizationCriterion={}'.format(optimization_criterion) + "\n" \
        'FromDate={}'.format(self.dto.opti_start_date) + "\n" \
        'ToDate={}'.format(self.dto.opti_end_date) + "\n" \
        'ForwardMode={}'.format(forward_mode) + "\n" \
        'ForwardDate={}'.format(self.dto.forward_date) + "\n" \
        'Report={}'.format(self.results) + "\n" \
        ';--- If the specified report already exists, it will be overwritten' + "\n" \
        'ReplaceReport={}'.format(replace_report) + "\n" \
        ';--- Set automatic platform shutdown upon completion of testing/optimization' + "\n" \
        'ShutdownTerminal={}'.format(shutdown_terminal) + "\n" \
        'Deposit={}'.format(self.dto.initial_deposit) + "\n" \
        'Currency={}'.format(self.dto.deposit_currency) + "\n" \
        ';Uses or refuses local network resources' + "\n" \
        'UseLocal={}'.format(use_local) + "\n" \
        ';Uses Visual test Mode' + "\n" \
        ';Visual={}'.format(visual) + "\n" \
        'ProfitInPips=0' + "\n" \
        'Leverage={}'.format(str(leverage_value)) + "\n")
        file.close()
