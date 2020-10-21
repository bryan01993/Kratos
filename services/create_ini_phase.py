import os

FOLDER_PATH = "C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028"
REPORT_PATH = os.path.join(FOLDER_PATH, 'reports')

OPTI_START_DATE = "2007.01.01" #YYYY.MM.DD
OPTI_END_DATE = "2020.01.01" #YYYY.MM.DD
FORWARD_DATE = "2019.01.01"

INITIAL_DEPOSIT = 10000
DEPOSIT_CURRENCY = "USD"

class CreateIniPhase:
    """Creates a phase INIT file for every pair and timeframe selected"""
    def __init__(self, bot, pairs, time_frames, phase):
        self.bot = bot
        self.pairs = pairs
        self.time_frames = time_frames
        self.phase = phase

    def run(self):
        count = 0
        for pair in self.pairs:
            for time_frame in self.time_frames:
                if self.pairs[pair].get() == 1 and self.time_frames[time_frame].get() == 1:
                    self.create_init_file(pair, time_frame)
                    count += 1
                    print(pair, time_frame, 'INIT Phase {} Created'.format(self.phase))

        print('INITS for All Phase {} Created. Total'.format(self.phase), count, 'INIT files.')

    def create_init_file(self, pair, time_frame, optimization_criterion=6, model=2, optimization=2, shutdown_terminal=1, visual=0, leverage_value=33, replace_report=1, use_local=1, forward_mode=4, execution_mode=28):
        """Creates the INIT file specific for a Phase  Optimization
            TODO: Get opti_start_date, opti_end_date and forward_date by params
        """

        file_name = 'INIT-{}-{}-{}-Phase{}.ini'.format(self.bot, pair, time_frame, self.phase)
        path = os.path.join(REPORT_PATH, self.bot, 'INITS', self.phase, file_name)
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
        'ExpertParameters=Phase{}-{}.set'.format(self.phase, self.bot) + "\n" \
        'Symbol={}'.format(pair) + 'MT5' + "\n" \
        'Period={}'.format(time_frame) + "\n" \
        ';Login=XXXXXX' + "\n" \
        'Model={}'.format(model) + "\n" \
        'ExecutionMode={}'.format(str(execution_mode)) + "\n" \
        'Optimization={}'.format(optimization) + "\n" \
        'OptimizationCriterion={}'.format(optimization_criterion) + "\n" \
        'FromDate={}'.format(OPTI_START_DATE) + "\n" \
        'ToDate={}'.format(OPTI_END_DATE) + "\n" \
        'ForwardMode={}'.format(forward_mode) + "\n" \
        'ForwardDate={}'.format(FORWARD_DATE) + "\n" \
        'Report=reports\\{}\\{}\\{}\OptiResults-{}-{}-{}-Phase{}'.format(self.bot, pair, time_frame, self.bot, pair, time_frame, self.phase) + "\n" \
        ';--- If the specified report already exists, it will be overwritten' + "\n" \
        'ReplaceReport={}'.format(replace_report) + "\n" \
        ';--- Set automatic platform shutdown upon completion of testing/optimization' + "\n" \
        'ShutdownTerminal={}'.format(shutdown_terminal) + "\n" \
        'Deposit={}'.format(INITIAL_DEPOSIT) + "\n" \
        'Currency={}'.format(DEPOSIT_CURRENCY) + "\n" \
        ';Uses or refuses local network resources' + "\n" \
        'UseLocal={}'.format(use_local) + "\n" \
        ';Uses Visual test Mode' + "\n" \
        ';Visual={}'.format(visual) + "\n" \
        'ProfitInPips=0' + "\n" \
        'Leverage={}'.format(str(leverage_value)) + "\n")

        file.close()
