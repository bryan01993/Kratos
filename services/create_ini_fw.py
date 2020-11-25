import os
from .create_ini_phase import CreateIniPhase

FOLDER_PATH = "C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028"
REPORT_PATH = os.path.join(FOLDER_PATH, 'reports')

class CreateIniWF(CreateIniPhase):
    def create_init_file(self, pair, time_frame, optimization_criterion=6, model=2, optimization=2, shutdown_terminal=1, visual=0, leverage_value=33, replace_report=1, use_local=1, forward_mode=4, execution_mode=28):
        """Creates the INIT file specific for a Walk Forward Optimization"""
        file_name = 'WF-INIT-{}-{}-{}.ini'.format(self.bot, pair, time_frame)

        path = os.path.join(REPORT_PATH, self.bot, pair, time_frame, 'WF_Inits', file_name)

        print(path)
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
        'FromDate={}'.format(self.dto.opti_start_date) + "\n" \
        'ToDate={}'.format(self.dto.opti_end_date) + "\n" \
        'ForwardMode={}'.format(forward_mode) + "\n" \
        'ForwardDate={}'.format(self.dto.forward_date) + "\n" \
        'Report=reports\\{}\\{}\\{}\\{}\OptiWFResults-{}-{}-{}'.format(self.bot, pair, time_frame, 'WF_Report', self.bot, pair, time_frame) + "\n" \
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