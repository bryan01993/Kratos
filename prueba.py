def HCInit (pair ='EURUSD', time_frame ='H1', optimization_criterion=0, model=2, optimization=1, shutdown=1, visual=0, leverage=33, replace_report=1, use_local=1, forward_mode=0, execution_mode=28):
    f =open(FOLDER_PATH +'/reports/{}/INITS/HC/'.format(BotName.get()) + 'INIT-HC-{}-{}-{}-Phase1.ini'.format(BotName.get(), pair, time_frame),'w')
    f.write(';[Common]' + "\n" \
    ';Login=40539843' + "\n" \
    ';Password=jPHIWVnmZUFn' + "\n"  \
    ';[Charts]' + "\n" \
    ';[Experts]' + "\n" \
    'AllowLiveTrading=1' + "\n" \
    'AllowDllImport=1' + "\n" \
    'Enabled=1' + "\n" \
    '\n' \
    '[Tester]' + "\n" \
    'Expert=Advisors\{}'.format(BotName.get()) + "\n" \
    'ExpertParameters= HCB-Phase1-{}.set'.format(BotName.get()) + "\n" \
    'Symbol={}'.format(pair) + 'MT5' + "\n" \
    'Period={}'.format(time_frame) + "\n" \
    ';Login=XXXXXX' + "\n" \
    'Model={}'.format(model) + "\n" \
    'ExecutionMode={}'.format(str(execution_value)) + "\n" \
    'Optimization={}'.format(optimization) + "\n" \
    'OptimizationCriterion={}'.format(optimization_criterion) + "\n" \
    'FromDate={}'.format(Opti_start_date.get()) + "\n" \
    'ToDate={}'.format(Opti_end_date.get()) + "\n" \
    ';ForwardMode={}'.format(forward_mode) + "\n" \
    ';ForwardDate={}'.format(ForwardDate.get()) + "\n" \
    'Report=reports\{}\INITS\HC-Phase1-{}-{}-{}'.format(BotName.get(), BotName.get(), pair, time_frame) + "\n" \
    ';--- If the specified report already exists, it will be overwritten' + "\n" \
    'ReplaceReport={}'.format(replace_report) + "\n" \
    ';--- Set automatic platform shutdown upon completion of testing/optimization' + "\n" \
    'ShutdownTerminal={}'.format(shutdown) + "\n" \
    'Deposit={}'.format(Initial_deposit) + "\n" \
    'Currency={}'.format(Deposit_Currency) + "\n" \
    ';Uses or refuses local network resources' + "\n" \
    'UseLocal={}'.format(use_local) + "\n" \
    ';Uses Visual test Mode' + "\n" \
    ';Visual={}'.format(visual) + "\n" \
    'ProfitInPips=0' + "\n" \
    'Leverage={}'.format(str(leverage)) + "\n")
    f.close()