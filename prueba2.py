def HCInit (PairList ='EURUSD',TimeFrameList ='H1',OptimizationCriterionList=0,ModelList=2,OptimizationList=1,ShutdownTerminalList=1,VisualList=0,LeverageValue=33,ReplaceReportList=1,UseLocalList=1,ForwardModeList=0,ExecutionValue=28):
    f =open(FOLDER_PATH +'/reports/{}/INITS/HC/'.format(BotName.get()) + 'INIT-HC-{}-{}-{}-Phase1.ini'.format(BotName.get(),PairList,TimeFrameList),'w')
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
    'Symbol={}'.format(PairList) + 'MT5' + "\n" \
    'Period={}'.format(TimeFrameList) + "\n" \
    ';Login=XXXXXX' + "\n" \
    'Model={}'.format(ModelList) + "\n" \
    'ExecutionMode={}'.format(str(ExecutionValue)) + "\n" \
    'Optimization={}'.format(OptimizationList) + "\n" \
    'OptimizationCriterion={}'.format(OptimizationCriterionList) + "\n" \
    'FromDate={}'.format(Opti_start_date.get()) + "\n" \
    'ToDate={}'.format(Opti_end_date.get()) + "\n" \
    ';ForwardMode={}'.format(ForwardModeList) + "\n" \
    ';ForwardDate={}'.format(ForwardDate.get()) + "\n" \
    'Report=reports\{}\INITS\HC-Phase1-{}-{}-{}'.format(BotName.get(),BotName.get(),PairList,TimeFrameList) + "\n" \
    ';--- If the specified report already exists, it will be overwritten' + "\n" \
    'ReplaceReport={}'.format(ReplaceReportList) + "\n" \
    ';--- Set automatic platform shutdown upon completion of testing/optimization' + "\n" \
    'ShutdownTerminal={}'.format(ShutdownTerminalList) + "\n" \
    'Deposit={}'.format(Initial_deposit) + "\n" \
    'Currency={}'.format(Deposit_Currency) + "\n" \
    ';Uses or refuses local network resources' + "\n" \
    'UseLocal={}'.format(UseLocalList) + "\n" \
    ';Uses Visual test Mode' + "\n" \
    ';Visual={}'.format(VisualList) + "\n" \
    'ProfitInPips=0' + "\n" \
    'Leverage={}'.format(str(LeverageValue)) + "\n")
    f.close()