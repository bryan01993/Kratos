# Modulo para Lanzamiento en vivo de los sets que si pasaron todos los filtros y que fueron seleccionados en el portafolio
import os
import subprocess
import time
""" NOTAS DE AVANCE: ya logro que se lance el bot en par y tf correcto en vivo, el problema es que borra la grafica una vez que se cierra entonces no se puede lanzar mas de una
Tareas : (los sets deben estar en la carpeta Launch)
1) En una iteracion de file on folder: separar el titulo de cada archivo que hay dentro de esa carpeta (EG: Phase3-EA-SA2v2-EURCHF-H1-4.htm)
2) Crear un archivo INIT con el nombre del bot
3) El Optiset tiene que estar en la carpeta de Presets, funciona para uno solo, no para una sucesion de Inis ni optisets en vivo"""
Real_Currency = "USD"
MT5_Path="C:/Program Files/Darwinex MetaTrader 5/terminal64.exe"
Launch_folder = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/Portfolios/Launch'
Launch_folder_Init = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/Portfolios/Launch/LAUNCH_INIT/'
Tom_Test_folder = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/Portfolios/Tom_Test/'
Tom_Test_Init_folder = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/Portfolios/Tom_Test/Tom_Test_INITS/'
#process = subprocess.call(MT5_Path + " /config:C:\\Users\\bryan\\AppData\\Roaming\\MetaQuotes\\Terminal\\6C3C6A11D1C3791DD4DBF45421BF8028\\MQL5\\Profiles\\Tester\\717.ini")
#process = subprocess.call(MT5_Path + " /config:C:\\Users\\bryan\\AppData\\Roaming\\MetaQuotes\\Terminal\\6C3C6A11D1C3791DD4DBF45421BF8028\\MQL5\\Profiles\\Tester\\718.ini")
Opti_start_date = '2018.01.01'
Opti_end_date = '2019.01.01'
ForwardDate = '2020.01.01'
Initial_deposit = 10000
#print('Done')


def LiveLaunch():
 for file in os.listdir(Launch_folder):
  try:
   file = file.split('.htm')
   print('this is file:',file)
   FileSplitted = file[0].split('-')
   FileEA = FileSplitted[1] + '-' + FileSplitted[2]
   print('this is fileEA:',FileEA)
   FilePair = FileSplitted[3]
   print('this is filepair:',FilePair)
   FileTimeFrame = FileSplitted[4]
   print('this is fileTF:',FileTimeFrame)
   FileInit = FileSplitted[0] + '-' + FileEA + '-' + FilePair + '-' + FileTimeFrame + '-' + FileSplitted[5]
   FileOptiset = FileSplitted[0] + '-' + FileEA + '-' + FilePair + '-' + FileTimeFrame + '-' + FileSplitted[5] + '.set'
   print('this is fileoptiset complete:',FileOptiset)
  except IndexError:
   print('This is not a Launchable Set:',file)

 def CreateInitLive (FileInit=file[0],LaunchEA = 'EA-SA2v2',PairList ='EURUSD',TimeFrameList ='H1',OptimizationCriterionList=0,ModelList=2,OptimizationList=0,ShutdownTerminalList=1,VisualList=0,LeverageValue=33,ReplaceReportList=1,UseLocalList=1,ForwardModeList=0,ExecutionValue=28,Phase=3,TailNumber=0):
  f =open(Launch_folder_Init + '{}.ini'.format(FileInit),"w")
  f.write('[Common]' + "\n" \
  'Login=3000018800' + "\n" \
  'Password=CpkBVUH2c4' + "\n"  \
  ';Funciona la parte anterior' + "\n"  \
  '\n' \
  '[Charts]' + "\n" \
  'ProfileLast = Default' + "\n" \
  'MaxBars=50000' + "\n" \
  'PrintColor=0' + "\n" \
  'SaveDeleted=1' + "\n" \
  '\n' \
  '[Experts]' + "\n" \
  'AllowLiveTrading=1' + "\n" \
  'AllowDllImport=1' + "\n" \
  'Enabled=1' + "\n" \
  'Account=0' + "\n" \
  'Profile=0' + "\n" \
  '\n' \
  '[StartUp]' + "\n" \
  'Expert=Advisors\{}'.format(LaunchEA) + "\n" \
  'ExpertParameters={}'.format(FileOptiset) + "\n" \
  ';Script=Examples\ObjectSphere\SphereSample' + "\n" \
  'Symbol={}'.format(PairList) + "\n" \
  'Period={}'.format(TimeFrameList) + "\n" \
  'Template=Standard.tpl' + "\n" )
  f.close()
  f = open(Launch_folder_Init + '{}.ini'.format(FileInit), "r")
 CreateInitLive(LaunchEA = FileEA, PairList = FilePair,FileInit=FileInit,TimeFrameList = FileTimeFrame)
 print(FileInit)
 process = subprocess.call(MT5_Path + " /config:C:\\Users\\bryan\\AppData\\Roaming\\MetaQuotes\\Terminal\\6C3C6A11D1C3791DD4DBF45421BF8028\\reports\\Portfolios\\Launch\\LAUNCH_INIT\\{}.ini".format(FileInit))
 print(MT5_Path + " /config:C:\\Users\\bryan\\AppData\\Roaming\\MetaQuotes\\Terminal\\6C3C6A11D1C3791DD4DBF45421BF8028\\reports\\Portfolios\\Launch\\LAUNCH_INIT\\{}.ini".format(FileInit))
#LiveLaunch()

def Tom_Test():
 for file in os.listdir('C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/Portfolios/Tom_Test/'):
  TomFullStart = time.time()
  try:
   file = file.split('.htm')
   print("This set file:",file[0],"is Tom-Testable.")
   FileSplitted = file[0].split('-')
   FileEA = FileSplitted[1] + '-' + FileSplitted[2]
   FilePair = FileSplitted[3]
   FileTimeFrame = FileSplitted[4]
   FileInit = FileSplitted[0] + '-' + FileEA + '-' + FilePair + '-' + FileTimeFrame + '-' + FileSplitted[5]
   FileOptiset = FileSplitted[0] + '-' + FileEA + '-' + FilePair + '-' + FileTimeFrame + '-' + FileSplitted[5] + '.set'
   def CreateTomIniReal (PairList =FilePair,TimeFrameList =FileTimeFrame,OptimizationCriterionList=0,ModelList=2,OptimizationList=0,ShutdownTerminalList=1,VisualList=0,LeverageValue=33,ReplaceReportList=1,UseLocalList=1,ForwardModeList=0,ExecutionValue=28,Phase=3,TailNumber=0):
    f =open(Tom_Test_Init_folder + 'INIT-TTR-{}.ini'.format(FileInit),"w")
    f.write(';[Common]' + "\n" \
    'Login=3000018800' + "\n" \
    'Password=CpkBVUH2c4' + "\n"  \
    ';[Charts]' + "\n" \
    ';[Experts]' + "\n" \
    'AllowLiveTrading=1' + "\n" \
    'AllowDllImport=1' + "\n" \
    'Enabled=1' + "\n" \
    '\n' \
    '[Tester]' + "\n" \
    'Expert=Advisors\{}'.format(FileEA) + "\n" \
    'ExpertParameters=\{}\{}'.format(FileEA,FileOptiset) + "\n" \
    'Symbol={}'.format(PairList) + "\n" \
    'Period={}'.format(TimeFrameList) + "\n" \
    ';Login=XXXXXX' + "\n" \
    'Model={}'.format(ModelList) + "\n" \
    'ExecutionMode={}'.format(str(ExecutionValue)) + "\n" \
    'Optimization={}'.format(OptimizationList) + "\n" \
    'OptimizationCriterion={}'.format(OptimizationCriterionList) + "\n" \
    'FromDate={}'.format(Opti_start_date) + "\n" \
    'ToDate={}'.format(Opti_end_date) + "\n" \
    ';ForwardMode={}'.format(ForwardModeList) + "\n" \
    ';ForwardDate={}'.format(ForwardDate) + "\n" \
    'Report=reports\Portfolios\Tom_Test\Tom_Test_Results\\TTR-{}'.format(FileInit) + "\n" \
    ';--- If the specified report already exists, it will be overwritten' + "\n" \
    'ReplaceReport={}'.format(ReplaceReportList) + "\n" \
    ';--- Set automatic platform shutdown upon completion of testing/optimization' + "\n" \
    'ShutdownTerminal={}'.format(ShutdownTerminalList) + "\n" \
    'Deposit={}'.format(Initial_deposit) + "\n" \
    'Currency={}'.format(Real_Currency) + "\n" \
    ';Uses or refuses local network resources' + "\n" \
    'UseLocal={}'.format(UseLocalList) + "\n" \
    ';Uses Visual test Mode' + "\n" \
    ';Visual={}'.format(VisualList) + "\n" \
    'ProfitInPips=0' + "\n" \
    'Leverage={}'.format(str(LeverageValue)) + "\n" )
    f.close()

   def CreateTomIniData (PairList =FilePair,TimeFrameList =FileTimeFrame,OptimizationCriterionList=0,ModelList=2,OptimizationList=0,ShutdownTerminalList=1,VisualList=0,LeverageValue=33,ReplaceReportList=1,UseLocalList=1,ForwardModeList=0,ExecutionValue=28,Phase=3,TailNumber=0):
    f =open(Tom_Test_Init_folder + 'INIT-TTD-{}.ini'.format(FileInit),"w")
    f.write(';[Common]' + "\n" \
    'Login=3000018800' + "\n" \
    'Password=CpkBVUH2c4' + "\n"  \
    ';[Charts]' + "\n" \
    ';[Experts]' + "\n" \
    'AllowLiveTrading=1' + "\n" \
    'AllowDllImport=1' + "\n" \
    'Enabled=1' + "\n" \
    '\n' \
    '[Tester]' + "\n" \
    'Expert=Advisors\{}'.format(FileEA) + "\n" \
    'ExpertParameters=\{}\{}'.format(FileEA,FileOptiset) + "\n" \
    'Symbol={}MT5'.format(PairList) + "\n" \
    'Period={}'.format(TimeFrameList) + "\n" \
    ';Login=XXXXXX' + "\n" \
    'Model={}'.format(ModelList) + "\n" \
    'ExecutionMode={}'.format(str(ExecutionValue)) + "\n" \
    'Optimization={}'.format(OptimizationList) + "\n" \
    'OptimizationCriterion={}'.format(OptimizationCriterionList) + "\n" \
    'FromDate={}'.format(Opti_start_date) + "\n" \
    'ToDate={}'.format(Opti_end_date) + "\n" \
    ';ForwardMode={}'.format(ForwardModeList) + "\n" \
    ';ForwardDate={}'.format(ForwardDate) + "\n" \
    'Report=reports\Portfolios\Tom_Test\Tom_Test_Results\\TTD-{}'.format(FileInit) + "\n" \
    ';--- If the specified report already exists, it will be overwritten' + "\n" \
    'ReplaceReport={}'.format(ReplaceReportList) + "\n" \
    ';--- Set automatic platform shutdown upon completion of testing/optimization' + "\n" \
    'ShutdownTerminal={}'.format(ShutdownTerminalList) + "\n" \
    'Deposit={}'.format(Initial_deposit) + "\n" \
    'Currency={}'.format(Real_Currency) + "\n" \
    ';Uses or refuses local network resources' + "\n" \
    'UseLocal={}'.format(UseLocalList) + "\n" \
    ';Uses Visual test Mode' + "\n" \
    ';Visual={}'.format(VisualList) + "\n" \
    'ProfitInPips=0' + "\n" \
    'Leverage={}'.format(str(LeverageValue)) + "\n" )
    f.close()
   #CreateTomIniReal()
   CreateTomIniData()
   print("Created Tom INITS for Data and Real for {}.".format(file[0]))
  except IndexError:
    print('This is not a Tom-Testable Set:',file)
    pass
 filecount = 0
 for file in os.listdir(Tom_Test_Init_folder):
  print("Now Testing: ",file)
  Tom_Test_Process = subprocess.call(MT5_Path + " /config:C:\\Users\\bryan\\AppData\\Roaming\\MetaQuotes\\Terminal\\6C3C6A11D1C3791DD4DBF45421BF8028\\reports\\Portfolios\\Tom_Test\\Tom_Test_INITS\\{}".format(file))
  filecount += 1
 TomEndTime= time.time()
 Tom_time = TomEndTime-TomFullStart
 print("Tom Test Finished in {} seconds and processed a total of {} Sets.".format(round(Tom_time, ndigits=2),filecount/2))

Tom_Test()