import time
import os.path
import shutil
from tkinter import *
import tkinter as tk
import xml.etree.ElementTree as et
import subprocess
import pandas as pd
from bokeh.plotting import curdoc, figure
from bokeh.models import Select, CustomJS
from bokeh.layouts import row, column
from bokeh.io import output_notebook


#---------------------------------------------------VARIABLES PARA EL LANZAMIENTO ----------------------------------------------------
BotName = 'EA-R1v1'                           # EA Name OMITIR espacios en blancos, usar como simbolo solamente el "-".
BotMagicNumberSeries = '01'   # should be last numbers of the EA 09 for S3
UserSeries = '01' #01 La Opti la hizo bryan, 02 la hizo richard
#--------------------------------------------PATHS----------------------------------------------------------------------
MT5_PATH = "C:/Program Files/Darwinex MetaTrader 5/terminal64.exe"
FOLDER_PATH = "C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028"
FOLDER_LAUNCH = os.path.join(FOLDER_PATH, "reports/{}/INITS/Phase1".format(BotName))
#Launch_folder = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/Portfolios/Launch'
#Launch_folder_Init = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/Portfolios/Launch/LAUNCH_INIT/'
#Tom_Test_folder = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/Portfolios/Tom_Test/'
#Tom_Test_Init_folder = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/Portfolios/Tom_Test/Tom_Test_INITS/'
#------------------------------------------INIT Parameters--------------------------------------------------------------
Opti_start_date = "2007.01.01" #YYYY.MM.DD
Opti_end_date = "2020.01.01" #YYYY.MM.DD
Initial_deposit = 10000 #Default Value es 10000 para todos los analisis
Deposit_Currency = "USD" #Default Currency para todos los  analisis
Real_Currency = "EUR" #Default Currency para Prueba de Tom
PairList = {'GBPUSD':'01', 'EURUSD':'02', 'USDCAD':'03', 'USDCHF':'04', 'USDJPY':'05', 'GBPJPY':'06', 'EURAUD':'07', 'EURGBP':'08', 'EURJPY':'09', 'EURCHF':'10'} # List Of Pairs to select for launch.
TimeFrameList = {'H4':'96', 'H1':'95', 'M30':'94', 'M15':'93', 'M5':'92', 'M1':'91'} # Agregar 'D1', no parece obtener la data de M1 dentro del MT5.
ModelList = [0, 1, 2, 3] #(0 — "Every tick", 1 — "1 minute OHLC", 2 — "Open price only", 3 — "Math calculations", 4 — "Every tick based on real ticks"). If this parameter is not specified, Every Tick mode is used. Default 2
OptimizationList = [0, 1, 2] #(0 — optimization disabled, 1 — "Slow complete algorithm", 2 — "Fast genetic based algorithm", 3 — "All symbols selected in Market Watch")
OptimizationCriterionList = [0, 1, 2, 3, 4, 5, 6] #(0 — the maximum balance value, 1 — the maximum value of product of the balance and profitability, 2 — the product of the balance and expected payoff, 3 — the maximum value of the expression (100% - Drawdown)*Balance, 4 — the product of the balance and the recovery factor, 5 — the product of the balance and the Sharpe Ratio, 6 — a custom optimization criterion received from the OnTester() function in the Expert Advisor); Default 0
ForwardModeList = [0, 1, 2, 3, 4] # (0 — off, 1 — 1/2 of the testing period, 2 — 1/3 of the testing period, 3 — 1/4 of the testing period, 4 — custom interval specified using the ForwardDate parameter); Default 3
ForwardDate = "2019.01.01" #YYYY.MM.DD Only if ForwardModeList = 4
ReplaceReportList = [0, 1] #(0 — disable, 1 — enable). If overwriting is forbidden and a file with the same name already exists, a number in square brackets will be added to the file name. For example, tester[1].htm. If this parameter is not set, default 0 is used (overwriting is not allowed)
ShutdownTerminalList = [0, 1] #0 Won't close platform after finished, 1 will close the platform once finished; Default 1
UseLocalList = [0, 1] # 0 Uses NO local resources, 1 Uses Local Resources ; Default 0
VisualList = [0, 1] # Does not use Visual mode; Default 0
Phase = [1, 2, 3, 4, 5] # Phase in Production/Analisis Line
LeverageValue = 33
ExecutionValue = 28
OptimizedVariables = 4 #Number of variables to be optimized
#-------------------------------------------FILTERS FOR PHASE 1 RESULTS---------------------------------------------
FilterNetProfitPhase1 = 7000
FilterExpectedPayoffPhase1 = 8
FilterProfitFactorPhase1 = 1.29
FilterCustomPhase1 = -0.5
FilterEquityDDPhase1 = 1500 #Esta en valor absoluto, se encuentra despejando del Recovery Factor.
FilterTradesPhase1 = 200
ForwardFilterNetProfitPhase1 = 700
ForwardFilterExpectedPayoffPhase1 = 10
ForwardFilterProfitFactorPhase1 = 1.25
ForwardFilterCustomPhase1 = -0.5
ForwardFilterEquityDDPhase1 = 800 #Esta en valor absoluto, se encuentra despejando del Recovery Factor.
ForwardFilterTradesPhase1 = 20



#---------------------------------------------CREATES BOTNAME & INIT & OPTISETS & RESULTS FOLDERS---------------------------------------------------
def CreateALLFoldersPhase1():
    """Creates Folders for Results, Optisets and INIT files"""
    try:
        os.makedirs(FOLDER_PATH +'/reports/{}/INITS/Phase1'.format(BotName.get()))
        print('Directory for', BotName.get(), ' INITS Phase 1 Created')
    except FileExistsError:
        print('Directory for', BotName.get(), ' INITS Phase 1 already exists')

    try:
        os.makedirs(FOLDER_PATH +'/MQL5/Profiles/Tester/{}'.format(BotName.get()))
        print('Directory for', BotName.get(), ' OPTISETS Phase 2 Created')
    except FileExistsError:
        print('Directory for', BotName.get(), ' OPTISETS Phase 2 already exists')

    try:
        os.makedirs(FOLDER_PATH +'/reports/{}/INITS/HC'.format(BotName.get()))
        print('Directory for', BotName.get(), ' Hill Climbing Created')
    except FileExistsError:
        print('Directory for', BotName.get(), ' Hill Climbing already exists')

    try:
        os.makedirs(FOLDER_PATH +'/reports/{}/INITS/Phase2'.format(BotName.get()))
        print('Directory for', BotName.get(), ' Phase 2 Created')
    except FileExistsError:
        print('Directory for', BotName.get(), ' Phase 2 already exists')

    try:
        os.makedirs(FOLDER_PATH +'/reports/{}/INITS/Phase3'.format(BotName.get()))
        print('Directory for', BotName.get(), ' Phase 3 Created')
    except FileExistsError:
        print('Directory for', BotName.get(), ' Phase 3 already exists')

    try:
        os.makedirs(FOLDER_PATH +'/reports/{}/SETS'.format(BotName.get()))
        print('Directory for', BotName.get(), ' Sets Created')
    except FileExistsError:
        print('Directory for', BotName.get(), ' Sets already exists')

    for i in PairList:
        for j in TimeFrameList:
            try:
                os.makedirs(FOLDER_PATH +"/reports/"+"{}".format(BotName.get())+"/"+ i +"/"+ j)
                print("Directory for", BotName.get(), i, "and", j, "created")
            except FileExistsError:
                print("Directory for", BotName.get(), i, "and", j, "already exists")
    print("Path Folders for All Pairs and TimeFrames Have Been Created")

#----------------------------CREATES INIT FILES FOR PHASE 1-----(BLIND OPTI)--------------------------------------------
def CreateIniFilesPhase1(BotName="MACD Sample", PairList='EURUSD', TimeFrameList='H1', OptimizationCriterionList=6,
                         ModelList=2, OptimizationList=2, ShutdownTerminalList=1, VisualList=0, LeverageValue=33,
                         ReplaceReportList=1, UseLocalList=1, ForwardModeList=4, ExecutionValue=28, Phase=1):
    """Creates INIT file specific for the Phase 1 optimization"""
    f = open(FOLDER_PATH +'/reports/{}/INITS/Phase1/'.format(BotName.get()) + 'INIT-{}-{}-{}-Phase{}.ini'.
             format(BotName.get(), PairList, TimeFrameList, Phase), "w")
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
    'ExpertParameters=Phase{}-{}.set'.format(Phase, BotName.get()) + "\n" \
    'Symbol={}'.format(PairList) + 'MT5' + "\n" \
    'Period={}'.format(TimeFrameList) + "\n" \
    ';Login=XXXXXX' + "\n" \
    'Model={}'.format(ModelList) + "\n" \
    'ExecutionMode={}'.format(str(ExecutionValue)) + "\n" \
    'Optimization={}'.format(OptimizationList) + "\n" \
    'OptimizationCriterion={}'.format(OptimizationCriterionList) + "\n" \
    'FromDate={}'.format(Opti_start_date.get()) + "\n" \
    'ToDate={}'.format(Opti_end_date.get()) + "\n" \
    'ForwardMode={}'.format(ForwardModeList) + "\n" \
    'ForwardDate={}'.format(ForwardDate.get()) + "\n" \
    'Report=reports\{}\{}\{}\OptiResults-{}-{}-{}-Phase{}'.format(BotName.get(), PairList, TimeFrameList, BotName.get(), PairList, TimeFrameList, Phase) + "\n" \
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
    f = open(FOLDER_PATH +'/reports/{}/INITS/Phase1/'.format(BotName.get()) + 'INIT-{}-{}-{}-Phase{}.ini'.
             format(BotName.get(), PairList, TimeFrameList, Phase), "r")

#-----------------------------------CREATES INIT FILES FOR ALL PAIRS PHASE 1--------------------------------------------
def CreateIniForAllPhase1():
    """Creates INIT files for all pairs and timeframes selected"""
    inicount = 0
    for i in PairListTest:
        for j in TimeFrameListTest:
            if PairListTest[i].get() == 1:
                if TimeFrameListTest[j].get() == 1:
                    CreateIniFilesPhase1(BotName, PairList=i, TimeFrameList=j)
                    inicount += 1
                    print(i, j, 'INIT Phase 1 Created')
    print("INITS for All Phase 1 Created. Total", inicount, 'INIT files.')

#------------------------------------LAUNCHES INIT FILES IN CMD FOR PHASE 1---------------------------------------------
def LaunchPhase1():
    """Executes in CMD the INIT file on MT5 for every pair and timeframe selected for Phase 1"""
    FullStart = time.time()
    for file in os.listdir(FOLDER_LAUNCH):
        start = time.time()
        print(str((MT5_PATH + " /config:" + "{}/".format(FOLDER_PATH) + "reports/{}/INITS/Phase1/{}"
                   .format(BotName.get(), file))))
        process = subprocess.call(MT5_PATH + " /config:C:\\Users\\bryan\\AppData\\Roaming\\MetaQuotes\\Terminal"
                                             "\\6C3C6A11D1C3791DD4DBF45421BF8028\\reports\\{}\\INITS\\Phase1\{}"
                                  .format(BotName.get(), file))
        end = time.time()
        print('Duration for Phase 1 on', BotName.get(), 'was of', (end - start)/60, 'minutes')
    FullEnd = time.time()
    print('Launch from Phase 1 Ended during a total', (FullEnd-FullStart)/60, 'minutes')

#-------------PROCESS TO OBTAIN OPTISET VALUES FOR PHASE 2 --- CREATE CSV ACCOTATTED FROM XML AND FILTERS IT------------
def AccotateResultsPhase1():
    """Filtrates the Results for Phase 1 Optimization and keeps the results that passes"""

    print('Begins Phase 1 Results Filtering')
    TimeStartResultsPhase2start = time.time()
    Totalcount = 0
    Projectcount = 0

    def movecol(df, cols_to_move=[], ref_col='', place='After'):
        cols = df.columns.tolist()
        if place == 'After':
            seg1 = cols[:list(cols).index(ref_col) + 1]
            seg2 = cols_to_move
        if place == 'Before':
            seg1 = cols[:list(cols).index(ref_col)]
            seg2 = cols_to_move + [ref_col]
        seg1 = [i for i in seg1 if i not in seg2]
        seg3 = [i for i in cols if i not in seg1 + seg2]
        return df[seg1 + seg2 + seg3]
    for i in PairList:
        for j in TimeFrameList:
            try:
                # ----------------------------------------BACKTEST FILE-------------------------------------------------
                Nullvaluesindex = 9 + OptimizedVariables
                NullvaluesColumns = 8 + OptimizedVariables
                csvFileNameBack = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal//6C3C6A11D1C3791DD4DBF45421BF8028/' \
                                  '/reports/{}/{}/{}/OptiResults-{}-{}-{}-Phase1.csv'.format(BotName.get(), i, j,
                                                                                             BotName.get(), i, j)
                treeback = et.parse('C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal//6C3C6A11D1C3791DD4DBF45421BF8028/'
                                    '/reports/{}/{}/{}/OptiResults-{}-{}-{}-Phase1.xml'.format(BotName.get(), i, j
                                                                                               , BotName.get(), i, j))
                rootback = treeback.getroot()
                csv_list_back = []
                for child in rootback:
                    for Section in child:
                        for Row in Section:
                            Row_list_back = []
                            csv_list_back.append(Row_list_back)
                            for Cell in Row:
                                for Data in Cell:
                                    Row_list_back.append(Data.text)
                dfback = pd.DataFrame(data=csv_list_back)
                dfback = dfback.drop(dfback.index[:Nullvaluesindex])
                dfback_columns_name = csv_list_back[NullvaluesColumns]
                dfback.columns = dfback_columns_name  ###
                dfback['Lots'] = 0.1
                dfback['Average Loss'] = dfback['Result']
                dfback['Win Ratio'] = dfback['Result']
                dfback['Average Loss'] = dfback['Average Loss'].str.slice(start=-6, stop=-3)
                dfback['Win Ratio'] = dfback['Win Ratio'].str.slice(start=-3)
                dfback = dfback.apply(pd.to_numeric)
                dfback['Absolute DD'] = dfback['Profit'] / dfback['Recovery Factor']
                dfback = movecol(dfback, cols_to_move=['Absolute DD'], ref_col='Equity DD %')
                dfback = movecol(dfback, cols_to_move=['Average Loss'], ref_col='Equity DD %', place='Before')
                dfback = movecol(dfback, cols_to_move=['Win Ratio'], ref_col='Trades')
                dfback = movecol(dfback, cols_to_move=['Lots'], ref_col='Win Ratio')
                dfback = dfback.apply(pd.to_numeric)
                dfback.sort_values(by=['Pass'], ascending=False, inplace=True)
                dfback.to_csv(csvFileNameBack, sep=',', index=False)
                dfback.reset_index(inplace=True)
                #-----------------------------------------THIS IS NORMALIZATION-----------------------------------------
                for index, row in dfback.iterrows():
                    try:
                        AvgLossNorm = 100 / row['Average Loss']
                        AbsoluteDDNorm = float(FilterEquityDDPhase1.get()) / row['Absolute DD']
                        NormalizeRow = float(min(AvgLossNorm, AbsoluteDDNorm))
                        row['Lots'] = float(round(row['Lots'] * NormalizeRow, 2))
                        LotRatio = row['Lots'] / 0.1
                        row['Profit'] = row['Profit'] * LotRatio
                        row['Expected Payoff'] = row['Expected Payoff'] * LotRatio
                        row['Absolute DD'] = row['Absolute DD'] * LotRatio
                        row['Average Loss'] = row['Average Loss'] * LotRatio
                        dfback.loc[index, 'Profit'] = row['Profit']
                        dfback.loc[index, 'Expected Payoff'] = row['Expected Payoff']
                        dfback.loc[index, 'Absolute DD'] = row['Absolute DD']
                        dfback.loc[index, 'Average Loss'] = row['Average Loss']
                        dfback.loc[index, 'Lots'] = row['Lots']
                        pass
                    except IndexError:
                        pass
                print('Done Backtest Results for:', i, j)
                # ------------------------------------------FORWARD FILE------------------------------------------------
                csvFileNameForward = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal//6C3C6A11D1C3791DD4DBF45421BF8028/' \
                                     '/reports/{}/{}/{}/OptiResults-{}-{}-{}-Phase1.forward.csv'.format(BotName.get(), i, j,
                                                                                                        BotName.get(), i, j)
                treeforward = et.parse('C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal//6C3C6A11D1C3791DD4DBF45421BF8028/'
                                       '/reports/{}/{}/{}/OptiResults-{}-{}-{}-Phase1.forward.xml'.format(BotName.get(), i, j,
                                                                                                          BotName.get(), i, j))
                rootforward = treeforward.getroot()
                csv_list_forward = []
                for child in rootforward:
                    for Section in child:
                        for Row in Section:
                            Row_list_forward = []
                            csv_list_forward.append(Row_list_forward)
                            for Cell in Row:
                                for Data in Cell:
                                    Row_list_forward.append(Data.text)
                dfforward = pd.DataFrame(data=csv_list_forward)
                dfforward = dfforward.drop(dfforward.index[:Nullvaluesindex])
                dfforward_columns_name = csv_list_forward[NullvaluesColumns]
                dfforward.columns = dfforward_columns_name
                dfforward['Forward Lots'] = 0.1
                dfforward['Forward Average Loss'] = dfforward['Forward Result']
                dfforward['Forward Win Ratio'] = dfforward['Forward Result']
                dfforward['Forward Average Loss'] = dfforward['Forward Average Loss'].str.slice(start=-6, stop=-3)
                dfforward['Forward Win Ratio'] = dfforward['Forward Win Ratio'].str.slice(start=-3)
                dfforward = dfforward.apply(pd.to_numeric)
                dfforward['Forward Absolute DD'] = dfforward['Profit'] / dfforward['Recovery Factor']
                dfforward = movecol(dfforward, cols_to_move=['Forward Absolute DD'], ref_col='Equity DD %')
                dfforward = movecol(dfforward, cols_to_move=['Forward Average Loss'], ref_col='Equity DD %', place='Before')
                dfforward = movecol(dfforward, cols_to_move=['Forward Win Ratio'], ref_col='Trades')
                dfforward = movecol(dfforward, cols_to_move=['Forward Lots'], ref_col='Forward Win Ratio')
                dfforward = dfforward.apply(pd.to_numeric)
                dfforward.sort_values(by=['Pass'], ascending=False, inplace=True)
                dfforward.reset_index(inplace=True)
                dfforward.to_csv(csvFileNameForward, sep=',', index=False)
                dfforward.rename(columns={'Profit': 'Forward Profit', 'Expected Payoff': 'Forward Expected Payoff',
                                          'Profit Factor': 'Forward Profit Factor',
                                          'Recovery Factor': 'Forward Recovery Factor',
                                          'Sharpe Ratio': 'Forward Sharpe Ratio', 'Custom': 'Forward Custom',
                                          'Equity DD %': 'Forward Equity DD %', 'Trades': 'Forward Trades'},
                                 inplace=True)
                print('Done Forward Results for:', i, j)

                # ----------------------------------------Join DATAFRAMES-----------------------------------------------
                csvFileNameComplete = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal//6C3C6A11D1C3791DD4DBF45421BF8028/' \
                                      '/reports/{}/{}/{}/OptiResults-{}-{}-{}-Phase1.Complete.csv'.format(BotName.get(), i, j,
                                                                                                          BotName.get(), i, j)
                csvFileNameCompleteFiltered = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/' \
                                              '/6C3C6A11D1C3791DD4DBF45421BF8028//reports/{}/{}/{}/OptiResults' \
                                              '-{}-{}-{}-Phase1.Complete-Filtered.csv'.format(BotName.get(), i, j,
                                                                                              BotName.get(), i, j)

                Complete_df = pd.concat([dfback, dfforward], axis=1)
                Complete_df = Complete_df.loc[:, ~Complete_df.columns.duplicated()]
                Complete_df = movecol(Complete_df, cols_to_move=['Forward Result'], ref_col='Result')
                Complete_df = movecol(Complete_df, cols_to_move=['Forward Profit'], ref_col='Profit')
                Complete_df = movecol(Complete_df, cols_to_move=['Forward Expected Payoff'], ref_col='Expected Payoff')
                Complete_df = movecol(Complete_df, cols_to_move=['Forward Profit Factor'], ref_col='Profit Factor')
                Complete_df = movecol(Complete_df, cols_to_move=['Forward Recovery Factor'], ref_col='Recovery Factor')
                Complete_df = movecol(Complete_df, cols_to_move=['Forward Sharpe Ratio'], ref_col='Sharpe Ratio')
                Complete_df = movecol(Complete_df, cols_to_move=['Forward Custom'], ref_col='Custom')
                Complete_df = movecol(Complete_df, cols_to_move=['Forward Average Loss'], ref_col='Average Loss')
                Complete_df = movecol(Complete_df, cols_to_move=['Forward Equity DD %'], ref_col='Equity DD %')
                Complete_df = movecol(Complete_df, cols_to_move=['Forward Absolute DD'], ref_col='Absolute DD')
                Complete_df = movecol(Complete_df, cols_to_move=['Forward Trades'], ref_col='Trades')
                Complete_df = movecol(Complete_df, cols_to_move=['Forward Win Ratio'], ref_col='Win Ratio')
                Complete_df = movecol(Complete_df, cols_to_move=['Forward Lots'], ref_col='Lots')
                Complete_df = Complete_df.drop(['Back Result'], axis=1)
                Complete_df.to_csv(csvFileNameComplete, sep=',', index=False)

                # --------------------------------AFTER UNION FILTER----------------------------------------------------
                print('Before filtering the len of Complete DataFrame for', i, j, 'is:', len(Complete_df))
                for index, row in Complete_df.iterrows():
                    Totalcount += 1
                    AvgLossNorm = 100 / row['Average Loss']
                    AbsoluteDDNorm = float(FilterEquityDDPhase1.get()) / row['Absolute DD']
                    NormalizeRow = float(min(AvgLossNorm, AbsoluteDDNorm))
                    row['Lots'] = float(round(row['Lots'] * NormalizeRow, 2))
                    LotRatio = row['Lots'] / 0.1
                    row['Forward Lots'] = row['Lots']
                    row['Forward Profit'] = row['Forward Profit'] * LotRatio
                    row['Forward Expected Payoff'] = row['Forward Expected Payoff'] * LotRatio
                    row['Forward Absolute DD'] = row['Forward Absolute DD'] * LotRatio
                    row['Forward Average Loss'] = row['Forward Average Loss'] * LotRatio
                    Complete_df.loc[index, 'Forward Profit'] = row['Forward Profit']
                    Complete_df.loc[index, 'Forward Expected Payoff'] = row['Forward Expected Payoff']
                    Complete_df.loc[index, 'Forward Absolute DD'] = row['Forward Absolute DD']
                    Complete_df.loc[index, 'Forward Average Loss'] = row['Forward Average Loss']
                    Complete_df.loc[index, 'Forward Lots'] = row['Forward Lots']

                    try:
                        if (row['Profit'] >= int(FilterNetProfitPhase1.get())
                            and row['Expected Payoff'] >= float(FilterExpectedPayoffPhase1.get())
                            and row['Profit Factor'] >= float(FilterProfitFactorPhase1.get())
                            and row['Custom'] >= float(FilterCustomPhase1.get())
                            and row['Absolute DD'] <= float(FilterEquityDDPhase1.get()) + 100
                            and row['Trades'] >= int(FilterTradesPhase1.get())
                            and row['Forward Profit'] >= int(ForwardFilterNetProfitPhase1.get())
                            and row['Forward Expected Payoff'] >= float(ForwardFilterExpectedPayoffPhase1.get())
                            and row['Forward Profit Factor'] >= float(ForwardFilterProfitFactorPhase1.get())
                            and row['Forward Custom'] >= float(ForwardFilterCustomPhase1.get())
                            and row['Forward Absolute DD'] <= float(ForwardFilterEquityDDPhase1.get()) + 100
                            and row['Forward Trades'] >= int(ForwardFilterTradesPhase1.get())):
                            Projectcount += 1
                            pass
                        else:
                            Complete_df.drop(labels=index, inplace=True)
                    except IndexError:
                        pass
                print('After filtering the len of Filtered DataFrame from Phase 1 for', i, j, 'is:', len(Complete_df))

                try:
                    Complete_df.drop(columns=['index'], inplace=True)
                except KeyError:
                    pass
                Complete_df.to_csv(csvFileNameCompleteFiltered, sep=',', index=False)
                print("Filtered Dataframe saved")
            except FileNotFoundError:
                pass
    TimeStartResultsPhase2end = time.time()
    TimeResult = (TimeStartResultsPhase2end - TimeStartResultsPhase2start) / 60
    ProjecttoTotalRatio = (Projectcount / Totalcount) * 100
    print('Filtered by:', '\n', \
          'Forward Min. Net Profit:', ForwardFilterNetProfitPhase1.get(), '\n', \
          'Forward Min. Exp. Payoff:', ForwardFilterExpectedPayoffPhase1.get(), '\n', \
          'Forward Min. Profit Factor:', ForwardFilterProfitFactorPhase1.get(), '\n', \
          'Forward Min. Custom:', ForwardFilterCustomPhase1.get(), '\n', \
          'Forward Max. Equity DD:', ForwardFilterEquityDDPhase1.get(), '\n', \
          'Forward Min. Trades:', ForwardFilterTradesPhase1.get(), '\n')
    print('From a Total of :', Totalcount, 'backtests')
    print('Only', Projectcount, ' passed the filters.', round(ProjecttoTotalRatio, ndigits=2), '%')
    print('Phase 1 Results Accotated in', round(TimeResult), 'minutes')

#-----------------------------------------PROCESS TO OBTAIN OPTISET VALUES FOR PHASE 2----------------------------------
def AccotateOptisetsPhase1():
    """Generates the Optiset for the Results that passed the previous filter"""
    FullStart = time.time()
    try:
        for i in PairList:
            for j in TimeFrameList:
                with open(FOLDER_PATH +'\MQL5\Profiles\Tester\Phase1-{}.set'.format(BotName.get()), 'r', encoding='utf-16') as f:
                    with open(FOLDER_PATH +'\MQL5\Profiles\Tester\{}\Phase2-{}-{}-{}.set'.format(BotName.get(), BotName.get(), i, j), 'w', encoding='utf-16') as f1:
                        for line in f:
                            try:
                                first_letter = line[0]
                                if first_letter == ';' or first_letter == '\n' or first_letter == '':
                                    f1.write(line)
                                else:
                                    try:
                                        dfOpti = pd.read_csv('C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/{}/{}/{}/OptiResults-{}-{}-{}-Phase1-Filtered.csv'.format(BotName.get(), i, j, BotName.get(), i, j))
                                        if (len(dfOpti)) < 1:
                                            continue
                                        try:
                                            OptiColumnName = line.split('=')
                                            OptiColumnMax = dfOpti['{}'.format(OptiColumnName[0])].max()
                                            OptiColumnMin = dfOpti['{}'.format(OptiColumnName[0])].min()
                                            if  OptiColumnMax == OptiColumnMin:
                                                if OptiColumnMax <= 1:
                                                    OptiColumnMin = 0
                                                    OptiColumnMax = 1
                                                elif OptiColumnMax >= 2:
                                                    OptiColumnMin -= 1
                                                    OptiColumnMax += 1
                                            if type(OptiColumnMax) == str:
                                                pass
                                            OptiColumnBaseValue = OptiColumnName[1].split('|')
                                            OptiColumnSteps = OptiColumnBaseValue[4]
                                            OptiFormat = "{}={}||{}||{}||{}||Y \n".format(OptiColumnName[0], OptiColumnBaseValue[0], OptiColumnMin, OptiColumnSteps, OptiColumnMax)
                                            if  OptiColumnName[0] == 'Lots':
                                                OptiFormat = "{}={}||{}||{}||{}||N \n".format(OptiColumnName[0], OptiColumnBaseValue[0], OptiColumnMin, OptiColumnSteps, OptiColumnMax)
                                                pass
                                            f1.write('\n'+ OptiFormat)
                                        except KeyError:
                                            f1.write(line)
                                            pass
                                    except FileNotFoundError:
                                        pass
                            except IndexError:
                                pass
            print('This Pair', i, ' and TimeFrame ', j, ' was not Launched')
        print('All Pairs and TimeFrames Optisets Created')
    except FileNotFoundError:
        pass
        print('This Pair and Timeframe has no File', i, j)
    FullEnd = time.time()
    print('Phase 1 Optisets Accotated in :', (FullEnd-FullStart), ' seconds.')

#----------------------------CREATES INIT FILES FOR PHASE 2-----(OPTI ACCOTATED)----------------------------------------
def CreateIniFilesPhase2(BotName="MACD Sample", PairList='EURUSD', TimeFrameList='H1', OptimizationCriterionList=6, ModelList=2, OptimizationList=2, ShutdownTerminalList=1, VisualList=0, LeverageValue=33, ReplaceReportList=1, UseLocalList=1, ForwardModeList=4, ExecutionValue=28, Phase=2):
    """Creates the INIT file specific for a Phase 2 Optimization"""
    f = open(FOLDER_PATH +'/reports/{}/INITS/Phase2/'.format(BotName.get()) + 'INIT-{}-{}-{}-Phase{}.ini'.format(BotName.get(), PairList, TimeFrameList, Phase), "w")
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
    'ExpertParameters=\{}\Phase{}-{}-{}-{}.set'.format(BotName.get(), Phase, BotName.get(), PairList, TimeFrameList) + "\n" \
    'Symbol={}'.format(PairList) + 'MT5' + "\n" \
    'Period={}'.format(TimeFrameList) + "\n" \
    ';Login=XXXXXX' + "\n" \
    'Model={}'.format(ModelList) + "\n" \
    'ExecutionMode={}'.format(str(ExecutionValue)) + "\n" \
    'Optimization={}'.format(OptimizationList) + "\n" \
    'OptimizationCriterion={}'.format(OptimizationCriterionList) + "\n" \
    'FromDate={}'.format(Opti_start_date.get()) + "\n" \
    'ToDate={}'.format(Opti_end_date.get()) + "\n" \
    'ForwardMode={}'.format(ForwardModeList) + "\n" \
    'ForwardDate={}'.format(ForwardDate.get()) + "\n" \
    'Report=reports\{}\{}\{}\OptiResults-{}-{}-{}-Phase{}'.format(BotName.get(), PairList, TimeFrameList, BotName.get(), PairList, TimeFrameList, Phase) + "\n" \
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
    f = open(FOLDER_PATH +'/reports/{}/INITS/Phase2/'.format(BotName.get()) + 'INIT-{}-{}-{}-Phase{}.ini'
             .format(BotName.get(), PairList, TimeFrameList, Phase), "r")

#------------------------------------CREATES INIT FILES FOR ALL PAIRS PHASE 2-------------------------------------------
def CreateIniForAllPhase2():
    """Creates a Phase 2 INIT file for every pair and timeframe selected"""
    inicount = 0
    for i in PairListTest:
        for j in TimeFrameListTest:
            if PairListTest[i].get() == 1:
                if TimeFrameListTest[j].get() == 1:
                    CreateIniFilesPhase2(BotName, PairList=i, TimeFrameList=j)
                    print(i, j, 'INIT Phase 2 Created')
                    inicount += 1
    print("INITS for All Phase 2 Created. Total", inicount, 'INIT files.')

#-----------------------------------LAUNCHES INIT FILES IN CMD FOR PHASE 2----------------------------------------------
def LaunchPhase2():
    """Executes in CMD the INIT file on MT5 for every pair and timeframe selected for Phase 2"""
    FullStart = time.time()
    for file in os.listdir("C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/"
                           "reports/{}/INITS/Phase2".format(BotName.get())):
        start = time.time()
        print(str((MT5_PATH + " /config:" + "{}/".format(FOLDER_PATH) + "reports/{}/INITS/Phase2/{}".
                   format(BotName.get(), file))))
        process = subprocess.call(MT5_PATH + " /config:C:\\Users\\bryan\\AppData\\Roaming\\MetaQuotes\\Terminal\\"
                                             "6C3C6A11D1C3791DD4DBF45421BF8028\\reports\\{}\\INITS\\Phase2\{}".
                                  format(BotName.get(), file))
        end = time.time()
        print('Duration for Phase 2 on was of', (end - start)/60, 'minutes')
    FullEnd = time.time()
    print('Launch from Phase 2 Ended during a total', (FullEnd-FullStart)/60, 'minutes')

#--PROCESS TO OBTAIN OPTISET VALUES FOR PHASE 3 --CREATE CSV ACCOTATTED FROM XML, JOINS BACK AND FORWARD AND FILTERS IT-
def AccotateResultsPhase2():
    """Filtrates the Results for Phase 2 Optimization and keeps the results that passes"""
    print('Begins Phase 2 Results Filtering')
    TimeStartResultsPhase2start = time.time()
    Totalcount = 0
    Projectcount = 0
    def movecol(df, cols_to_move=[], ref_col='', place='After'):

        cols = df.columns.tolist()
        if place == 'After':
            seg1 = cols[:list(cols).index(ref_col) + 1]
            seg2 = cols_to_move
        if place == 'Before':
            seg1 = cols[:list(cols).index(ref_col)]
            seg2 = cols_to_move + [ref_col]

        seg1 = [i for i in seg1 if i not in seg2]
        seg3 = [i for i in cols if i not in seg1 + seg2]

        return df[seg1 + seg2 + seg3]
    for i in PairList:
        for j in TimeFrameList:
            try:
                #---------------------------------------------------------------BACKTEST FILE-------------------------------------------------------------------------------------------------------------------
                Nullvaluesindex = 9 + OptimizedVariables
                NullvaluesColumns = 8 + OptimizedVariables
                csvFileNameBack = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal//6C3C6A11D1C3791DD4DBF45421BF8028//reports/{}/{}/{}/OptiResults-{}-{}-{}-Phase1.csv'.format(BotName.get(), i, j, BotName.get(), i, j)
                treeback = et.parse('C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal//6C3C6A11D1C3791DD4DBF45421BF8028//reports/{}/{}/{}/OptiResults-{}-{}-{}-Phase1.xml'.format(BotName.get(), i, j, BotName.get(), i, j))
                rootback = treeback.getroot()
                csv_list_back = []
                for child in rootback:
                    for Section in child:
                        for Row in Section:
                            Row_list_back = []
                            csv_list_back.append(Row_list_back)
                            for Cell in Row:
                                for Data in Cell:
                                    Row_list_back.append(Data.text)
                dfback = pd.DataFrame(data=csv_list_back)
                dfback = dfback.drop(dfback.index[:Nullvaluesindex]) # 4 variables hacen 13 9
                dfback_columns_name = csv_list_back[NullvaluesColumns] # 4 variables hacen 12
                dfback.columns = dfback_columns_name
                dfback['Lots'] = 0.1
                dfback['Average Loss'] = dfback['Result']
                dfback['Win Ratio'] = dfback['Result']
                dfback['Average Loss'] = dfback['Average Loss'].str.slice(start=-6, stop=-3)
                dfback['Win Ratio'] = dfback['Win Ratio'].str.slice(start=-3)
                dfback = dfback.apply(pd.to_numeric)
                dfback['Absolute DD'] = dfback['Profit'] / dfback['Recovery Factor']
                dfback = movecol(dfback, cols_to_move=['Absolute DD'], ref_col='Equity DD %')
                dfback = movecol(dfback, cols_to_move=['Average Loss'], ref_col='Equity DD %', place='Before')
                dfback = movecol(dfback, cols_to_move=['Win Ratio'], ref_col='Trades')
                dfback = movecol(dfback, cols_to_move=['Lots'], ref_col='Win Ratio')
                dfback = dfback.apply(pd.to_numeric)
                dfback.to_csv(csvFileNameBack, sep=',', index=False)
                dfback.reset_index(inplace=True)
                for index, row in dfback.iterrows():  # NORMALIZATION FIRST
                    try:
                        AvgLossNorm = 100 / row['Average Loss']
                        AbsoluteDDNorm = float(FilterEquityDDPhase1.get()) / row['Absolute DD']
                        NormalizeRow = float(min(AvgLossNorm, AbsoluteDDNorm))
                        row['Lots'] = float(round(row['Lots'] * NormalizeRow, 2))
                        LotRatio = row['Lots'] / 0.1
                        row['Profit'] = row['Profit'] * LotRatio
                        row['Expected Payoff'] = row['Expected Payoff'] * LotRatio
                        row['Absolute DD'] = row['Absolute DD'] * LotRatio
                        row['Average Loss'] = row['Average Loss'] * LotRatio
                        dfback.loc[index, 'Profit'] = row['Profit']
                        dfback.loc[index, 'Expected Payoff'] = row['Expected Payoff']
                        dfback.loc[index, 'Absolute DD'] = row['Absolute DD']
                        dfback.loc[index, 'Average Loss'] = row['Average Loss']
                        dfback.loc[index, 'Lots'] = row['Lots']
                        pass
                    except IndexError:
                        pass
                print('Done Backtest Results for:', i, j)

                #------------------------------------------------------------FORWARD FILE----------------------------------------------------------------------------------------------------------------------------
                csvFileNameForward = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal//6C3C6A11D1C3791DD4DBF45421BF8028//reports/{}/{}/{}/OptiResults-{}-{}-{}-Phase1.forward.csv'.format(BotName.get(), i, j, BotName.get(), i, j)
                treeforward = et.parse('C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal//6C3C6A11D1C3791DD4DBF45421BF8028//reports/{}/{}/{}/OptiResults-{}-{}-{}-Phase1.forward.xml'.format(BotName.get(), i, j, BotName.get(), i, j))
                rootforward = treeforward.getroot()
                csv_list_forward = []
                for child in rootforward:
                    for Section in child:
                        for Row in Section:
                            Row_list_forward = []
                            csv_list_forward.append(Row_list_forward)
                            for Cell in Row:
                                for Data in Cell:
                                    Row_list_forward.append(Data.text)
                dfforward = pd.DataFrame(data=csv_list_forward)
                dfforward = dfforward.drop(dfforward.index[:Nullvaluesindex]) # 4 variables hacen 13 9
                dfforward_columns_name = csv_list_forward[NullvaluesColumns] # 4 variables hacen 12
                dfforward.columns = dfforward_columns_name
                dfforward['Forward Lots'] = 0.1
                dfforward['Forward Average Loss'] = dfforward['Forward Result']
                dfforward['Forward Win Ratio'] = dfforward['Forward Result']
                dfforward['Forward Average Loss'] = dfforward['Forward Average Loss'].str.slice(start=-6, stop=-3)
                dfforward['Forward Win Ratio'] = dfforward['Forward Win Ratio'].str.slice(start=-3)
                dfforward = dfforward.apply(pd.to_numeric)
                dfforward['Forward Absolute DD'] = dfforward['Profit'] / dfforward['Recovery Factor']
                dfforward = movecol(dfforward, cols_to_move=['Forward Absolute DD'], ref_col='Equity DD %')
                dfforward = movecol(dfforward, cols_to_move=['Forward Average Loss'], ref_col='Equity DD %', place='Before')
                dfforward = movecol(dfforward, cols_to_move=['Forward Win Ratio'], ref_col='Trades')
                dfforward = movecol(dfforward, cols_to_move=['Forward Lots'], ref_col='Forward Win Ratio')
                dfforward = dfforward.apply(pd.to_numeric)
                dfforward.sort_values(by=['Back Result'], ascending=False, inplace=True)
                dfforward.reset_index(inplace=True)
                dfforward.to_csv(csvFileNameForward, sep=',', index=False)
                dfforward.rename(columns={'Profit':'Forward Profit', 'Expected Payoff':'Forward Expected Payoff', 'Profit Factor':'Forward Profit Factor', 'Recovery Factor':'Forward Recovery Factor', 'Sharpe Ratio':'Forward Sharpe Ratio', 'Custom':'Forward Custom', 'Equity DD %':'Forward Equity DD %', 'Trades':'Forward Trades'}, inplace=True)
                print('Done Forward Results for:', i, j)

                #---------------------------------------------------------------------------------Join DATAFRAMES---------------------------------------------------------------------------------------
                csvFileNameComplete = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal//6C3C6A11D1C3791DD4DBF45421BF8028//reports/{}/{}/{}/OptiResults-{}-{}-{}-Phase1.Complete.csv'.format(BotName.get(), i, j, BotName.get(), i, j)
                csvFileNameCompleteFiltered = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal//6C3C6A11D1C3791DD4DBF45421BF8028//reports/{}/{}/{}/OptiResults-{}-{}-{}-Phase1.Complete-Filtered.csv'.format(BotName.get(), i, j, BotName.get(), i, j)

                Complete_df = pd.concat([dfback, dfforward], axis=1)
                Complete_df = Complete_df.loc[:, ~Complete_df.columns.duplicated()]
                Complete_df = movecol(Complete_df, cols_to_move=['Forward Result'], ref_col='Result')
                Complete_df = movecol(Complete_df, cols_to_move=['Forward Profit'], ref_col='Profit')
                Complete_df = movecol(Complete_df, cols_to_move=['Forward Expected Payoff'], ref_col='Expected Payoff')
                Complete_df = movecol(Complete_df, cols_to_move=['Forward Profit Factor'], ref_col='Profit Factor')
                Complete_df = movecol(Complete_df, cols_to_move=['Forward Recovery Factor'], ref_col='Recovery Factor')
                Complete_df = movecol(Complete_df, cols_to_move=['Forward Sharpe Ratio'], ref_col='Sharpe Ratio')
                Complete_df = movecol(Complete_df, cols_to_move=['Forward Custom'], ref_col='Custom')
                Complete_df = movecol(Complete_df, cols_to_move=['Forward Average Loss'], ref_col='Average Loss')
                Complete_df = movecol(Complete_df, cols_to_move=['Forward Equity DD %'], ref_col='Equity DD %')
                Complete_df = movecol(Complete_df, cols_to_move=['Forward Absolute DD'], ref_col='Absolute DD')
                Complete_df = movecol(Complete_df, cols_to_move=['Forward Trades'], ref_col='Trades')
                Complete_df = movecol(Complete_df, cols_to_move=['Forward Win Ratio'], ref_col='Win Ratio')
                Complete_df = movecol(Complete_df, cols_to_move=['Forward Lots'], ref_col='Lots')
                Complete_df = Complete_df.drop(['Back Result'], axis=1)
                Complete_df.to_csv(csvFileNameComplete, sep=',', index=False)

                #---------------------------------------------------------------------------------AFTER UNION FILTER---------------------------------------------------------------------------------------
                print('Before filtering the len of Complete DataFrame for', i, j, 'is:', len(Complete_df))
                for index, row in Complete_df.iterrows():
                    Totalcount += 1
                    AvgLossNorm = 100 / row['Average Loss']
                    AbsoluteDDNorm = float(FilterEquityDDPhase1.get()) / row['Absolute DD']
                    NormalizeRow = float(min(AvgLossNorm, AbsoluteDDNorm))
                    row['Lots'] = float(round(row['Lots'] * NormalizeRow, 2))
                    LotRatio = row['Lots'] / 0.1
                    row['Forward Lots'] = row['Lots']
                    row['Forward Profit'] = row['Forward Profit'] * LotRatio
                    row['Forward Expected Payoff'] = row['Forward Expected Payoff'] * LotRatio
                    row['Forward Absolute DD'] = row['Forward Absolute DD'] * LotRatio
                    row['Forward Average Loss'] = row['Forward Average Loss'] * LotRatio
                    Complete_df.loc[index, 'Forward Profit'] = row['Forward Profit']
                    Complete_df.loc[index, 'Forward Expected Payoff'] = row['Forward Expected Payoff']
                    Complete_df.loc[index, 'Forward Absolute DD'] = row['Forward Absolute DD']
                    Complete_df.loc[index, 'Forward Average Loss'] = row['Forward Average Loss']
                    Complete_df.loc[index, 'Forward Lots'] = row['Forward Lots']

                    #print('Forward Profit Normalized:', row['Forward Profit'], 'with ',row['Forward Lots'])
                    #print(row['Profit'], row['Expected Payoff'], row['Profit Factor'],row['Absolute DD'], row['Trades'])
                    #print(row['Forward Profit'],row['Forward Expected Payoff'],row['Forward Profit Factor'],row['Forward Absolute DD'],row['Forward Trades'])
                    try:
                        if (row['Profit'] >= int(FilterNetProfitPhase1.get())) and (row['Expected Payoff'] >= float(FilterExpectedPayoffPhase1.get())) and (row['Profit Factor'] >= float(FilterProfitFactorPhase1.get())) and (row['Custom'] >= float(FilterCustomPhase1.get())) and ((row['Absolute DD']) <= float(FilterEquityDDPhase1.get())+100) and (row['Trades'] >= int(FilterTradesPhase1.get())) and \
                            (row['Forward Profit'] >= int(ForwardFilterNetProfitPhase1.get())) and (row['Forward Expected Payoff'] >= float(ForwardFilterExpectedPayoffPhase1.get())) and (row['Forward Profit Factor'] >= float(ForwardFilterProfitFactorPhase1.get())) and (row['Forward Custom'] >= float(ForwardFilterCustomPhase1.get())) and ((row['Forward Absolute DD']) <= float(ForwardFilterEquityDDPhase1.get())+100) and (row['Forward Trades'] >= int(ForwardFilterTradesPhase1.get())):
                            Projectcount += 1
                            pass
                        else:
                            Complete_df.drop(labels=index, inplace=True)

                    except IndexError:
                        pass

                print('After filtering the len of Filtered DataFrame from Phase 2 for', i, j, 'is:', len(Complete_df))
                #Complete_df = Complete_df.drop_duplicates(subset='Profit',inplace=True) drop duplicates attempt
                print('After Removing Duplicates for', i, j, ' the len is: unknown')
                try:
                    Complete_df.drop(columns=['index'], inplace=True)
                except KeyError:
                    pass
                Complete_df.to_csv(csvFileNameCompleteFiltered, sep=',', index=False)
                print("Filtered Dataframe saved")
            except FileNotFoundError:
                pass
    TimeStartResultsPhase2end = time.time()
    TimeResult = (TimeStartResultsPhase2end-TimeStartResultsPhase2start)/60
    ProjecttoTotalRatio = (Projectcount/Totalcount)*100
    print('Filtered by:', '\n', \
          'Forward Min. Net Profit:', ForwardFilterNetProfitPhase1.get(), '\n', \
          'Forward Min. Exp. Payoff:', ForwardFilterExpectedPayoffPhase1.get(), '\n', \
          'Forward Min. Profit Factor:', ForwardFilterProfitFactorPhase1.get(), '\n', \
          'Forward Min. Custom:', ForwardFilterCustomPhase1.get(), '\n', \
          'Forward Max. Equity DD:', ForwardFilterEquityDDPhase1.get(), '\n', \
          'Forward Min. Trades:', ForwardFilterTradesPhase1.get(), '\n')
    print('From a Total of :', Totalcount, 'backtests')
    print('Only', Projectcount, ' passed the filters.', round(ProjecttoTotalRatio, ndigits=2), '%')
    print('Phase 2 Results Accotated in', round(TimeResult), 'minutes')

#---------------------------------------------------PROCESS TO OBTAIN OPTISET VALUES FOR PHASE 3---(ONLY OPTISET)---------------------------------
def AccotateOptisetsPhase2(): #Still not decided if this STEP SHOULD BE INCLUDED IN THE PROCESS
    try:
        for i in PairList:
            for j in TimeFrameList:
                with open(FOLDER_PATH +'\MQL5\Profiles\Tester\Phase1-{}.set'.format(BotName.get()), 'r', encoding='utf-16') as f:
                    with open(FOLDER_PATH +'\MQL5\Profiles\Tester\{}\Phase3-{}-{}-{}.set'.format(BotName.get(), BotName.get(), i, j), 'w', encoding='utf-16') as f1:
                        for line in f:
                            try:
                                first_letter = line[0]
                                if first_letter == ';' or first_letter == '\n' or first_letter == '':
                                    f1.write(line)
                                else:
                                    try:
                                        dfOpti = pd.read_csv('C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/{}/{}/{}/OptiResults-{}-{}-{}-Phase2.Complete-Filtered.csv'.format(BotName.get(), i, j, BotName.get(), i, j))
                                        if (len(dfOpti)) < 1:
                                            continue
                                            try:
                                                OptiColumnName = line.split('=')
                                                OptiColumnMax = dfOpti['{}'.format(OptiColumnName[0])].max()
                                                OptiColumnMin = dfOpti['{}'.format(OptiColumnName[0])].min()
                                                Min_Steps = 50
                                                if  OptiColumnMax == OptiColumnMin:
                                                    if OptiColumnMax <= 1:
                                                        OptiColumnMin = 0
                                                        OptiColumnSteps = 1
                                                        OptiColumnMax = 1
                                                    elif OptiColumnMax >= 2:
                                                        OptiColumnMin -= 1
                                                        OptiColumnSteps = 1
                                                        OptiColumnMax += 1
                                                elif (OptiColumnMax - OptiColumnMin) < 30:
                                                    OptiColumnSteps = 1
                                                else:
                                                    OptiColumnSteps = (dfOpti['{}'.format(OptiColumnName[0])].max() - dfOpti['{}'.format(OptiColumnName[0])].min())/Min_Steps
                                                if type(OptiColumnMax) == str:
                                                    pass
                                                OptiColumnBaseValue = OptiColumnName[1].split('|')
                                                OptiFormat = "{}={}||{}||{}||{}||Y \n".format(OptiColumnName[0],OptiColumnBaseValue[0],OptiColumnMin,OptiColumnSteps,OptiColumnMax)
                                                f1.write('\n'+ OptiFormat)
                                            except KeyError:
                                                f1.write(line)
                                                pass
                                    except FileNotFoundError:
                                        pass
                            except IndexError:
                                pass
                    print('This Pair', i, ' and TimeFrame ', j, ' did not Pass Phase 2')

        print('All Pairs and TimeFrames Optisets Created')
    except FileNotFoundError:
        pass
        print('This Pair and Timeframe has no File', i, j)


# =============================================================================================== HILL CLIMBING INIT FILE ==========================================================================================
def HCInit(pair='EURUSD', time_frame='H1', optimization_criterion=0, model=2, optimization=1, shutdown=1, visual=0, leverage=33, replace_report=1, use_local=1, forward_mode=0, execution_mode=28):
    f = open(FOLDER_PATH +'/reports/{}/INITS/HC/'.format(BotName.get()) + 'INIT-HC-{}-{}-{}-Phase1.ini'.format(BotName.get(), pair, time_frame), 'w')
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
    'ExecutionMode={}'.format(str(execution_mode)) + "\n" \
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
    # ---------------------------------------------------PROCESS TO OBTAIN OPTISET VALUES FOR HILL CLIMBING PROCESS---------------------------------


def HillClimbPhase2():  # Hill Climbing Step
    # user must define the number of laps (loops) that will go through the Optiset EG: if it has 7 Optimizable variables then 2 laps should be 14 separate Opti
    # needs to read previous Phase 2 Results UNFILTERED     EG:    OptiResults-EA-S1v2-EURCHF-H1-Phase2.Complete
    # needs to read the number of rows that finish in "Y" in the Optiset    EG:     Phase1-EA-S1v2
    # start a loop that: 1.- Creates a INIT optimizing only "H" variable using a previously created HOptiset.
    #                    2.- Save that result with a OptiResults-EA-S1v2-EURCHF-H1-Phase2.Complete     H name avoid rewriting, only annexes values
    #                    3.- Set Highest value from Opti as Base Value for next Opti       BaseValue |  xmin  | xstep  | xmax     Y
    HCCount = 0
    HCLaps = 3
    HCOptimizable = []
    FullStart = time.time()
    # Creates a Copy of the Optiset so that we can work with a HC Optiset and set all previous optimizable variables to "N"
    OriginalSet = FOLDER_PATH + '\MQL5\Profiles\Tester\Phase1-{}.set'.format(BotName.get())
    HCBaseOptiset = FOLDER_PATH + '\MQL5\Profiles\Tester\HCB-Phase1-{}.set'.format(BotName.get())
    HCOptiset = FOLDER_PATH + '\MQL5\Profiles\Tester\HC-Phase1-{}.set'.format(BotName.get())

    with open(OriginalSet, 'r', encoding='utf-16') as f:
        lines = f.readlines()
        optimizablevars = []
        with open(HCBaseOptiset, "w") as f1:
            for line in lines:
                if (line[0] != ';' or line[0] == ' ') and line[-2] == 'Y':
                    varname = line.split('=')
                    optimizablevars.append(varname[0])
                    varline = varname[1].split('||')
                    varline[-1] = varline[-1].replace("Y", "N")
                    line = '{}={}||{}||{}||{}||N\n'.format(varname[0], varline[0], varline[1], varline[2], varline[3])
                    f1.writelines(line)
                else:
                    pass
            print('These variables are going to be optimized', optimizablevars)
            print('HCBaseOptiset for {} Created'.format(BotName.get()))

        f1.close()

    while HCCount < HCLaps:
        with open(HCBaseOptiset, 'r') as f2:
            HCtext = f2.readlines()
            f2.close()
            print('top', HCtext)
        """with open(HCBaseOptiset, 'w+') as f2:
            f2.truncate(0)
            for v in optimizablevars:
                for line in HCtext:
                    lineval = line.split('=')
                    #f2.write(line)
                    print(line,lineval[0])
                    if lineval[0] == v:
                        line = line.replace('N\n','Y\n')
                        f2.write(line)
                        print('Set to Y:',line)
                    elif lineval[0] != v  and line[-2] == 'Y':
                        line = line.replace('Y\n','N\n')
                        f2.write(line)
                        print('Set to N:',line)
                    elif line[-2] == 'N':
                        print('Simple Line',line)
                print('bot',HCtext)
                HCPrint = f2.readlines()
                print(HCPrint)

                print('This is HCBaseOptiset in top line 839:', HCtext)"""

        HCInit()
        print('INIT File Created for {}'.format(v))

        for file in os.listdir("C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/{}/INITS/HC/".format(BotName.get())):
            start = time.time()
            print(str((MT5_PATH + " /config:" + "{}/".format(FOLDER_PATH) + "reports/{}/INITS/HC/{}".format(BotName.get(), file))))
            process = subprocess.call(MT5_PATH + " /config:C:\\Users\\bryan\\AppData\\Roaming\\MetaQuotes\\Terminal\\6C3C6A11D1C3791DD4DBF45421BF8028\\reports\\{}\\INITS\\HC\{}".format(BotName.get(), file))
            end = time.time()
            print('Duration for HC on {} was of'.format(v), (end - start), 'seconds')

            Nullvaluesindex = 9 + OptimizedVariables
            NullvaluesColumns = 8 + OptimizedVariables
            csvFileNameHC = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal//6C3C6A11D1C3791DD4DBF45421BF8028//reports/{}/INITS/HC-Phase1-{}-EURUSD-H1.csv'.format(BotName.get(), BotName.get())
            treeHC = et.parse('C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal//6C3C6A11D1C3791DD4DBF45421BF8028//reports/{}/INITS/HC-Phase1-{}-EURUSD-H1.xml'.format(BotName.get(), BotName.get()))
            rootback = treeHC.getroot()
            csv_list_hc = []
            for child in rootback:
                for Section in child:
                    for Row in Section:
                        Row_list_HC = []
                        csv_list_hc.append(Row_list_HC)
                        for Cell in Row:
                            for Data in Cell:
                                Row_list_HC.append(Data.text)
            dfHC = pd.DataFrame(data=csv_list_hc)
            dfHC = dfHC.drop(dfHC.index[:Nullvaluesindex])
            dfHC_columns_name = csv_list_hc[NullvaluesColumns]
            dfHC.columns = dfHC_columns_name
            dfHC = dfHC.apply(pd.to_numeric)
            dfHC['Absolute DD'] = dfHC['Profit'] / dfHC['Recovery Factor']
            dfHC = dfHC.apply(pd.to_numeric)
            dfHC.to_csv(csvFileNameHC, sep=',', index=False)
            dfHC.reset_index(inplace=True)
            print('analizo resultados para', v)
            vBestResult = dfHC.loc[0, v]
            print('best result is:', vBestResult)

            """ Write Best Result Value from previous loop in Optiset text file"""
            with open(HCOptiset, "r+") as f4:
                HCtext2 = f4.readlines()
                print('this is HC text2 before loop line:', HCtext2)
                for line in HCtext2:

                    if (line[0] != ';' or line[0] == ' ') and line[-2] == 'Y':
                        varname = line.split('=')
                        varline = varname[1].split('||')
                        if varname[0] == v:
                            line = '{}={}||{}||{}||{}||N\n'.format(varname[0], vBestResult, varline[1], varline[2], varline[3])
                            HCtext2.append(line)
                            print('this is a line:', line)
                        else:
                            line = '{}={}||{}||{}||{}||N\n'.format(varname[0], varline[0], varline[1], varline[2], varline[3])
                            HCtext2.append(line)
                            print('this is a line:', line)
                f4.writelines(HCtext2)
                print('this is HCtext2 down: ', HCtext2)
                f4.close()
            print('Finished {} on Lap {}/{}'.format(v, HCCount, HCLaps))
        HCCount += 1
        FullEnd = time.time()
        print('Finished Lap {} of {}'.format(HCCount, HCLaps))


    """# Changes the variable that is going to run to "Y"
    with open(HCOptiset) as f:
        for line in f:
            count = 0
            if count == HCCount:
                print('Here Creates INIT starts based on HC Optiset')
                print('Now It Launches MetaTrader5 based on the INIT')
                print('Now it saves results here')
                print('Open Results and changes the Base Value of this Variable based on best result')
                HCCount += 1
                count += 1"""

#---------------------------------------------------PROCESS TO OBTAIN OPTISET VALUES FOR PHASE 3--(FOR BACKTESTS)--------------------------------
def BTSetsForPhase3():
    BTsetsPhase3timestart = time.time()
    TotalSets = 0
    for i in PairList:
        for j in TimeFrameList:
            try:
                file_a = FOLDER_PATH +'\MQL5\Profiles\Tester\Phase1-{}.set'.format(BotName.get(),i,j)
                file_a_open = open(file_a, 'rb')
                with open(file_a, 'r', encoding='utf-16') as f:
                    dfOpti = pd.read_csv('C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/{}/{}/{}/OptiResults-{}-{}-{}-Phase1.Complete-Filtered.csv'.format(BotName.get(),i,j,BotName.get(),i,j))
                    #print('For',i,j,'Opti lenght is',len(dfOpti))
                    if (len(dfOpti)) < 1:
                        continue
                    Var_Name_list = []
                    for g, row in dfOpti.iterrows():
                        file_b = FOLDER_PATH +'\MQL5\Profiles\Tester\{}\Phase3-{}-{}-{}-{}.set'.format(BotName.get(), BotName.get(), i, j, g)
                        shutil.copyfile(file_a, file_b)
                        file_b_open = open(file_b, 'wb')
                        TotalSets += 1
                        with open (file_b, 'w', encoding='utf-16') as f1:
                            for line in f:
                                if line[0] == ';' or line[0] == '\n':
                                    continue
                                OptiColumnName = line.split('=')
                                Var_Name_list.append(OptiColumnName[0])
                            for x in Var_Name_list:
                                try:
                                    dfOptiValueSpot = dfOpti.iloc[g][x]
                                except KeyError:
                                    continue
                                OptiFormat = "{}={}||1000||1000||2000||N \n".format(x, dfOptiValueSpot)  # a esta altura es donde ocurren los calculos
                                f1.write('{} \n'.format(OptiFormat))
                            MagicStartLine = 'MagicStart={}{}{}{}{}||1000||1000||2000||N \n'.format(BotMagicNumberSeries, UserSeries, PairList[i], TimeFrameList[j], g)
                            f1.write(MagicStartLine)
                        CreateIniFilesPhase3(pair=i, time_frame=j, tail_number=g)
                        print('INIT FILE for :', i, j, g, 'created')
            except FileNotFoundError:
                pass
                print('This Pair and Timeframe has no File', i, j)
            BTsetsPhase3timeend = time.time()
            TimeCalculus = BTsetsPhase3timeend-BTsetsPhase3timestart
    print('A Total of ', TotalSets, 'were created.')
    print('Done All Pairs and TimeFrames BT Sets Created in:', round(TimeCalculus, ndigits=2), 'seconds')
#----------------------------------------------------CREATES INIT FILES FOR PHASE 3----(SET PRODUCTION)-------------------------------------------
def CreateIniFilesPhase3(pair='EURUSD', time_frame='H1', optimization_criterion=0, model=2, optimization=0, shutdown=1, visual=0, leverage=33, replace_report=1, use_local=1, forward_mode=0, execution_mode=28, phase=3, tail_number=0):

    f = open(FOLDER_PATH +'/reports/{}/INITS/Phase3/'.format(BotName.get()) + 'INIT-BT-{}-{}-{}-Phase{}-{}.ini'.format(BotName.get(), pair, time_frame, phase, tail_number), "w")
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
    'ExpertParameters=\{}\Phase3-{}-{}-{}-{}.set'.format(BotName.get(), BotName.get(), pair, time_frame, tail_number) + "\n" \
    'Symbol={}'.format(pair) + 'MT5' + "\n" \
    'Period={}'.format(time_frame) + "\n" \
    ';Login=XXXXXX' + "\n" \
    'Model={}'.format(model) + "\n" \
    'ExecutionMode={}'.format(str(execution_mode)) + "\n" \
    'Optimization={}'.format(optimization) + "\n" \
    'OptimizationCriterion={}'.format(optimization_criterion) + "\n" \
    'FromDate={}'.format(Opti_start_date.get()) + "\n" \
    'ToDate={}'.format(Opti_end_date.get()) + "\n" \
    ';ForwardMode={}'.format(forward_mode) + "\n" \
    ';ForwardDate={}'.format(ForwardDate.get()) + "\n" \
    'Report=reports\{}\SETS\Phase3-{}-{}-{}-{}'.format(BotName.get(), BotName.get(), pair, time_frame, tail_number) + "\n" \
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
    f = open(FOLDER_PATH +'/reports/{}/INITS/Phase3/'.format(BotName.get()) + 'INIT-BT-{}-{}-{}-Phase{}-{}.ini'.format(BotName.get(), pair, time_frame, phase, tail_number), "r")

#----------------------------------------------------LAUNCHES INIT FILES IN CMD FOR PHASE 3------SET GENERATION---------------------------
def LaunchPhase3():
    FullStart = time.time()
    TotalSets = 0
    for file in os.listdir('C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/{}/INITS/Phase3/'.format(BotName.get())):
        start = time.time()
        print(str((MT5_PATH + " /config:" + "{}/".format(FOLDER_PATH) + "reports/{}/INITS/Phase3/{}".format(BotName.get(), file))))
        process = subprocess.call(MT5_PATH + " /config:C:\\Users\\bryan\\AppData\\Roaming\\MetaQuotes\\Terminal\\6C3C6A11D1C3791DD4DBF45421BF8028\\reports\\{}\\INITS\\Phase3\{}".format(BotName.get(), file))
        end = time.time()
        print('Duration for Phase 3 on one Pair and TF was of', end - start, 'seconds.')
        TotalSets += 1
    FullEnd = time.time()
    print('A Total of ', TotalSets, 'were created.')
    print('Launch from Phase 3 Ended during a total', (FullEnd-FullStart)/60, 'minutes.')



#-----------------------------------------------------COMIENZA INTERFACE--------------------------------------
class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.hi_there = tk.Button(self)
        self.hi_there["text"] = "Hello World\n(click me)"
        self.hi_there["command"] = self.say_hi
        self.hi_there.pack(side="top")


        self.quit = tk.Button(self, text="QUIT", fg="red", command=self.master.destroy)
        self.quit.pack(side="bottom")

    def say_hi(self):
        print(BotName.get())


root = tk.Tk() # raiz es el cuadro mas grande
root.title("Optimizer Ecoelite")
root.config(bg='gray')
myFrame = Frame(root)


myFrame.pack(fill='both', expand=True) # para colocarlo en algun borde consultar Packer options
myFrame.config(bg='#292524', width=1200, height=1000)
Label(myFrame, text='Lanzamiento', fg='white', bg='#292524', font=(18)).grid(row=0, column=0, columnspan=2) #Label Launch Panel

Label(myFrame, text="EA Name:", fg='white', bg='#292524').grid(row=1, column=0, sticky='w', padx=10, pady=10) #Label BotName
BotName = Entry(myFrame, fg='white', bg='#151312', width=12)
BotName.grid(row=1, column=1, sticky='w', padx=10, pady=10)                   #Entry BotName
BotName.get()

Label(myFrame, text="Start Date:", fg='white', bg='#292524').grid(row=2, column=0, sticky='w', padx=10, pady=10) #Label Start Date
Opti_start_date = Entry(myFrame, fg='white', bg='#151312', width=10)
Opti_start_date.grid(row=2, column=1, sticky='w', padx=10, pady=10)                   #Entry Start Date
Opti_start_date.get()

Label(myFrame, text="Forward Date:", fg='white', bg='#292524').grid(row=3, column=0, sticky='w', padx=10, pady=10) #Label Forward Date
ForwardDate = Entry(myFrame, fg='white', bg='#151312', width=10)
ForwardDate.grid(row=3, column=1, sticky='w', padx=10, pady=10)                   #Entry Forward Date
ForwardDate.get()

Label(myFrame, text="End Date:", fg='white', bg='#292524').grid(row=4, column=0, sticky='w', padx=10, pady=10) #Label End Date
Opti_end_date = Entry(myFrame, fg='white', bg='#151312', width=10)
Opti_end_date.grid(row=4, column=1, sticky='w', padx=10, pady=10)                   #Entry End Date
Opti_end_date.get()

def IteratetionsFunction():
    Opticount = 0
    for i in PairListTest:
        for j in TimeFrameListTest:
            if PairListTest[i].get() == 1:
                if TimeFrameListTest[j].get() == 1:
                    Opticount += 1
                    print(i, j)
    print('EA to analize:', BotName.get(), 'Total:', Opticount, 'Optimizations')


Button(myFrame, text="Create ALL Folders", fg='black', bg='#34D7DF', borderwidth=0, command=CreateALLFoldersPhase1).grid(row=2, column=2, padx=10, pady=10)

Button(myFrame, text="Create Ini Files For Phase 1", fg='black', bg='#34D7DF', borderwidth=0, command=CreateIniForAllPhase1).grid(row=4, column=2, padx=10, pady=10)

Button(myFrame, text='LAUNCH Phase 1', fg='black', bg='#E74C3C', borderwidth=0, command=LaunchPhase1).grid(row=5, column=2, padx=10, pady=10)

Button(myFrame, text="Accotate Results from Phase 1", fg='black', bg='#34D7DF', borderwidth=0, command=AccotateResultsPhase1).grid(row=6, column=2, padx=10, pady=10)

Button(myFrame, text="Accotate Optisets for Phase 2", fg='black', bg='#34D7DF', borderwidth=0, command=AccotateOptisetsPhase1).grid(row=7, column=2, padx=10, pady=10)

Button(myFrame, text="Create Ini Files for Phase 2", fg='black', bg='#34D7DF', borderwidth=0, command=CreateIniForAllPhase2).grid(row=8, column=2, padx=10, pady=10)

Button(myFrame, text='LAUNCH Phase 2', fg='black', bg='#E74C3C', borderwidth=0, command=LaunchPhase2).grid(row=9, column=2, padx=10, pady=10)

Button(myFrame, text='Join Results and Filter for Phase 3', fg='black', bg='#34D7DF', borderwidth=0, command=AccotateResultsPhase2).grid(row=10, column=2, padx=10, pady=10)

Button(myFrame, text='Accotate Optisets for Phase 3 TO FIX', fg='black', bg='#34D7DF', borderwidth=0, command=AccotateOptisetsPhase2).grid(row=11, column=2, padx=10, pady=10)

Button(myFrame, text='Produce Sets and Inis for Phase 3', fg='black', bg='#34D7DF', borderwidth=0, command=BTSetsForPhase3).grid(row=12, column=2, padx=10, pady=10)

Button(myFrame, text='LAUNCH Phase 3', fg='black', bg='#E74C3C', borderwidth=0, command=LaunchPhase3).grid(row=13, column=2, padx=10, pady=10)

Button(myFrame, text='LAUNCH HC', fg='black', bg='#E74C3C', borderwidth=0, command=HillClimbPhase2).grid(row=14, column=2, padx=10, pady=10)  # HILL CLIMBING TEST BUTTON

Label(myFrame, text="Pairs", fg='white', bg='#292524').grid(row=5, column=0, padx=10, pady=10) #Label Pairs


pairGBPUSD = IntVar()
Checkbutton(myFrame, text="GBPUSD", fg='black', bg='#68E552', variable=pairGBPUSD, onvalue=1, offvalue=0).grid(row=6, column=0) #CheckBox EURUSD

pairEURUSD = IntVar()
Checkbutton(myFrame, text="EURUSD", fg='black', bg='#68E552', variable=pairEURUSD, onvalue=1, offvalue=0).grid(row=7, column=0) #CheckBox EURUSD

pairUSDCAD = IntVar()
Checkbutton(myFrame, text="USDCAD", fg='black', bg='#68E552', variable=pairUSDCAD, onvalue=1, offvalue=0).grid(row=8, column=0) #CheckBox USDCAD

pairUSDCHF = IntVar()
Checkbutton(myFrame, text="USDCHF", fg='black', bg='#68E552', variable=pairUSDCHF, onvalue=1, offvalue=0).grid(row=9, column=0) #CheckBox USDCHF

pairUSDJPY = IntVar()
Checkbutton(myFrame, text="USDJPY", fg='black', bg='#68E552', variable=pairUSDJPY, onvalue=1, offvalue=0).grid(row=10, column=0) #CheckBox USDJPY

pairGBPJPY = IntVar()
Checkbutton(myFrame, text="GBPJPY", fg='black', bg='#68E552', variable=pairGBPJPY, onvalue=1, offvalue=0).grid(row=11, column=0) #CheckBox GBPJPY

pairEURAUD = IntVar()
Checkbutton(myFrame, text="EURAUD", fg='black', bg='#68E552', variable=pairEURAUD, onvalue=1, offvalue=0).grid(row=12, column=0) #CheckBox EURAUD

pairEURGBP = IntVar()
Checkbutton(myFrame, text="EURGBP", fg='black', bg='#68E552', variable=pairEURGBP, onvalue=1, offvalue=0).grid(row=13, column=0) #CheckBox EURGBP

pairEURJPY = IntVar()
Checkbutton(myFrame, text="EURJPY", fg='black', bg='#68E552', variable=pairEURJPY, onvalue=1, offvalue=0).grid(row=14, column=0) #CheckBox EURJPY

pairEURCHF = IntVar()
Checkbutton(myFrame, text="EURCHF", fg='black', bg='#68E552', variable=pairEURCHF, onvalue=1, offvalue=0).grid(row=15, column=0) #CheckBox EURCHF


PairListTest = {'GBPUSD':pairGBPUSD, 'EURUSD':pairEURUSD, 'USDCAD':pairUSDCAD, 'USDCHF':pairUSDCHF, 'USDJPY':pairUSDJPY, 'GBPJPY':pairGBPJPY, 'EURAUD':pairEURAUD, 'EURGBP':pairEURGBP, 'EURJPY':pairEURJPY, 'EURCHF':pairEURCHF}


def checkallpairs():
    pairGBPUSD.set(1), pairEURUSD.set(1), pairUSDCAD.set(1), pairUSDJPY.set(1), pairUSDCHF.set(1), pairGBPJPY.set(1), pairEURAUD.set(1), pairEURGBP.set(1), pairEURJPY.set(1), pairEURCHF.set(1)

Button(myFrame, text='Check all pairs', fg='black', bg='#34D7DF', borderwidth=0, command=checkallpairs).grid(row=18, column=0, padx=10, pady=10)

Label(myFrame, text="Timeframes", fg='white', bg='#292524').grid(row=5, column=1, padx=10, pady=10) #Label TimeFrameList

TFH4 = IntVar()
Checkbutton(myFrame, text="H4", fg='black', bg='#68E552', variable=TFH4).grid(row=6, column=1) #CheckBox H4

TFH1 = IntVar()
Checkbutton(myFrame, text="H1", fg='black', bg='#68E552', variable=TFH1).grid(row=7, column=1) #CheckBox H1

TFM30 = IntVar()
Checkbutton(myFrame, text="M30", fg='black', bg='#68E552', variable=TFM30).grid(row=8, column=1) #CheckBox M30

TFM15 = IntVar()
Checkbutton(myFrame, text="M15", fg='black', bg='#68E552', variable=TFM15).grid(row=9, column=1) #CheckBox M15

TFM5 = IntVar()
Checkbutton(myFrame, text="M5", fg='black', bg='#68E552', variable=TFM5).grid(row=10, column=1) #CheckBox M5

TFM1 = IntVar()
Checkbutton(myFrame, text="M1", fg='black', bg='#68E552', variable=TFM1).grid(row=11, column=1) #CheckBox M1

TimeFrameListTest = {'H4':TFH4, 'H1':TFH1, 'M30':TFM30, 'M15':TFM15, 'M5':TFM5, 'M1':TFM1}

#-----------------------------------------------------------------------PHASE 1 SETTINGS----------------------------------------------------------------------------------------------
Phase1FilterSettingswindow = tk.Toplevel(myFrame, bg='#292524', width=800, height=600)
Phase1FilterSettingswindow.title("Base Filter Settings")
Label(Phase1FilterSettingswindow, text='Base Filter Settings', fg='white', bg='#292524', font=(18)).grid(row=0, column=0, columnspan=2) #Label Launch Panel

Label(Phase1FilterSettingswindow, text="Min. Net Profit:", fg='white', bg='#292524').grid(row=1, column=0, sticky='w', padx=10, pady=10) #Label Net Profit
FilterNetProfitPhase1 = Entry(Phase1FilterSettingswindow, fg='white', bg='#151312', width=12)                   #Entry Net Profit
FilterNetProfitPhase1.grid(row=1, column=1, sticky='w', padx=10, pady=10)
FilterNetProfitPhase1.get()

Label(Phase1FilterSettingswindow, text="Min. Exp. Payoff:", fg='white', bg='#292524').grid(row=2, column=0, sticky='w', padx=10, pady=10) #Label Expected Payoff
FilterExpectedPayoffPhase1 = Entry(Phase1FilterSettingswindow, fg='white', bg='#151312', width=12)                  #Entry Expected Payoff
FilterExpectedPayoffPhase1.grid(row=2, column=1, sticky='w', padx=10, pady=10)
FilterExpectedPayoffPhase1.get()

Label(Phase1FilterSettingswindow, text="Min. Profit Factor:", fg='white', bg='#292524').grid(row=3, column=0, sticky='w', padx=10, pady=10) #Label Min Net Profit
FilterProfitFactorPhase1 = Entry(Phase1FilterSettingswindow, fg='white', bg='#151312', width=12)                    #Entry Profit Factor
FilterProfitFactorPhase1.grid(row=3, column=1, sticky='w', padx=10, pady=10)
FilterProfitFactorPhase1.get()

Label(Phase1FilterSettingswindow, text="Min. Custom Value:", fg='white', bg='#292524').grid(row=4, column=0, sticky='w', padx=10, pady=10) #Label Custom Value
FilterCustomPhase1 = Entry(Phase1FilterSettingswindow, fg='white', bg='#151312', width=12)                  #Entry Custom Value
FilterCustomPhase1.grid(row=4, column=1, sticky='w', padx=10, pady=10)
FilterCustomPhase1.get()

Label(Phase1FilterSettingswindow, text="Max. Drawdown:", fg='white', bg='#292524').grid(row=5, column=0, sticky='w', padx=10, pady=10) #Label Equity DD
FilterEquityDDPhase1 = Entry(Phase1FilterSettingswindow, fg='white', bg='#151312', width=12)                    #Entry Equity DD
FilterEquityDDPhase1.grid(row=5, column=1, sticky='w', padx=10, pady=10)
FilterEquityDDPhase1.get()

Label(Phase1FilterSettingswindow, text="Min. Trades:", fg='white', bg='#292524').grid(row=6, column=0, sticky='w', padx=10, pady=10) #Label Filter Trades
FilterTradesPhase1 = Entry(Phase1FilterSettingswindow, fg='white', bg='#151312', width=12)                  #Entry Filter Trades
FilterTradesPhase1.grid(row=6, column=1, sticky='w', padx=10, pady=10)
FilterTradesPhase1.get()

#--------------------------------------------------------------------------PHASE 2 SETTINGS--------------------------------------------------------------------------------------------
Phase2FilterSettingswindow = tk.Toplevel(myFrame, bg='#292524', width=800, height=600)
Phase2FilterSettingswindow.title("Proyection Filter Settings")
Label(Phase2FilterSettingswindow, text='Proyection Filter Settings', fg='white', bg='#292524', font=(18)).grid(row=0, column=0, columnspan=2) #Label Launch Panel

Label(Phase2FilterSettingswindow, text="Forward Min. Net Profit:", fg='white', bg='#292524').grid(row=1, column=0, sticky='w', padx=10, pady=10) #Label Net Profit
ForwardFilterNetProfitPhase1 = Entry(Phase2FilterSettingswindow, fg='white', bg='#151312', width=12)                   #Entry Net Profit
ForwardFilterNetProfitPhase1.grid(row=1, column=1, sticky='w', padx=10, pady=10)
ForwardFilterNetProfitPhase1.get()

Label(Phase2FilterSettingswindow, text="Forward Min. Exp. Payoff:", fg='white', bg='#292524').grid(row=2, column=0, sticky='w', padx=10, pady=10) #Label Expected Payoff
ForwardFilterExpectedPayoffPhase1 = Entry(Phase2FilterSettingswindow, fg='white', bg='#151312', width=12)                  #Entry Expected Payoff
ForwardFilterExpectedPayoffPhase1.grid(row=2, column=1, sticky='w', padx=10, pady=10)
ForwardFilterExpectedPayoffPhase1.get()

Label(Phase2FilterSettingswindow, text="Forward Min. Profit Factor:", fg='white', bg='#292524').grid(row=3, column=0, sticky='w', padx=10, pady=10) #Label Min Net Profit
ForwardFilterProfitFactorPhase1 = Entry(Phase2FilterSettingswindow, fg='white', bg='#151312', width=12)                    #Entry Profit Factor
ForwardFilterProfitFactorPhase1.grid(row=3, column=1, sticky='w', padx=10, pady=10)
ForwardFilterProfitFactorPhase1.get()

Label(Phase2FilterSettingswindow, text="Forward Min. Custom Value:", fg='white', bg='#292524').grid(row=4, column=0, sticky='w', padx=10, pady=10) #Label Custom Value
ForwardFilterCustomPhase1 = Entry(Phase2FilterSettingswindow, fg='white', bg='#151312', width=12)                  #Entry Custom Value
ForwardFilterCustomPhase1.grid(row=4, column=1, sticky='w', padx=10, pady=10)
ForwardFilterCustomPhase1.get()

Label(Phase2FilterSettingswindow, text="Forward Max. Drawdown:", fg='white', bg='#292524').grid(row=5, column=0, sticky='w', padx=10, pady=10) #Label Equity DD
ForwardFilterEquityDDPhase1 = Entry(Phase2FilterSettingswindow, fg='white', bg='#151312', width=12)                    #Entry Equity DD
ForwardFilterEquityDDPhase1.grid(row=5, column=1, sticky='w', padx=10, pady=10)
ForwardFilterEquityDDPhase1.get()

Label(Phase2FilterSettingswindow, text="Forward Min. Trades:", fg='white', bg='#292524').grid(row=6, column=0, sticky='w', padx=10, pady=10) #Label Filter Trades
ForwardFilterTradesPhase1 = Entry(Phase2FilterSettingswindow, fg='white', bg='#151312', width=12)                  #Entry Filter Trades
ForwardFilterTradesPhase1.grid(row=6, column=1, sticky='w', padx=10, pady=10)
ForwardFilterTradesPhase1.get()


def GraphBacktestPhase1():
    for i in PairListTest:
        for j in TimeFrameListTest:
            if PairListTest[i].get() == 1:
                if TimeFrameListTest[j].get() == 1:
                    df = pd.read_csv('C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/{}/{}/{}/OptiResults-{}-{}-{}-Phase1.csv'.format(BotName.get(), i, j, BotName.get(), i, j))
                    print('Before')
                    print(i, j)
                    print(df)
                    BKBU.Interactive_Graph(df)
                    #BokehINTERACTIVE.Interactive_Graph(BotName.get(),df,i,j)
                    # Execution
                    """def Interactive_Graph(BotName, df, i, j):
                        columns = sorted(df.columns)
                        discrete = [x for x in columns if df[x].dtype == object]
                        continuous = [x for x in columns if x not in discrete]
                        SIZES = list(range(6, 28, 3))
                        COLORS = Inferno256
                        N_SIZES = len(SIZES)
                        N_COLORS = len(COLORS)
                        def create_figure():
                            print('this is Create Figure on Production')
                            xs = df[x.value].values
                            ys = df[y.value].values
                            x_title = x.value.title()
                            y_title = y.value.title()

                            kw = dict()
                            if x.value in discrete:
                                kw['x_range'] = sorted(set(xs))
                            if y.value in discrete:
                                kw['y_range'] = sorted(set(ys))
                            kw['title'] = "%s vs %s" % (x_title, y_title) + " for {} on {} and {}".format(BotName, i, j)

                            p = figure(plot_height=900, plot_width=1700, tools='pan,box_zoom,hover,reset,lasso_select',
                                       **kw)
                            p.xaxis.axis_label = x_title
                            p.yaxis.axis_label = y_title

                            if x.value in discrete:
                                p.xaxis.major_label_orientation = pd.np.pi / 4

                            sz = 9
                            if size.value != 'None':
                                if len(set(df[size.value])) > N_SIZES:
                                    groups = pd.qcut(df[size.value].values, N_SIZES, duplicates='drop')
                                else:
                                    groups = pd.Categorical(df[size.value])
                                sz = [SIZES[xx] for xx in groups.codes]

                            c = "#31AADE"
                            if color.value != 'None':
                                if len(set(df[color.value])) > N_COLORS:
                                    groups = pd.qcut(df[color.value].values, N_COLORS, duplicates='drop')
                                else:
                                    groups = pd.Categorical(df[color.value])
                                c = [COLORS[xx] for xx in groups.codes]

                            Var_color_mapper = LinearColorMapper(palette="Inferno256", low=min(df['Profit']), high=max(
                                df['Profit']))  # arreglar Maximo y minimo para que agarren el valor
                            # Var_color_mapper = LinearColorMapper(palette="Inferno256",low=min(df[color.value]),high=max(df[color.value]))  # arreglar Maximo y minimo para que agarren el valor
                            GraphTicker = AdaptiveTicker(base=50, desired_num_ticks=10, num_minor_ticks=20,
                                                         max_interval=1000)
                            Color_legend = ColorBar(color_mapper=Var_color_mapper, ticker=GraphTicker,
                                                    label_standoff=12, border_line_color=None, location=(
                                0, 0))  # arreglar LogTicker para que muestre por al escala del color
                            p.circle(x=xs, y=ys, color=c, size=sz, line_color="white", alpha=0.6, hover_color='white',
                                     hover_alpha=0.5)
                            p.add_layout(Color_legend, 'right')
                            p.circle(x=xs, y=ys, color=c, size=sz, line_color="white", alpha=0.6, hover_color='white',
                                     hover_alpha=0.5)
                            return p

                        def callback(attr, old, new):
                            layout.children[1] = create_figure()
                            callback = CustomJS(code="console.log('tap event occurred')")

                        source = ColumnDataSource(data=dict(x=df['Pass'], y=df['Profit']))
                        x = Select(title='X-Axis', value='Pass', options=columns)
                        x.on_change('value', callback)

                        y = Select(title='Y-Axis', value='Profit', options=columns)
                        y.on_change('value', callback)

                        size = Select(title='Size', value='None', options=['None'] + continuous)
                        size.on_change('value', callback)

                        color = Select(title='Color', value='None', options=['None'] + continuous)
                        color.on_change('value', callback)

                        controls = column(y, x, color, size, width=200)
                        layout = row(controls, create_figure())

                        curdoc().add_root(layout)
                        # show(layout)
                        output_notebook()
                        curdoc().title = "Phase 1 Graph unfiltered",

                        process = subprocess.call('bokeh serve --show ProductionOpti.py')
                    Interactive_Graph(BotName.get(), df, i, j)
                    # bokeh serve - -show bokehINTERACTIVE.py
                    print('After')"""

Button(myFrame, text='Graph Selected Pair/TF Opti Phase 1', fg='black', bg='#34D7DF', borderwidth=0, command=GraphBacktestPhase1).grid(row=12, column=3, padx=10, pady=10)

def GraphBacktestPhase2():
    for i in PairListTest:
        for j in TimeFrameListTest:
            if PairListTest[i].get() == 1:
                if TimeFrameListTest[j].get() == 1:
                    df = pd.read_csv(
                        'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/{}/{}/{}/OptiResults-{}-{}-{}-Phase2.csv'.format(
                            BotName.get(), i, j, BotName.get(), i, j))

                    columns = sorted(df.columns)
                    discrete = [x for x in columns if df[x].dtype == object]
                    continuous = [x for x in columns if x not in discrete]

                    def create_figure():
                        xs = df[x.value].values
                        ys = df[y.value].values
                        x_title = x.value.title()
                        y_title = y.value.title()

                        kw = dict()
                        if x.value in discrete:
                            kw['x_range'] = sorted(set(xs))
                        if y.value in discrete:
                            kw['y_range'] = sorted(set(ys))
                        kw['title'] = "%s vs %s" % (x_title, y_title) + " for {} on {} and {}".format(BotName, i, j)

                        p = figure(plot_height=600, plot_width=800, tools='pan,box_zoom,hover,reset,lasso_select', **kw)
                        p.xaxis.axis_label = x_title
                        p.yaxis.axis_label = y_title

                        if x.value in discrete:
                            p.xaxis.major_label_orientation = pd.np.pi / 4

                        sz = 9
                        if size.value != 'None':
                            if len(set(df[size.value])) > N_SIZES:
                                groups = pd.qcut(df[size.value].values, N_SIZES, duplicates='drop')
                            else:
                                groups = pd.Categorical(df[size.value])
                            sz = [SIZES[xx] for xx in groups.codes]

                        c = "#31AADE"
                        if color.value != 'None':
                            if len(set(df[color.value])) > N_COLORS:
                                groups = pd.qcut(df[color.value].values, N_COLORS, duplicates='drop')
                            else:
                                groups = pd.Categorical(df[color.value])
                            c = [COLORS[xx] for xx in groups.codes]

                        p.circle(x=xs, y=ys, color=c, size=sz, line_color="white", alpha=0.6, hover_color='white',
                                 hover_alpha=0.5)
                        return p

                    def callback(attr, old, new):
                        layout.children[1] = create_figure()
                        callback = CustomJS(code="console.log('tap event occurred')")

                    # source = ColumnDataSource(data=dict(x=df['Pass'], y=df['Profit']))
                    x = Select(title='X-Axis', value='Pass', options=columns)
                    x.on_change('value', callback)

                    y = Select(title='Y-Axis', value='Profit', options=columns)
                    y.on_change('value', callback)

                    size = Select(title='Size', value='None', options=['None'] + continuous)
                    size.on_change('value', callback)

                    color = Select(title='Color', value='None', options=['None'] + continuous)
                    color.on_change('value', callback)

                    controls = column(y, x, color, size, width=200)
                    layout = row(controls, create_figure())

                    curdoc().add_root(layout)
                    # show(layout)
                    output_notebook()
                    curdoc().title = "Phase 1 Graph unfiltered for {} on {} at {}".format(BotName.get(), i, j)

                    process = subprocess.call('bokeh serve --show ProductionOpti.py')
                    # Posteriormente agregar lineas del CSV en vivo con sliders tal como en https://demo.bokeh.org/export_csv
                    # bokeh serve - -show ProductionOpti.py

Button(myFrame, text='Graph Selected Pair/TF Opti Phase 2', fg='black', bg='#34D7DF', borderwidth=0, command=GraphBacktestPhase2).grid(row=11, column=3, padx=10, pady=10)
def checkalltfs():
    TFH4.set(1), TFH1.set(1), TFM30.set(1), TFM15.set(1), TFM5.set(1), TFM1.set(1)

Button(myFrame, text='Check all timeframes', fg='black', bg='#34D7DF', borderwidth=0, command=checkalltfs).grid(row=18, column=1, padx=10, pady=10)


IteratetionsFunction = Button(myFrame, text='Iterate Selections', fg='black', bg='#34D7DF', borderwidth=0, command=IteratetionsFunction)
IteratetionsFunction.grid(row=21, column=0, padx=10, pady=10, columnspan=2)

def printfilters():
    print(type(FilterNetProfitPhase1), type(FilterExpectedPayoffPhase1), type(FilterProfitFactorPhase1), type(FilterCustomPhase1), type(FilterEquityDDPhase1), type(FilterTradesPhase1))
    print('Filtered by:', '\n', \
          'Min. Net Profit:', FilterNetProfitPhase1.get(), '\n', \
          'Min. Exp. Payoff:', FilterExpectedPayoffPhase1.get(), '\n', \
          'Min. Profit Factor:', FilterProfitFactorPhase1.get(), '\n', \
          'Min. Custom:', FilterCustomPhase1.get(), '\n', \
          'Max. Equity DD:', FilterEquityDDPhase1.get(), '\n', \
          'Min. Trades:', FilterTradesPhase1.get(), '\n')
    print(FilterNetProfitPhase1.get(), FilterExpectedPayoffPhase1.get(), FilterProfitFactorPhase1.get(), FilterCustomPhase1.get(), FilterEquityDDPhase1.get(), FilterTradesPhase1.get())

def printfilters2():
    print(type(ForwardFilterNetProfitPhase1), type(ForwardFilterExpectedPayoffPhase1), type(ForwardFilterProfitFactorPhase1), type(ForwardFilterCustomPhase1), type(ForwardFilterEquityDDPhase1), type(ForwardFilterTradesPhase1))
    print('Filtered by:', '\n', \
          'Forward Min. Net Profit:', ForwardFilterNetProfitPhase1.get(), '\n', \
          'Forward Min. Exp. Payoff:', ForwardFilterExpectedPayoffPhase1.get(), '\n', \
          'Forward Min. Profit Factor:', ForwardFilterProfitFactorPhase1.get(), '\n', \
          'Forward Min. Custom:', ForwardFilterCustomPhase1.get(), '\n', \
          'Forward Max. Equity DD:', ForwardFilterEquityDDPhase1.get(), '\n', \
          'Forward Min. Trades:', ForwardFilterTradesPhase1.get(), '\n')
    print(ForwardFilterNetProfitPhase1.get(), ForwardFilterExpectedPayoffPhase1.get(), ForwardFilterProfitFactorPhase1.get(), ForwardFilterCustomPhase1.get(), ForwardFilterEquityDDPhase1.get(), ForwardFilterTradesPhase1.get())

FiltersTestFunction = Button(myFrame, text='Test Filters', fg='black', bg='#34D7DF', borderwidth=0, command=printfilters)
FiltersTestFunction.grid(row=13, column=3, padx=10, pady=10, columnspan=2)

FiltersTestFunction = Button(Phase1FilterSettingswindow, text='Test Filters', fg='black', bg='#34D7DF', borderwidth=0, command=printfilters)
FiltersTestFunction.grid(row=13, column=3, padx=10, pady=10, columnspan=2)

FiltersTestFunction = Button(Phase2FilterSettingswindow, text='Test Filters', fg='black', bg='#34D7DF', borderwidth=0, command=printfilters2)
FiltersTestFunction.grid(row=13, column=3, padx=10, pady=10, columnspan=2)
#myFrame.config(cursor='hand2') transforma el cursor en una mano
app = Application(master=root) # frame dentro de la raiz
app.mainloop()


#-----------------------------------------------------TERMINA INTERFACE--------------------------------------
