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

from Dto import Dto
from services.create_folders import CreateFolders
from services.accotate_results_phase1 import AccotateResultsPhase1
from services.create_ini_phase import CreateIniPhase
from services.launch_phase import LaunchPhase
from services.accotate_optisets_phase1 import AccotateOptisetsPhase1
from services.accotate_optisets_phase2 import AccotateOptisetsPhase2
from services.hill_climb_phase2 import HillClimbPhase2
from services.bt_sets_for_phase3 import BTSetsForPhase3
from services.accotate_results_phase2 import AccotateResultsPhase2


# CONSTANTS
BOT_NAME = 'EA-B1v1'

OPTI_START_DATE = "2007.01.01" #YYYY.MM.DD
OPTI_END_DATE = "2020.01.01" #YYYY.MM.DD
FORWARD_DATE = "2019.01.01" #YYYY.MM.DD Only if ForwardModeList = 4


FILTER_NET_PROFIT_PHASE1 = 7000
FILTER_EXPECTED_PAYOFF_PHASE1 = 8
FILTER_PROFIT_FACTOR_PHASE1 = 1.29
FILTER_CUSTOM_PHASE1 = -0.5
FILTER_EQUITITY_DD_PHASE1 = 1500 #Esta en valor absoluto, se encuentra despejando del Recovery Factor.
FILTER_TRADES_PHASE1 = 200
FORWARD_FILTER_NET_PROFIT_PHASE1 = 700
FORWARD_FILTER_EXPECTED_PAYOFF_PHASE1 = 10
FORWARD_FILTER_PROFIT_FACTOR_PHASE1 = 1.25
FORWARD_FILTER_CUSTOM_PHASE1 = -0.5
FORWARD_FILTER_EQUITY_DD_PHASE1 = 800
FORWARD_FILTER_TRADES_PHASE1 = 20

INITIAL_DEPOSIT = 10000 #Default Value es 10000 para todos los analisis
DEPOSIT_CURRENCY = "USD" #Default Currency para todos los  analisis





#---------------------------------------------------VARIABLES PARA EL LANZAMIENTO ----------------------------------------------------
BotName = 'EA-B1v1'                           # EA Name OMITIR espacios en blancos, usar como simbolo solamente el "-".
BotMagicNumberSeries = '01'   # should be last numbers of the EA 09 for S3
UserSeries = '01' #01 La Opti la hizo bryan, 02 la hizo richard
#--------------------------------------------PATHS----------------------------------------------------------------------
MT5_PATH = "C:/Program Files/Darwinex MetaTrader 5/terminal64.exe"
FOLDER_PATH = "C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028"
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


def create_dto():
    """Creates DTO(data transfer object), in this object we get all the data needed for the services"""

    bot = BotName.get() or BOT_NAME

    dto = Dto(bot)
    dto.pairs = PairList
    dto.time_frames = TimeFrameList
    dto.filter_net_profit_phase1 = FilterNetProfitPhase1.get() or FILTER_NET_PROFIT_PHASE1
    dto.filter_expected_payoff_phase1 = FilterExpectedPayoffPhase1.get() or FILTER_EXPECTED_PAYOFF_PHASE1
    dto.filter_profit_factor_phase1 = FilterProfitFactorPhase1.get() or FILTER_PROFIT_FACTOR_PHASE1
    dto.filter_custom_phase1 = FilterCustomPhase1.get() or FILTER_CUSTOM_PHASE1
    dto.filter_equitity_dd_phase1 = FilterEquityDDPhase1.get() or FILTER_EQUITITY_DD_PHASE1
    dto.filter_trades_phase1 = FilterTradesPhase1.get() or FILTER_TRADES_PHASE1
    dto.forward_filter_net_profit_phase1 = ForwardFilterNetProfitPhase1.get() or FORWARD_FILTER_NET_PROFIT_PHASE1
    dto.forward_filter_expected_payoff_phase1 = ForwardFilterExpectedPayoffPhase1.get() or FORWARD_FILTER_EXPECTED_PAYOFF_PHASE1
    dto.forward_filter_profit_factor_phase1 = ForwardFilterProfitFactorPhase1.get() or FORWARD_FILTER_PROFIT_FACTOR_PHASE1
    dto.forward_filter_custom_phase1 = ForwardFilterCustomPhase1.get() or FORWARD_FILTER_CUSTOM_PHASE1
    dto.forward_filter_equitity_dd_phase1 = ForwardFilterEquityDDPhase1.get() or FORWARD_FILTER_EQUITY_DD_PHASE1
    dto.forward_filter_trades_phase1 = ForwardFilterTradesPhase1.get() or FORWARD_FILTER_TRADES_PHASE1
    dto.opti_start_date = Opti_start_date.get() or OPTI_START_DATE
    dto.opti_end_date = Opti_end_date.get() or OPTI_END_DATE
    dto.forward_date = ForwardDate.get() or FORWARD_DATE
    dto.initial_deposit = INITIAL_DEPOSIT
    dto.deposit_currency = DEPOSIT_CURRENCY

    return dto


#---------------------------------------------CREATES BOTNAME & INIT & OPTISETS & RESULTS FOLDERS---------------------------------------------------
def CreateALLFoldersPhase1():
    """Creates Folders for Results, Optisets and INIT files"""

    service = CreateFoldersService(create_dto())
    service.run()

#-----------------------------------CREATES INIT FILES FOR ALL PAIRS PHASE 1--------------------------------------------
def CreateIniForAllPhase1():
    """Creates INIT files for all pairs and timeframes selected"""
    bot = BotName.get() or BOT_NAME

    service = CreateIniPhase(bot, PairListTest, TimeFrameListTest, 1)
    service.run()

#------------------------------------LAUNCHES INIT FILES IN CMD FOR PHASE 1---------------------------------------------
def LaunchPhase1():
    """Executes in CMD the INIT file on MT5 for every pair and timeframe selected for Phase 1"""
    bot = BotName.get() or BOT_NAME

    service = LaunchPhase(bot, 1)
    service.launch()

#-------------PROCESS TO OBTAIN OPTISET VALUES FOR PHASE 2 --- CREATE CSV ACCOTATTED FROM XML AND FILTERS IT------------
def AccotateResultsPhase1():
    """Filtrates the Results for Phase 1 Optimization and keeps the results that passes"""

    service = CreateFoldersService(create_dto())
    service.run()

#-----------------------------------------PROCESS TO OBTAIN OPTISET VALUES FOR PHASE 2----------------------------------
def AccotateOptisetsPhase1():
    """Generates the Optiset for the Results that passed the previous filter"""
    service = AccotateOptisetsPhase1(create_dto())
    service.run()

#------------------------------------CREATES INIT FILES FOR ALL PAIRS PHASE 2-------------------------------------------
def CreateIniForAllPhase2():
    """Creates a Phase 2 INIT file for every pair and timeframe selected"""
    bot = BotName.get() or BOT_NAME

    service = CreateIniPhase(bot, PairListTest, TimeFrameListTest, 2)
    service.run()

#-----------------------------------LAUNCHES INIT FILES IN CMD FOR PHASE 2----------------------------------------------
def LaunchPhase2():
    """Executes in CMD the INIT file on MT5 for every pair and timeframe selected for Phase 2"""
    bot = BotName.get() or BOT_NAME

    service = LaunchPhase(bot, 2)
    service.launch()

#--PROCESS TO OBTAIN OPTISET VALUES FOR PHASE 3 --CREATE CSV ACCOTATTED FROM XML, JOINS BACK AND FORWARD AND FILTERS IT-
def AccotateResultsPhase2():
    bot = BotName.get() or BOT_NAME

    service = LaunchPhase(bot, 3)
    service.launch()

#---------------------------------------------------PROCESS TO OBTAIN OPTISET VALUES FOR PHASE 3---(ONLY OPTISET)---------------------------------

def HillClimbPhase2():
    service = HillClimbPhase2(create_dto())
    service.run()

#---------------------------------------------------PROCESS TO OBTAIN OPTISET VALUES FOR PHASE 3--(FOR BACKTESTS)--------------------------------
def BTSetsForPhase3():
    service = BTSetsForPhase3(create_dto())
    service.run

#----------------------------------------------------LAUNCHES INIT FILES IN CMD FOR PHASE 3------SET GENERATION---------------------------
def LaunchPhase3():
    bot = BotName.get() or BOT_NAME

    service = LaunchPhase(bot, 3)
    service.launch()


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
