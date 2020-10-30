import os.path
import tkinter as tk
from tkinter import Frame, Button, Label, Tk, Toplevel, Entry, IntVar, Checkbutton
import subprocess
import pandas as pd

from Dto import Dto
from services.create_folders import CreateFolders
from services.accotate_results_phase1 import AccotateResultsPhase1
from services.create_ini_phase import CreateIniPhase
from services.launch_phase import LaunchPhase
from services.accotate_optisets_phase1 import AccotateOptisetsPhase1
from services.accotate_optisets_phase2 import AccotateOptisetsPhase2
from services.hill_climb_phase2 import HillClimbPhase2
from services.bt_sets_for_phase3 import BTSetsForPhase3
from services.print_filters import PrintFilters

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

OPTIMIZED_VARIABLES = 4

INITIAL_DEPOSIT = 10000 #Default Value es 10000 para todos los analisis
DEPOSIT_CURRENCY = "USD" #Default Currency para todos los  analisis
REAL_CURRENCY = "EUR" #Default Currency para Prueba de Tom


USER_SERIES = '01' #01 La Opti la hizo bryan, 02 la hizo richard
BOT_MAGIC_NUMBER_SERIES = '01'   # should be last numbers of the EA 09 for S3

# List Of Pairs to select for launch.
PAIRS = {'GBPUSD':'01', 'EURUSD':'02', 'USDCAD':'03', 'USDCHF':'04', 'USDJPY':'05', 'GBPJPY':'06', 'EURAUD':'07', 'EURGBP':'08', 'EURJPY':'09', 'EURCHF':'10'}

# Agregar 'D1', no parece obtener la data de M1 dentro del MT5.
TIME_FRAMES = {'H4':'96', 'H1':'95', 'M30':'94', 'M15':'93', 'M5':'92', 'M1':'91'}

def create_dto():
    """Creates DTO(data transfer object), in this object we get all the data needed for the services"""

    bot = bot_name.get() or BOT_NAME

    dto = Dto(bot)
    dto.pairs = PAIRS
    dto.time_frames = TIME_FRAMES
    dto.filter_net_profit_phase1 = FilterNetProfitPhase1.get() or FILTER_NET_PROFIT_PHASE1
    dto.filter_expected_payoff_phase1 = FilterExpectedPayoffPhase1.get() or FILTER_EXPECTED_PAYOFF_PHASE1
    dto.filter_profit_factor_phase1 = filter_profit_factor_phase1.get() or FILTER_PROFIT_FACTOR_PHASE1
    dto.filter_custom_phase1 = filter_custom_phase1.get() or FILTER_CUSTOM_PHASE1
    dto.filter_equitity_dd_phase1 = filter_equity_dd_phase1.get() or FILTER_EQUITITY_DD_PHASE1
    dto.filter_trades_phase1 = filter_trades_phase1.get() or FILTER_TRADES_PHASE1
    dto.forward_filter_net_profit_phase1 = forward_filter_net_profit_phase1.get() or FORWARD_FILTER_NET_PROFIT_PHASE1
    dto.forward_filter_expected_payoff_phase1 = forward_filter_expected_payoff_phase1.get() or FORWARD_FILTER_EXPECTED_PAYOFF_PHASE1
    dto.forward_filter_profit_factor_phase1 = forward_filter_profit_factor_phase1.get() or FORWARD_FILTER_PROFIT_FACTOR_PHASE1
    dto.forward_filter_custom_phase1 = forward_filter_custom_phase1.get() or FORWARD_FILTER_CUSTOM_PHASE1
    dto.forward_filter_equitity_dd_phase1 = forward_filter_equity_dd_phase1.get() or FORWARD_FILTER_EQUITY_DD_PHASE1
    dto.forward_filter_trades_phase1 = forward_filter_trades_phase1.get() or FORWARD_FILTER_TRADES_PHASE1
    dto.opti_start_date = opti_start_date.get() or OPTI_START_DATE
    dto.opti_end_date = opti_end_date.get() or OPTI_END_DATE
    dto.forward_date = forward_date.get() or FORWARD_DATE
    dto.initial_deposit = INITIAL_DEPOSIT
    dto.deposit_currency = DEPOSIT_CURRENCY
    dto.optimized_variables = OPTIMIZED_VARIABLES

    print(dto.forward_filter_equitity_dd_phase1)

    return dto


def create_folder_phase1():
    """Creates Folders for Results, Optisets and INIT files"""
    service = CreateFolders(create_dto())
    service.run()

def create_ini_phase1():
    """Creates INIT files for all pairs and timeframes selected"""
    service = CreateIniPhase(create_dto(), 1)
    service.run()

def launch_phase1():
    """Executes in CMD the INIT file on MT5 for every pair and timeframe selected for Phase 1"""
    service = LaunchPhase(create_dto(), 1)
    service.launch()

def accotate_results_phase1():
    """Filtrates the Results for Phase 1 Optimization and keeps the results that passes"""
    service = AccotateResultsPhase1(create_dto())
    service.run()

def accotate_optisets_phase1():
    """Generates the Optiset for the Results that passed the previous filter"""
    service = AccotateOptisetsPhase1(create_dto())
    service.run()

def create_ini_phase2():
    """Creates a Phase 2 INIT file for every pair and timeframe selected"""
    service = CreateIniPhase(create_dto(), 2)
    service.run()

def launch_phase2():
    """Executes in CMD the INIT file on MT5 for every pair and timeframe selected for Phase 2"""
    service = LaunchPhase(create_dto(), 2)
    service.launch()

def accotate_results_phase2():
    """Filtrates the Results for Phase 1 Optimization and keeps the results that passes"""
    service = AccotateResultsPhase1(create_dto())
    service.run()

def hill_climb_phase2():
    """PROCESS TO OBTAIN OPTISET VALUES FOR PHASE 3---(ONLY OPTISET)"""
    service = HillClimbPhase2(create_dto())
    service.run()

def bt_set_for_phase3():
    """PROCESS TO OBTAIN OPTISET VALUES FOR PHASE 3--(FOR BACKTESTS)"""
    service = BTSetsForPhase3(create_dto())
    service.run()

def launch_phase3():
    """ LAUNCHES INIT FILES IN CMD FOR PHASE 3------SET GENERATION"""
    service = LaunchPhase(create_dto(), 3)
    service.launch()

#-----------------------------------------------------COMIENZA INTERFACE--------------------------------------
class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.quit = tk.Button(self, text="QUIT", fg="red", command=self.master.destroy)
        self.quit.pack(side="bottom")


root = tk.Tk()
root.title("Optimizer Ecoelite")
root.config(bg='gray')
my_frame = Frame(root)


my_frame.pack(fill='both', expand=True)
my_frame.config(bg='#292524', width=1200, height=1000)
Label(my_frame, text='Lanzamiento', fg='white', bg='#292524', font=(18)).grid(row=0, column=0, columnspan=2)

Label(my_frame, text="EA Name:", fg='white', bg='#292524').grid(row=1, column=0, sticky='w', padx=10, pady=10)
bot_name = Entry(my_frame, fg='white', bg='#151312', width=12)
bot_name.grid(row=1, column=1, sticky='w', padx=10, pady=10)

Label(my_frame, text="Start Date:", fg='white', bg='#292524').grid(row=2, column=0, sticky='w', padx=10, pady=10)
opti_start_date = Entry(my_frame, fg='white', bg='#151312', width=10)
opti_start_date.grid(row=2, column=1, sticky='w', padx=10, pady=10)

Label(my_frame, text="Forward Date:", fg='white', bg='#292524').grid(row=3, column=0, sticky='w', padx=10, pady=10)
forward_date = Entry(my_frame, fg='white', bg='#151312', width=10)
forward_date.grid(row=3, column=1, sticky='w', padx=10, pady=10)

Label(my_frame, text="End Date:", fg='white', bg='#292524').grid(row=4, column=0, sticky='w', padx=10, pady=10)
opti_end_date = Entry(my_frame, fg='white', bg='#151312', width=10)
opti_end_date.grid(row=4, column=1, sticky='w', padx=10, pady=10)

def iterations_function():
    dto = create_dto()
    count = 0
    for pair in dto.pairs:
        for time_frame in dto.time_frames:
            if dto.pairs[pair] == 1 and dto.time_frames[time_frame] == 1:
                count += 1
                print(pair, time_frame)
    print('EA to analize:', dto.bot, 'Total:', count, 'Optimizations')


Button(my_frame, text="Create ALL Folders", fg='black', bg='#34D7DF', borderwidth=0, command=create_folder_phase1).grid(row=2, column=2, padx=10, pady=10)

Button(my_frame, text="Create Ini Files For Phase 1", fg='black', bg='#34D7DF', borderwidth=0, command=create_ini_phase1).grid(row=4, column=2, padx=10, pady=10)

Button(my_frame, text='LAUNCH Phase 1', fg='black', bg='#E74C3C', borderwidth=0, command=launch_phase1).grid(row=5, column=2, padx=10, pady=10)

Button(my_frame, text="Accotate Results from Phase 1", fg='black', bg='#34D7DF', borderwidth=0, command=accotate_results_phase1).grid(row=6, column=2, padx=10, pady=10)

Button(my_frame, text="Accotate Optisets for Phase 2", fg='black', bg='#34D7DF', borderwidth=0, command=accotate_optisets_phase1).grid(row=7, column=2, padx=10, pady=10)

Button(my_frame, text="Create Ini Files for Phase 2", fg='black', bg='#34D7DF', borderwidth=0, command=create_ini_phase2).grid(row=8, column=2, padx=10, pady=10)

Button(my_frame, text='LAUNCH Phase 2', fg='black', bg='#E74C3C', borderwidth=0, command=launch_phase2).grid(row=9, column=2, padx=10, pady=10)

Button(my_frame, text='Join Results and Filter for Phase 3', fg='black', bg='#34D7DF', borderwidth=0, command=accotate_results_phase2).grid(row=10, column=2, padx=10, pady=10)

Button(my_frame, text='Accotate Optisets for Phase 3 TO FIX', fg='black', bg='#34D7DF', borderwidth=0, command=AccotateOptisetsPhase2).grid(row=11, column=2, padx=10, pady=10)

Button(my_frame, text='Produce Sets and Inis for Phase 3', fg='black', bg='#34D7DF', borderwidth=0, command=bt_set_for_phase3).grid(row=12, column=2, padx=10, pady=10)

Button(my_frame, text='LAUNCH Phase 3', fg='black', bg='#E74C3C', borderwidth=0, command=launch_phase3).grid(row=13, column=2, padx=10, pady=10)

Button(my_frame, text='LAUNCH HC', fg='black', bg='#E74C3C', borderwidth=0, command=hill_climb_phase2).grid(row=14, column=2, padx=10, pady=10)

Label(my_frame, text="Pairs", fg='white', bg='#292524').grid(row=5, column=0, padx=10, pady=10)


pairGBPUSD = IntVar()
Checkbutton(my_frame, text="GBPUSD", fg='black', bg='#68E552', variable=pairGBPUSD, onvalue=1, offvalue=0).grid(row=6, column=0)

pairEURUSD = IntVar()
Checkbutton(my_frame, text="EURUSD", fg='black', bg='#68E552', variable=pairEURUSD, onvalue=1, offvalue=0).grid(row=7, column=0)

pairUSDCAD = IntVar()
Checkbutton(my_frame, text="USDCAD", fg='black', bg='#68E552', variable=pairUSDCAD, onvalue=1, offvalue=0).grid(row=8, column=0)

pairUSDCHF = IntVar()
Checkbutton(my_frame, text="USDCHF", fg='black', bg='#68E552', variable=pairUSDCHF, onvalue=1, offvalue=0).grid(row=9, column=0)

pairUSDJPY = IntVar()
Checkbutton(my_frame, text="USDJPY", fg='black', bg='#68E552', variable=pairUSDJPY, onvalue=1, offvalue=0).grid(row=10, column=0)

pairGBPJPY = IntVar()
Checkbutton(my_frame, text="GBPJPY", fg='black', bg='#68E552', variable=pairGBPJPY, onvalue=1, offvalue=0).grid(row=11, column=0)

pairEURAUD = IntVar()
Checkbutton(my_frame, text="EURAUD", fg='black', bg='#68E552', variable=pairEURAUD, onvalue=1, offvalue=0).grid(row=12, column=0)

pairEURGBP = IntVar()
Checkbutton(my_frame, text="EURGBP", fg='black', bg='#68E552', variable=pairEURGBP, onvalue=1, offvalue=0).grid(row=13, column=0)

pairEURJPY = IntVar()
Checkbutton(my_frame, text="EURJPY", fg='black', bg='#68E552', variable=pairEURJPY, onvalue=1, offvalue=0).grid(row=14, column=0)

pairEURCHF = IntVar()
Checkbutton(my_frame, text="EURCHF", fg='black', bg='#68E552', variable=pairEURCHF, onvalue=1, offvalue=0).grid(row=15, column=0)


PairListTest = {'GBPUSD':pairGBPUSD, 'EURUSD':pairEURUSD, 'USDCAD':pairUSDCAD, 'USDCHF':pairUSDCHF, 'USDJPY':pairUSDJPY, 'GBPJPY':pairGBPJPY, 'EURAUD':pairEURAUD, 'EURGBP':pairEURGBP, 'EURJPY':pairEURJPY, 'EURCHF':pairEURCHF}


def checkallpairs():
    pairGBPUSD.set(1), pairEURUSD.set(1), pairUSDCAD.set(1), pairUSDJPY.set(1), pairUSDCHF.set(1), pairGBPJPY.set(1), pairEURAUD.set(1), pairEURGBP.set(1), pairEURJPY.set(1), pairEURCHF.set(1)

Button(my_frame, text='Check all pairs', fg='black', bg='#34D7DF', borderwidth=0, command=checkallpairs).grid(row=18, column=0, padx=10, pady=10)

Label(my_frame, text="Timeframes", fg='white', bg='#292524').grid(row=5, column=1, padx=10, pady=10)

TFH4 = IntVar()
Checkbutton(my_frame, text="H4", fg='black', bg='#68E552', variable=TFH4).grid(row=6, column=1)

TFH1 = IntVar()
Checkbutton(my_frame, text="H1", fg='black', bg='#68E552', variable=TFH1).grid(row=7, column=1)

TFM30 = IntVar()
Checkbutton(my_frame, text="M30", fg='black', bg='#68E552', variable=TFM30).grid(row=8, column=1)

TFM15 = IntVar()
Checkbutton(my_frame, text="M15", fg='black', bg='#68E552', variable=TFM15).grid(row=9, column=1)

TFM5 = IntVar()
Checkbutton(my_frame, text="M5", fg='black', bg='#68E552', variable=TFM5).grid(row=10, column=1)

TFM1 = IntVar()
Checkbutton(my_frame, text="M1", fg='black', bg='#68E552', variable=TFM1).grid(row=11, column=1)

TimeFrameListTest = {'H4':TFH4, 'H1':TFH1, 'M30':TFM30, 'M15':TFM15, 'M5':TFM5, 'M1':TFM1}

#-----------------------------------------------------------------------PHASE 1 SETTINGS----------------------------------------------------------------------------------------------
Phase1FilterSettingswindow = tk.Toplevel(my_frame, bg='#292524', width=800, height=600)
Phase1FilterSettingswindow.title("Base Filter Settings")
Label(Phase1FilterSettingswindow, text='Base Filter Settings', fg='white', bg='#292524', font=(18)).grid(row=0, column=0, columnspan=2)

Label(Phase1FilterSettingswindow, text="Min. Net Profit:", fg='white', bg='#292524').grid(row=1, column=0, sticky='w', padx=10, pady=10)
FilterNetProfitPhase1 = Entry(Phase1FilterSettingswindow, fg='white', bg='#151312', width=12)
FilterNetProfitPhase1.grid(row=1, column=1, sticky='w', padx=10, pady=10)

Label(Phase1FilterSettingswindow, text="Min. Exp. Payoff:", fg='white', bg='#292524').grid(row=2, column=0, sticky='w', padx=10, pady=10)
FilterExpectedPayoffPhase1 = Entry(Phase1FilterSettingswindow, fg='white', bg='#151312', width=12)
FilterExpectedPayoffPhase1.grid(row=2, column=1, sticky='w', padx=10, pady=10)

Label(Phase1FilterSettingswindow, text="Min. Profit Factor:", fg='white', bg='#292524').grid(row=3, column=0, sticky='w', padx=10, pady=10)
filter_profit_factor_phase1 = Entry(Phase1FilterSettingswindow, fg='white', bg='#151312', width=12)
filter_profit_factor_phase1.grid(row=3, column=1, sticky='w', padx=10, pady=10)

Label(Phase1FilterSettingswindow, text="Min. Custom Value:", fg='white', bg='#292524').grid(row=4, column=0, sticky='w', padx=10, pady=10)
filter_custom_phase1 = Entry(Phase1FilterSettingswindow, fg='white', bg='#151312', width=12)
filter_custom_phase1.grid(row=4, column=1, sticky='w', padx=10, pady=10)

Label(Phase1FilterSettingswindow, text="Max. Drawdown:", fg='white', bg='#292524').grid(row=5, column=0, sticky='w', padx=10, pady=10)
filter_equity_dd_phase1 = Entry(Phase1FilterSettingswindow, fg='white', bg='#151312', width=12)
filter_equity_dd_phase1.grid(row=5, column=1, sticky='w', padx=10, pady=10)

Label(Phase1FilterSettingswindow, text="Min. Trades:", fg='white', bg='#292524').grid(row=6, column=0, sticky='w', padx=10, pady=10)
filter_trades_phase1 = Entry(Phase1FilterSettingswindow, fg='white', bg='#151312', width=12)
filter_trades_phase1.grid(row=6, column=1, sticky='w', padx=10, pady=10)

#--------------------------------------------------------------------------PHASE 2 SETTINGS--------------------------------------------------------------------------------------------
Phase2FilterSettingswindow = tk.Toplevel(my_frame, bg='#292524', width=800, height=600)
Phase2FilterSettingswindow.title("Proyection Filter Settings")
Label(Phase2FilterSettingswindow, text='Proyection Filter Settings', fg='white', bg='#292524', font=(18)).grid(row=0, column=0, columnspan=2)

Label(Phase2FilterSettingswindow, text="Forward Min. Net Profit:", fg='white', bg='#292524').grid(row=1, column=0, sticky='w', padx=10, pady=10)
forward_filter_net_profit_phase1 = Entry(Phase2FilterSettingswindow, fg='white', bg='#151312', width=12)
forward_filter_net_profit_phase1.grid(row=1, column=1, sticky='w', padx=10, pady=10)

Label(Phase2FilterSettingswindow, text="Forward Min. Exp. Payoff:", fg='white', bg='#292524').grid(row=2, column=0, sticky='w', padx=10, pady=10)
forward_filter_expected_payoff_phase1 = Entry(Phase2FilterSettingswindow, fg='white', bg='#151312', width=12)
forward_filter_expected_payoff_phase1.grid(row=2, column=1, sticky='w', padx=10, pady=10)

Label(Phase2FilterSettingswindow, text="Forward Min. Profit Factor:", fg='white', bg='#292524').grid(row=3, column=0, sticky='w', padx=10, pady=10)
forward_filter_profit_factor_phase1 = Entry(Phase2FilterSettingswindow, fg='white', bg='#151312', width=12)
forward_filter_profit_factor_phase1.grid(row=3, column=1, sticky='w', padx=10, pady=10)

Label(Phase2FilterSettingswindow, text="Forward Min. Custom Value:", fg='white', bg='#292524').grid(row=4, column=0, sticky='w', padx=10, pady=10)
forward_filter_custom_phase1 = Entry(Phase2FilterSettingswindow, fg='white', bg='#151312', width=12)
forward_filter_custom_phase1.grid(row=4, column=1, sticky='w', padx=10, pady=10)

Label(Phase2FilterSettingswindow, text="Forward Max. Drawdown:", fg='white', bg='#292524').grid(row=5, column=0, sticky='w', padx=10, pady=10)
forward_filter_equity_dd_phase1 = Entry(Phase2FilterSettingswindow, fg='white', bg='#151312', width=12)
forward_filter_equity_dd_phase1.grid(row=5, column=1, sticky='w', padx=10, pady=10)

Label(Phase2FilterSettingswindow, text="Forward Min. Trades:", fg='white', bg='#292524').grid(row=6, column=0, sticky='w', padx=10, pady=10)
forward_filter_trades_phase1 = Entry(Phase2FilterSettingswindow, fg='white', bg='#151312', width=12)
forward_filter_trades_phase1.grid(row=6, column=1, sticky='w', padx=10, pady=10)


def graph_backtest_phase1():
    print('use service')

def graph_backtest_phase2():
    print('use service')

def checkalltfs():
    TFH4.set(1), TFH1.set(1), TFM30.set(1), TFM15.set(1), TFM5.set(1), TFM1.set(1)

Button(my_frame, text='Graph Selected Pair/TF Opti Phase 1', fg='black', bg='#34D7DF', borderwidth=0, command=graph_backtest_phase1).grid(row=12, column=3, padx=10, pady=10)
Button(my_frame, text='Graph Selected Pair/TF Opti Phase 2', fg='black', bg='#34D7DF', borderwidth=0, command=graph_backtest_phase2).grid(row=11, column=3, padx=10, pady=10)
Button(my_frame, text='Check all timeframes', fg='black', bg='#34D7DF', borderwidth=0, command=checkalltfs).grid(row=18, column=1, padx=10, pady=10)

IteratetionsFunction = Button(my_frame, text='Iterate Selections', fg='black', bg='#34D7DF', borderwidth=0, command=iterations_function)
IteratetionsFunction.grid(row=21, column=0, padx=10, pady=10, columnspan=2)

def print_filters():
    service = PrintFilters(create_dto())
    service.print_filters()

def print_forward_filters():
    service = PrintFilters(create_dto())
    service.print_forward_filters()

FiltersTestFunction = Button(my_frame, text='Test Filters', fg='black', bg='#34D7DF', borderwidth=0, command=print_filters)
FiltersTestFunction.grid(row=13, column=3, padx=10, pady=10, columnspan=2)

FiltersTestFunction = Button(Phase1FilterSettingswindow, text='Test Filters', fg='black', bg='#34D7DF', borderwidth=0, command=print_filters)
FiltersTestFunction.grid(row=13, column=3, padx=10, pady=10, columnspan=2)

FiltersTestFunction = Button(Phase2FilterSettingswindow, text='Test Filters', fg='black', bg='#34D7DF', borderwidth=0, command=print_forward_filters)
FiltersTestFunction.grid(row=13, column=3, padx=10, pady=10, columnspan=2)
app = Application(master=root)
app.mainloop()
