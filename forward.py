import os.path
import tkinter as tk
from tkinter import Frame, Button, Label, Tk, Toplevel, Entry, IntVar, Checkbutton
import subprocess
import pandas as pd

from Dto import Dto
from services.forward_walk import ForwardWalk

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

    bot = BOT_NAME

    dto = Dto(bot)
    dto.pairs = PAIRS
    dto.time_frames = TIME_FRAMES
    dto.opti_start_date = OPTI_START_DATE
    dto.opti_end_date = OPTI_END_DATE
    dto.forward_date = FORWARD_DATE
    dto.initial_deposit = INITIAL_DEPOSIT
    dto.deposit_currency = DEPOSIT_CURRENCY
    dto.optimized_variables = OPTIMIZED_VARIABLES
    dto.filter_net_profit_phase1 = FILTER_NET_PROFIT_PHASE1
    dto.filter_expected_payoff_phase1 = FILTER_EXPECTED_PAYOFF_PHASE1
    dto.filter_profit_factor_phase1 = FILTER_PROFIT_FACTOR_PHASE1
    dto.filter_custom_phase1 = FILTER_CUSTOM_PHASE1
    dto.filter_equitity_dd_phase1 = FILTER_EQUITITY_DD_PHASE1
    dto.filter_trades_phase1 = FILTER_TRADES_PHASE1
    dto.forward_filter_net_profit_phase1 = FORWARD_FILTER_NET_PROFIT_PHASE1
    dto.forward_filter_expected_payoff_phase1 = FORWARD_FILTER_EXPECTED_PAYOFF_PHASE1
    dto.forward_filter_profit_factor_phase1 = FORWARD_FILTER_PROFIT_FACTOR_PHASE1
    dto.forward_filter_custom_phase1 = FORWARD_FILTER_CUSTOM_PHASE1
    dto.forward_filter_equitity_dd_phase1 = FORWARD_FILTER_EQUITY_DD_PHASE1
    dto.forward_filter_trades_phase1 = FORWARD_FILTER_TRADES_PHASE1
    
    return dto

ForwardWalk(create_dto()).run()