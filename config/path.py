import os
import sys
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(dotenv_path='.env')

MT5_PATH = os.getenv("MT5_PATH")
MT5_PATH="C:/Program Files/Darwinex MetaTrader 5/terminal64.exe"
FOLDER_PATH = os.getenv("FOLDER_PATH")
FOLDER_PATH= "C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028"
REPORT_PATH = os.path.join(FOLDER_PATH, 'reports')
TESTER_PATH = os.path.join(FOLDER_PATH, 'MQL5', 'Profiles', 'Tester')