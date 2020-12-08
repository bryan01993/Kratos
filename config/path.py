import os
import sys
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(dotenv_path='.env')

MT5_PATH = os.getenv("MT5_PATH")
FOLDER_PATH = os.getenv("FOLDER_PATH")
REPORT_PATH = os.path.join(FOLDER_PATH, 'reports')
TESTER_PATH = os.path.join(FOLDER_PATH, 'MQL5', 'Profiles', 'Tester')