import time
import os
import xml.etree.ElementTree as et
import pandas as pd
from .print_filters import PrintFilters
from .helpers import movecol
from .helpers import get_csv_list

FOLDER_PATH = "C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028"
REPORT_PATH = os.path.join(FOLDER_PATH, 'reports')
OPTIMIZED_VARIABLES = 4

class AccotateResultsFw:
    def __init__(self, dto):
        self.bot = dto.bot
        self.pair = dto.pair
        self.time_frame = dto.time_frame
        self.dto = dto
        self.null_values_index = 9 + OPTIMIZED_VARIABLES
        self.null_values_columns = 8 + OPTIMIZED_VARIABLES
        self.print_filters = PrintFilters(dto)

    def run(self):
        # CREATE BACKTEST FILE AND APPLY NORMALIZATION
        df_backtest = self.create_backtest_file(self.pair, self.time_frame)

        # CREATE FORWARD FILE
        df_forward = self.create_forward_file(self.pair, self.time_frame)

        # Join DATAFRAMES
        df_complete = self.join_dataframes(df_backtest, df_forward, pair, time_frame)

        # Pick the best

    def create_backtest_file(self, pair, time_frame):
        """CREATE BACKTEST FILE AND APPLY NORMALIZATION"""
        csv_file_name_back = 'OptiWFResults-{}-{}-{}.xml'.format(self.bot, pair, time_frame)
        csv_file_name_back = os.path.join(REPORT_PATH, self.bot, pair, time_frame, 'WF-Results', csv_file_name_back)
        csv_list_back = self.get_csv_list_back(pair, time_frame)

        df_backtest = pd.DataFrame(data=csv_list_back)
        df_backtest = df_backtest.drop(df_backtest.index[:self.null_values_index])
        dfback_columns_name = csv_list_back[self.null_values_columns]
        df_backtest.columns = dfback_columns_name
        df_backtest['Lots'] = 0.1
        df_backtest['Average Loss'] = df_backtest['Result']
        df_backtest['Win Ratio'] = df_backtest['Result']
        df_backtest['Average Loss'] = df_backtest['Average Loss'].str.slice(start=-6, stop=-3)
        df_backtest['Win Ratio'] = df_backtest['Win Ratio'].str.slice(start=-3)
        df_backtest = df_backtest.apply(pd.to_numeric)
        df_backtest['Absolute DD'] = df_backtest['Profit'] / df_backtest['Recovery Factor']
        df_backtest = movecol(df_backtest, ['Absolute DD'], 'Equity DD %')
        df_backtest = movecol(df_backtest, ['Average Loss'], 'Equity DD %', 'Before')
        df_backtest = movecol(df_backtest, ['Win Ratio'], 'Trades')
        df_backtest = movecol(df_backtest, ['Lots'], 'Win Ratio')
        df_backtest = df_backtest.apply(pd.to_numeric)
        df_backtest.sort_values(by=['Pass'], ascending=False, inplace=True)
        df_backtest.to_csv(csv_file_name_back, sep=',', index=False)
        df_backtest.reset_index(inplace=True)

        return self.normalize(df_backtest, pair, time_frame)

    def create_forward_file(self, pair, time_frame):
        """CREATE FORWARD FILE"""

        csv_file_name_forward = 'OptiWFResults-{}-{}-{}.forward.csv'.format(self.bot, pair, time_frame)
        csv_file_name_forward = os.path.join(REPORT_PATH, self.bot, pair, time_frame, 'WF-Results', csv_file_name_forward)
        csv_list_forward = self.get_csv_list_forward(pair, time_frame)

        df_forward = pd.DataFrame(data=csv_list_forward)
        df_forward = df_forward.drop(df_forward.index[:self.null_values_index])
        dfforward_columns_name = csv_list_forward[self.null_values_columns]
        df_forward.columns = dfforward_columns_name
        df_forward['Forward Lots'] = 0.1
        df_forward['Forward Average Loss'] = df_forward['Forward Result']
        df_forward['Forward Win Ratio'] = df_forward['Forward Result']
        df_forward['Forward Average Loss'] = df_forward['Forward Average Loss'].str.slice(start=-6, stop=-3)
        df_forward['Forward Win Ratio'] = df_forward['Forward Win Ratio'].str.slice(start=-3)
        df_forward = df_forward.apply(pd.to_numeric)
        df_forward['Forward Absolute DD'] = df_forward['Profit'] / df_forward['Recovery Factor']
        df_forward = movecol(df_forward, ['Forward Absolute DD'], 'Equity DD %')
        df_forward = movecol(df_forward, ['Forward Average Loss'], 'Equity DD %', 'Before')
        df_forward = movecol(df_forward, ['Forward Win Ratio'], 'Trades')
        df_forward = movecol(df_forward, ['Forward Lots'], 'Forward Win Ratio')
        df_forward = df_forward.apply(pd.to_numeric)
        df_forward.sort_values(by=['Pass'], ascending=False, inplace=True)
        df_forward.reset_index(inplace=True)
        df_forward.to_csv(csv_file_name_forward, sep=',', index=False)
        columns = {
            'Profit': 'Forward Profit',
            'Expected Payoff': 'Forward Expected Payoff',
            'Profit Factor': 'Forward Profit Factor',
            'Recovery Factor': 'Forward Recovery Factor',
            'Sharpe Ratio': 'Forward Sharpe Ratio',
            'Custom': 'Forward Custom',
            'Equity DD %': 'Forward Equity DD %',
            'Trades': 'Forward Trades'
        }
        df_forward.rename(columns=columns, inplace=True)
        print('Done Forward Results for:', pair, time_frame)

        return df_forward

    def normalize(self, df_backtest, pair, time_frame):
        """ Apply Normalization"""
        for index, row in df_backtest.iterrows():
            try:
                avg_loss_norm = 100 / row['Average Loss']
                absolute_dd_norm = float(self.dto.filter_equitity_dd_phase1) / row['Absolute DD']
                normalize_row = float(min(avg_loss_norm, absolute_dd_norm))
                row['Lots'] = float(round(row['Lots'] * normalize_row, 2))
                lot_ration = row['Lots'] / 0.1
                row['Profit'] = row['Profit'] * lot_ration
                row['Expected Payoff'] = row['Expected Payoff'] * lot_ration
                row['Absolute DD'] = row['Absolute DD'] * lot_ration
                row['Average Loss'] = row['Average Loss'] * lot_ration
                df_backtest.loc[index, 'Profit'] = row['Profit']
                df_backtest.loc[index, 'Expected Payoff'] = row['Expected Payoff']
                df_backtest.loc[index, 'Absolute DD'] = row['Absolute DD']
                df_backtest.loc[index, 'Average Loss'] = row['Average Loss']
                df_backtest.loc[index, 'Lots'] = row['Lots']
            except IndexError:
                pass
        print('Done Backtest Results for:', pair, time_frame)

        return df_backtest

    
    def get_csv_list_back(self, pair, time_frame):
        """ get_csv_list_back """
        path = 'OptiWFResults-{}-{}-{}.xml'.format(self.bot, pair, time_frame)
        path = os.path.join(REPORT_PATH, self.bot, pair, time_frame, 'WF-Results', path)
        tree = et.parse(path)

        return get_csv_list(tree.getroot())

    def get_csv_list_forward(self, pair, time_frame):
        """ get_csv_list_forward """
        path = 'OptiWFResults-{}-{}-{}.forward.xml'.format(self.bot, pair, time_frame)
        path = os.path.join(REPORT_PATH, self.bot, pair, time_frame, 'WF-Results', path)
        tree = et.parse(path)

        return get_csv_list(tree.getroot())

    