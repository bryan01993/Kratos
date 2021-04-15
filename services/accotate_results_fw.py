import os
import xml.etree.ElementTree as et
import pandas as pd
from .print_filters import PrintFilters
from .helpers import movecol, get_csv_list

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
        df_complete = self.join_dataframes(df_backtest, df_forward, self.pair, self.time_frame)

        # Filter and generate csv
        self.filter(df_complete, self.pair, self.time_frame)

        # Pick the best

    def create_backtest_file(self, pair, time_frame):
        """CREATE BACKTEST FILE AND APPLY NORMALIZATION"""
        csv_file_name_back = 'OptiWFResults-{}-{}-{}-{}-{}.csv'.format(self.bot, pair, time_frame,self.dto.opti_start_date,self.dto.opti_end_date)
        csv_file_name_back = os.path.join(REPORT_PATH, self.bot, pair, time_frame, 'WF_Report', csv_file_name_back)
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

        return df_backtest

    def create_forward_file(self, pair, time_frame):
        """CREATE FORWARD FILE"""

        csv_file_name_forward = 'OptiWFResults-{}-{}-{}-{}-{}.forward.csv'.format(self.bot, pair, time_frame,self.dto.opti_start_date,self.dto.opti_end_date)
        csv_file_name_forward = os.path.join(REPORT_PATH, self.bot, pair, time_frame, 'WF_Report', csv_file_name_forward)
        csv_list_forward = self.get_csv_list_forward(pair, time_frame)

        df_forward = pd.DataFrame(data=csv_list_forward)
        df_forward = df_forward.drop(df_forward.index[:self.null_values_index])
        dfforward_columns_name = csv_list_forward[self.null_values_columns]
        df_forward.columns = dfforward_columns_name
        df_forward['Forward Lots'] = 0.1
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

        return df_forward

    def join_dataframes(self, df_backtest, df_forward, pair, time_frame):
        """ Join the backtest and the forward date frame"""
        file_name = 'OptiWFResults-{}-{}-{}-{}-{}-Complete.csv'.format(self.bot, pair, time_frame,self.dto.opti_start_date,self.dto.opti_end_date)
        file_name = os.path.join(REPORT_PATH, self.bot, pair, time_frame, 'WF_Report', file_name)

        df_backtest.sort_values(by=['Profit'], ascending=False,inplace=True)
        df_backtest['Rank'] = range(0,len(df_backtest))
        df_forward.sort_values(by=['Profit'], ascending=False,inplace=True)
        df_forward['Rank Forward'] = range(0, len(df_forward))

        df_complete = df_backtest.merge(df_forward, on='Pass', suffixes=('', 'Forward'))
        df_complete.to_csv(file_name, sep=',', index=False)

        return df_complete
    
    def get_csv_list_back(self, pair, time_frame):
        """ get_csv_list_back """
        path = 'OptiWFResults-{}-{}-{}-{}-{}.xml'.format(self.bot, pair, time_frame,self.dto.opti_start_date,self.dto.opti_end_date)
        path = os.path.join(REPORT_PATH, self.bot, pair, time_frame, 'WF_Report', path)
        tree = et.parse(path)
            
        return get_csv_list(tree.getroot())

    def get_csv_list_forward(self, pair, time_frame):
        """ get_csv_list_forward """
        path = 'OptiWFResults-{}-{}-{}-{}-{}.forward.xml'.format(self.bot, pair, time_frame,self.dto.opti_start_date,self.dto.opti_end_date)
        path = os.path.join(REPORT_PATH, self.bot, pair, time_frame, 'WF_Report', path)
        tree = et.parse(path)

        return get_csv_list(tree.getroot())

    def filter(self, df_complete, pair, time_frame):
        """ Filter the best row and save like csv"""
        file_name = 'OptiWFResults-{}-{}-{}-{}-{}-Complete-Filtered.csv'.format(self.bot, pair, time_frame,self.dto.opti_start_date,self.dto.opti_end_date)
        file_name = os.path.join(REPORT_PATH, self.bot, pair, time_frame, 'WF_Report', file_name)


        df_complete_filtered = df_complete[(df_complete['Result'] == (df_complete['Result']).max())]
        df_complete_filtered = df_complete_filtered[:1]
        df_complete_filtered.to_csv(file_name, sep=',', index=False)