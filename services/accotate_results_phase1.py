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

class AccotateResultsPhase1:
    """Filtrates the Results for Phase 1 Optimization and keeps the results that passes"""

    def __init__(self, dto):
        self.bot = dto.bot
        self.pairs = dto.pairs
        self.time_frames = dto.time_frames
        self.dto = dto
        self.null_values_index = 9 + OPTIMIZED_VARIABLES
        self.null_values_columns = 8 + OPTIMIZED_VARIABLES
        self.print_filters = PrintFilters(dto)

    def run(self):
        """Filtrates the Results for Phase 1 Optimization and keeps the results that passes"""

        print('Begins Phase 1 Results Filtering')
        phase_start = time.time()
        total_count = 0
        project_count = 0

        for pair in self.pairs:
            for time_frame in self.time_frames:
                try:
                    # CREATE BACKTEST FILE AND APPLY NORMALIZATION
                    df_backtest = self.create_backtest_file(pair, time_frame)

                    # CREATE FORWARD FILE
                    df_forward = self.create_forward_file(pair, time_frame)

                    # Join DATAFRAMES
                    df_complete = self.join_dataframes(df_backtest, df_forward, pair, time_frame)

                    # AFTER UNION FILTER
                    df_complete = self.filter(df_complete, pair, time_frame)

                    try:
                        df_complete.drop(columns=['index'], inplace=True)
                    except KeyError:
                        pass

                    file_name = 'OptiResults-{}-{}-{}-Phase1.Complete-Filtered.csv'.format(self.bot, pair, time_frame)
                    file_name = os.path.join(REPORT_PATH, self.bot, pair, time_frame, file_name)
                    df_complete.to_csv(file_name, sep=',', index=False)

                    print("Filtered Dataframe saved")
                except FileNotFoundError:
                    pass
        phase2_end = time.time()
        time_result = (phase2_end - phase_start) / 60
        project_total_ration = (project_count / total_count) * 100
        self.print_filters.print_forward_filters()

        print('From a Total of :', total_count, 'backtests')
        print('Only', project_count, ' passed the filters.', round(project_total_ration, ndigits=2), '%')
        print('Phase 1 Results Accotated in', round(time_result), 'minutes')

    def create_backtest_file(self, pair, time_frame):
        """CREATE BACKTEST FILE AND APPLY NORMALIZATION"""
        csv_file_name_back = 'OptiResults-{}-{}-{}-Phase1.csv'.format(self.bot, pair, time_frame)
        csv_file_name_back = os.path.join(REPORT_PATH, self.bot, pair, time_frame, csv_file_name_back)
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

    def create_forward_file(self, pair, time_frame):
        """CREATE BACKTEST FILE AND APPLY NORMALIZATION"""

        csv_file_name_forward = 'OptiResults-{}-{}-{}-Phase1.forward.csv'.format(self.bot, pair, time_frame)
        csv_file_name_forward = os.path.join(REPORT_PATH, self.bot, pair, time_frame, csv_file_name_forward)
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

    def join_dataframes(self, df_backtest, df_forward, pair, time_frame):
        """ Join the backtest and the forward date frame"""
        file_name = 'OptiResults-{}-{}-{}-Phase1.Complete-Filtered.csv'.format(self.bot, pair, time_frame)
        file_name = os.path.join(REPORT_PATH, self.bot, pair, time_frame, file_name)

        df_complete = pd.concat([df_backtest, df_forward], axis=1)
        df_complete = df_complete.loc[:, ~df_complete.columns.duplicated()]
        df_complete = movecol(df_complete, ['Forward Result'], 'Result')
        df_complete = movecol(df_complete, ['Forward Profit'], 'Profit')
        df_complete = movecol(df_complete, ['Forward Expected Payoff'], 'Expected Payoff')
        df_complete = movecol(df_complete, ['Forward Profit Factor'], 'Profit Factor')
        df_complete = movecol(df_complete, ['Forward Recovery Factor'], 'Recovery Factor')
        df_complete = movecol(df_complete, ['Forward Sharpe Ratio'], 'Sharpe Ratio')
        df_complete = movecol(df_complete, ['Forward Custom'], 'Custom')
        df_complete = movecol(df_complete, ['Forward Average Loss'], 'Average Loss')
        df_complete = movecol(df_complete, ['Forward Equity DD %'], 'Equity DD %')
        df_complete = movecol(df_complete, ['Forward Absolute DD'], 'Absolute DD')
        df_complete = movecol(df_complete, ['Forward Trades'], 'Trades')
        df_complete = movecol(df_complete, ['Forward Win Ratio'], 'Win Ratio')
        df_complete = movecol(df_complete, ['Forward Lots'], 'Lots')
        df_complete = df_complete.drop(['Back Result'], axis=1)
        df_complete.to_csv(file_name, sep=',', index=False)

        return df_complete

    def filter(self, df_complete, pair, time_frame):
        """ Filter complete dataframe"""

        print('Before filtering the len of Complete DataFrame for', pair, time_frame, 'is:', len(df_complete))
        for index, row in df_complete.iterrows():
            total_count += 1
            avg_loss_norm = 100 / row['Average Loss']
            absolute_dd_norm = float(self.dto.filter_equitity_dd_phase1) / row['Absolute DD']
            normalize_row = float(min(avg_loss_norm, absolute_dd_norm))
            row['Lots'] = float(round(row['Lots'] * normalize_row, 2))
            lot_ration = row['Lots'] / 0.1
            row['Forward Lots'] = row['Lots']
            row['Forward Profit'] = row['Forward Profit'] * lot_ration
            row['Forward Expected Payoff'] = row['Forward Expected Payoff'] * lot_ration
            row['Forward Absolute DD'] = row['Forward Absolute DD'] * lot_ration
            row['Forward Average Loss'] = row['Forward Average Loss'] * lot_ration
            df_complete.loc[index, 'Forward Profit'] = row['Forward Profit']
            df_complete.loc[index, 'Forward Expected Payoff'] = row['Forward Expected Payoff']
            df_complete.loc[index, 'Forward Absolute DD'] = row['Forward Absolute DD']
            df_complete.loc[index, 'Forward Average Loss'] = row['Forward Average Loss']
            df_complete.loc[index, 'Forward Lots'] = row['Forward Lots']

            try:
                if (row['Profit'] >= int(self.dto.filter_net_profit_phase1)
                        and row['Expected Payoff'] >= float(self.dto.filter_expected_payoff_phase1)
                        and row['Profit Factor'] >= float(self.dto.filter_profit_factor_phase1)
                        and row['Custom'] >= float(self.dto.filter_custom_phase1)
                        and row['Absolute DD'] <= float(self.dto.filter_equitity_dd_phase1) + 100
                        and row['Trades'] >= int(self.dto.filter_trades_phase1)
                        and row['Forward Profit'] >= int(self.bot.forward_filter_net_profit_phase1)
                        and row['Forward Expected Payoff'] >= float(self.dto.forward_filter_expected_payoff_phase1)
                        and row['Forward Profit Factor'] >= float(self.dto.forward_filter_profit_factor_phase1)
                        and row['Forward Custom'] >= float(self.dto.forward_filtler_custom_phase1)
                        and row['Forward Absolute DD'] <= float(self.dto.forward_filter_equitity_dd_phase1) + 100
                        and row['Forward Trades'] >= int(self.dto.forward_filter_trades_phase1)):
                    project_count += 1
                else:
                    df_complete.drop(labels=index, inplace=True)
            except IndexError:
                pass
        print('After filtering the len of Filtered DataFrame from Phase 1 for', pair, time_frame, 'is:', len(df_complete))

        return df_complete

    def get_csv_list_back(self, pair, time_frame):
        """ get_csv_list_back """
        path = 'OptiResults-{}-{}-{}-Phase1.xml'.format(self.bot, pair, time_frame)
        path = os.path.join(REPORT_PATH, self.bot, pair, time_frame, path)
        tree = et.parse(path)

        return get_csv_list(tree.getroot())

    def get_csv_list_forward(self, pair, time_frame):
        """ get_csv_list_forward """
        path = 'OptiResults-{}-{}-{}-Phase1.forward.xml'.format(self.bot, pair, time_frame)
        path = os.path.join(REPORT_PATH, self.bot, pair, time_frame, path)
        tree = et.parse(path)

        return get_csv_list(tree.getroot())
