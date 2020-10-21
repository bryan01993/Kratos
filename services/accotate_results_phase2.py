import time
import os
import xml.etree.ElementTree as et
import pandas as pd

FOLDER_PATH = "C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028"
REPORT_PATH = os.path.join(FOLDER_PATH, 'reports')
OPTIMIZED_VARIABLES = 4

class AccotateResultsPhase2:
    """Filtrates the Results for Phase 1 Optimization and keeps the results that passes"""

    def __init__(self, dto):
        self.bot = dto.bot
        self.pairs = dto.pairs
        self.time_frames = dto.time_frames
        self.dto = dto

    def run(self):
        """Filtrates the Results for Phase 2 Optimization and keeps the results that passes"""
        print('Begins Phase 2 Results Filtering')
        start = time.time()
        total_count = 0
        project_count = 0
        for pair in self.pairs:
            for time_frame in self.time_frames:
                try:
                    #---------------------------------------------------------------BACKTEST FILE-------------------------------------------------------------------------------------------------------------------
                    null_values_index = 9 + OPTIMIZED_VARIABLES
                    null_values_columns = 8 + OPTIMIZED_VARIABLES

                    csv_file_name_back = 'OptiResults-{}-{}-{}-Phase1.csv'.format(self.bot, pair, time_frame)
                    csv_file_name_back = os.path.join(REPORT_PATH, self.bot, pair, time_frame, csv_file_name_back)
                    csv_list_back = self.get_csv_list_back(pair, time_frame)

                    dfback = pd.DataFrame(data=csv_list_back)
                    dfback = dfback.drop(dfback.index[:null_values_index]) # 4 variables hacen 13 9
                    dfback_columns_name = csv_list_back[null_values_columns] # 4 variables hacen 12
                    dfback.columns = dfback_columns_name
                    dfback['Lots'] = 0.1
                    dfback['Average Loss'] = dfback['Result']
                    dfback['Win Ratio'] = dfback['Result']
                    dfback['Average Loss'] = dfback['Average Loss'].str.slice(start=-6, stop=-3)
                    dfback['Win Ratio'] = dfback['Win Ratio'].str.slice(start=-3)
                    dfback = dfback.apply(pd.to_numeric)
                    dfback['Absolute DD'] = dfback['Profit'] / dfback['Recovery Factor']
                    dfback = self.movecol(dfback, ['Absolute DD'], 'Equity DD %')
                    dfback = self.movecol(dfback, ['Average Loss'], 'Equity DD %', 'Before')
                    dfback = self.movecol(dfback, ['Win Ratio'], 'Trades')
                    dfback = self.movecol(dfback, ['Lots'], 'Win Ratio')
                    dfback = dfback.apply(pd.to_numeric)
                    dfback.to_csv(csv_file_name_back, sep=',', index=False)
                    dfback.reset_index(inplace=True)
                    for index, row in dfback.iterrows():  # NORMALIZATION FIRST
                        try:
                            avg_loss_norm = 100 / row['Average Loss']
                            absolute_dd_norm = float(self.dto.filter_equitity_dd_phase1) / row['Absolute DD']
                            normalize_row = float(min(avg_loss_norm, absolute_dd_norm))
                            row['Lots'] = float(round(row['Lots'] * normalize_row, 2))
                            lot_ratio = row['Lots'] / 0.1
                            row['Profit'] = row['Profit'] * lot_ratio
                            row['Expected Payoff'] = row['Expected Payoff'] * lot_ratio
                            row['Absolute DD'] = row['Absolute DD'] * lot_ratio
                            row['Average Loss'] = row['Average Loss'] * lot_ratio
                            dfback.loc[index, 'Profit'] = row['Profit']
                            dfback.loc[index, 'Expected Payoff'] = row['Expected Payoff']
                            dfback.loc[index, 'Absolute DD'] = row['Absolute DD']
                            dfback.loc[index, 'Average Loss'] = row['Average Loss']
                            dfback.loc[index, 'Lots'] = row['Lots']
                            pass
                        except IndexError:
                            pass
                    print('Done Backtest Results for:', pair, time_frame)

                    #------------------------------------------------------------FORWARD FILE----------------------------------------------------------------------------------------------------------------------------
                    csv_file_name_forward = 'OptiResults-{}-{}-{}-Phase1.forward.csv'.format(self.bot, pair, time_frame)
                    csv_file_name_forward = os.path.join(REPORT_PATH, self.bot, pair, time_frame, csv_file_name_forward)

                    csv_list_forward = self.get_csv_list_forward(pair, time_frame)
                    dfforward = pd.DataFrame(data=csv_list_forward)
                    dfforward = dfforward.drop(dfforward.index[:null_values_index]) # 4 variables hacen 13 9
                    dfforward_columns_name = csv_list_forward[null_values_columns] # 4 variables hacen 12
                    dfforward.columns = dfforward_columns_name
                    dfforward['Forward Lots'] = 0.1
                    dfforward['Forward Average Loss'] = dfforward['Forward Result']
                    dfforward['Forward Win Ratio'] = dfforward['Forward Result']
                    dfforward['Forward Average Loss'] = dfforward['Forward Average Loss'].str.slice(start=-6, stop=-3)
                    dfforward['Forward Win Ratio'] = dfforward['Forward Win Ratio'].str.slice(start=-3)
                    dfforward = dfforward.apply(pd.to_numeric)
                    dfforward['Forward Absolute DD'] = dfforward['Profit'] / dfforward['Recovery Factor']
                    dfforward = self.movecol(dfforward, ['Forward Absolute DD'], 'Equity DD %')
                    dfforward = self.movecol(dfforward, ['Forward Average Loss'], 'Equity DD %', place='Before')
                    dfforward = self.movecol(dfforward, ['Forward Win Ratio'], 'Trades')
                    dfforward = self.movecol(dfforward, ['Forward Lots'], 'Forward Win Ratio')
                    dfforward = dfforward.apply(pd.to_numeric)
                    dfforward.sort_values(by=['Back Result'], ascending=False, inplace=True)
                    dfforward.reset_index(inplace=True)
                    dfforward.to_csv(csv_file_name_forward, sep=',', index=False)
                    dfforward.rename(columns={'Profit':'Forward Profit', 'Expected Payoff':'Forward Expected Payoff', 'Profit Factor':'Forward Profit Factor', 'Recovery Factor':'Forward Recovery Factor', 'Sharpe Ratio':'Forward Sharpe Ratio', 'Custom':'Forward Custom', 'Equity DD %':'Forward Equity DD %', 'Trades':'Forward Trades'}, inplace=True)
                    print('Done Forward Results for:', pair, time_frame)

                    #---------------------------------------------------------------------------------Join DATAFRAMES---------------------------------------------------------------------------------------

                    csv_file_name_complete = 'OptiResults-{}-{}-{}-Phase1.Complete.csv'.format(self.bot, pair, time_frame)
                    csv_file_name_complete = os.path.join(REPORT_PATH, self.bot, pair, time_frame, csv_file_name_complete)

                    csv_file_name_complete_filtered = 'OptiResults-{}-{}-{}-Phase1.Complete-Filtered.csv'.format(self.bot, pair, time_frame)
                    csv_file_name_complete_filtered = os.path.join(REPORT_PATH, self.bot, pair, time_frame, csv_file_name_complete_filtered)

                    complete_df = pd.concat([dfback, dfforward], axis=1)
                    complete_df = complete_df.loc[:, ~complete_df.columns.duplicated()]
                    complete_df = self.movecol(complete_df, ['Forward Result'], 'Result')
                    complete_df = self.movecol(complete_df, ['Forward Profit'], 'Profit')
                    complete_df = self.movecol(complete_df, ['Forward Expected Payoff'], 'Expected Payoff')
                    complete_df = self.movecol(complete_df, ['Forward Profit Factor'], 'Profit Factor')
                    complete_df = self.movecol(complete_df, ['Forward Recovery Factor'], 'Recovery Factor')
                    complete_df = self.movecol(complete_df, ['Forward Sharpe Ratio'], 'Sharpe Ratio')
                    complete_df = self.movecol(complete_df, ['Forward Custom'], 'Custom')
                    complete_df = self.movecol(complete_df, ['Forward Average Loss'], 'Average Loss')
                    complete_df = self.movecol(complete_df, ['Forward Equity DD %'], 'Equity DD %')
                    complete_df = self.movecol(complete_df, ['Forward Absolute DD'], 'Absolute DD')
                    complete_df = self.movecol(complete_df, ['Forward Trades'], 'Trades')
                    complete_df = self.movecol(complete_df, ['Forward Win Ratio'], 'Win Ratio')
                    complete_df = self.movecol(complete_df, ['Forward Lots'], 'Lots')
                    complete_df = complete_df.drop(['Back Result'], axis=1)
                    complete_df.to_csv(csv_file_name_complete, sep=',', index=False)

                    #---------------------------------------------------------------------------------AFTER UNION FILTER---------------------------------------------------------------------------------------
                    print('Before filtering the len of Complete DataFrame for', pair, time_frame, 'is:', len(complete_df))
                    for index, row in complete_df.iterrows():
                        total_count += 1
                        avg_loss_norm = 100 / row['Average Loss']
                        absolute_dd_norm = float(self.dto.filter_equitity_dd_phase1) / row['Absolute DD']
                        normalize_row = float(min(avg_loss_norm, absolute_dd_norm))
                        row['Lots'] = float(round(row['Lots'] * normalize_row, 2))
                        lot_ratio = row['Lots'] / 0.1
                        row['Forward Lots'] = row['Lots']
                        row['Forward Profit'] = row['Forward Profit'] * lot_ratio
                        row['Forward Expected Payoff'] = row['Forward Expected Payoff'] * lot_ratio
                        row['Forward Absolute DD'] = row['Forward Absolute DD'] * lot_ratio
                        row['Forward Average Loss'] = row['Forward Average Loss'] * lot_ratio
                        complete_df.loc[index, 'Forward Profit'] = row['Forward Profit']
                        complete_df.loc[index, 'Forward Expected Payoff'] = row['Forward Expected Payoff']
                        complete_df.loc[index, 'Forward Absolute DD'] = row['Forward Absolute DD']
                        complete_df.loc[index, 'Forward Average Loss'] = row['Forward Average Loss']
                        complete_df.loc[index, 'Forward Lots'] = row['Forward Lots']

                        try:
                            if (row['Profit'] >= int(FilterNetProfitPhase1.get())) and (row['Expected Payoff'] >= float(FilterExpectedPayoffPhase1.get())) and (row['Profit Factor'] >= float(FilterProfitFactorPhase1.get())) and (row['Custom'] >= float(FilterCustomPhase1.get())) and ((row['Absolute DD']) <= float(self.dto.filter_equitity_dd_phase1)+100) and (row['Trades'] >= int(FilterTradesPhase1.get())) and \
                                (row['Forward Profit'] >= int(ForwardFilterNetProfitPhase1.get())) and (row['Forward Expected Payoff'] >= float(ForwardFilterExpectedPayoffPhase1.get())) and (row['Forward Profit Factor'] >= float(ForwardFilterProfitFactorPhase1.get())) and (row['Forward Custom'] >= float(ForwardFilterCustomPhase1.get())) and ((row['Forward Absolute DD']) <= float(ForwardFilterEquityDDPhase1.get())+100) and (row['Forward Trades'] >= int(ForwardFilterTradesPhase1.get())):
                                project_count += 1
                                pass
                            else:
                                complete_df.drop(labels=index, inplace=True)

                        except IndexError:
                            pass

                    print('After filtering the len of Filtered DataFrame from Phase 2 for', pair, time_frame, 'is:', len(complete_df))
                    #complete_df = complete_df.drop_duplicates(subset='Profit',inplace=True) drop duplicates attempt
                    print('After Removing Duplicates for', pair, time_frame, ' the len is: unknown')
                    try:
                        complete_df.drop(columns=['index'], inplace=True)
                    except KeyError:
                        pass
                    complete_df.to_csv(csv_file_name_complete_filtered, sep=',', index=False)
                    print("Filtered Dataframe saved")
                except FileNotFoundError:
                    pass
        end = time.time()
        time_result = (end - start) / 60
        project_ratio = (project_count/total_count)*100
        print('Filtered by:', '\n', \
            'Forward Min. Net Profit:', ForwardFilterNetProfitPhase1.get(), '\n', \
            'Forward Min. Exp. Payoff:', ForwardFilterExpectedPayoffPhase1.get(), '\n', \
            'Forward Min. Profit Factor:', ForwardFilterProfitFactorPhase1.get(), '\n', \
            'Forward Min. Custom:', ForwardFilterCustomPhase1.get(), '\n', \
            'Forward Max. Equity DD:', ForwardFilterEquityDDPhase1.get(), '\n', \
            'Forward Min. Trades:', ForwardFilterTradesPhase1.get(), '\n')
        print('From a Total of :', total_count, 'backtests')
        print('Only', project_count, ' passed the filters.', round(project_ratio, ndigits=2), '%')
        print('Phase 2 Results Accotated in', round(time_result), 'minutes')

    def get_csv_list_back(self, pair, time_frame):
        tree = et.parse(os.path.join(REPORT_PATH, self.bot, pair, time_frame, 'OptiResults-{}-{}-{}-Phase1.xml'.format(self.bot, pair, time_frame)))
        return self.get_csv_list(tree.getroot())

    def get_csv_list_forward(self, pair, time_frame):
        tree = et.parse(os.path.join(REPORT_PATH, self.bot, pair, time_frame, 'OptiResults-{}-{}-{}-Phase1.forward.xml'.format(self.bot, pair, time_frame)))
        return self.get_csv_list(tree.getroot())

    def get_csv_list(self, root):
        csv_list = []
        for child in root:
            for section in child:
                for row in section:
                    row_list = []
                    csv_list.append(row_list)
                    for cell in row:
                        for data in cell:
                            row_list.append(data.text)

        return csv_list


    def movecol(self, df, cols_to_move=[], ref_col='', place='After'):
        cols = df.columns.tolist()
        if place == 'After':
            seg1 = cols[:list(cols).index(ref_col) + 1]
            seg2 = cols_to_move
        if place == 'Before':
            seg1 = cols[:list(cols).index(ref_col)]
            seg2 = cols_to_move + [ref_col]

        seg1 = [pair for pair in seg1 if pair not in seg2]
        seg3 = [pair for pair in cols if pair not in seg1 + seg2]

        return df[seg1 + seg2 + seg3]

