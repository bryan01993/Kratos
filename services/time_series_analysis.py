import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import statsmodels.tsa.stattools as tsa
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import numpy as np
from hurst import compute_Hc
from arch.unitroot import VarianceRatio
from sklearn.linear_model import LinearRegression

# Data in candles directory
data_dir = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/CSV MODELOS'
store_dir = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/CSV Output'
# Date Selection for graphics and comparison
csv_dir = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/CSV MODELOS/NZDJPYMT5_H4.csv'
csv_dir_data2 = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/CSV MODELOS/AUDJPYMT5_H4.csv'
start_date = '2007.01.01'
end_date = '2011.01.01'
effective_list = ['99%', '95%', '90%']


# Transform .csv data into a DataFrame

def plot_returns_histogram():
    for file in os.listdir(data_dir):
        file_dataframe = pd.read_csv(data_dir + '/' + file, sep='\t', index_col='<DATE>')
        file_dataframe['<RETURN>'] = (file_dataframe['<CLOSE>'] - file_dataframe['<CLOSE>'].shift(1,fill_value=0))/file_dataframe['<CLOSE>'].shift(1, fill_value=0)
        snipped_dataframe = file_dataframe[start_date:end_date]
        snipped_dataframe.iat[0,-1] = 0
        #snipped_dataframe['<RETURN>'].plot(figsize=(16, 6))
        snipped_dataframe['<RETURN>'].plot.hist(bins = 100)
        kurt = snipped_dataframe.kurt(axis=0)
        print('File Name:', file)
        print(kurt)
        plt.show()

def simple_plot(dfname):
    dfname.plot()
    plt.show()

def create_dataframe(tf='H4'):

    for file in os.listdir(data_dir):
        file_pair = file[0:6]
        file_timeframe = file[10:13]
        if 'EURUSD' in file_pair and tf in file_timeframe:
            analysis_dataframe = pd.read_csv(data_dir + '/' + file, sep= '\t', index_col=['<DATE>','<TIME>'])
            analysis_dataframe = analysis_dataframe.drop(columns=['<OPEN>','<HIGH>','<LOW>','<CLOSE>','<VOL>','<SPREAD>','<TICKVOL>'])

    for file in os.listdir(data_dir):
        file_pair = file[0:6]
        file_timeframe = file[10:13]
        if tf in file_timeframe:
            file_dataframe = pd.read_csv(data_dir + '/' + file, sep='\t', index_col=['<DATE>','<TIME>'])
            file_dataframe = file_dataframe.rename(columns={'<CLOSE>': file_pair})
            analysis_dataframe = analysis_dataframe.join(file_dataframe[file_pair], how='left', on=['<DATE>','<TIME>'])

            print('Head:', '\n', analysis_dataframe.head())
            print('Tail:', '\n', analysis_dataframe.tail())
            print('Finished')
    return analysis_dataframe


def plot_correlations(df):
    sns.heatmap(df, cmap='rocket')
    plt.show()

def calculate_noise(dfname,dfSeries,n_window=60):

    EfficiencyRatioNum = dfSeries.diff(n_window).abs()
    print(n_window,type(n_window))
    EfficiencyRatioDen = pd.Series.rolling(n_window,min_periods=n_window)
    EfficiencyRatio = EfficiencyRatioNum/EfficiencyRatioDen
    print(EfficiencyRatioNum)
    print(EfficiencyRatio)
    

from datetime import datetime

OPTI_START_DATE = "2007.01.01" #YYYY.MM.DD
OPTI_END_DATE = "2020.01.01" #YYYY.MM.DD
brick = 12
brick_IS = brick * 4
brick_OOS = brick
brick_REAL = brick
init_dates_list=[]

def split_date(date):
    date_to_split = date.split('.')
    date_year = int(date_to_split[0])
    date_month = int(date_to_split[1])
    date_day = int(date_to_split[2])
    date_list=[date_year,date_month,date_day]
    return(date_list)

def count_months(startdate=OPTI_START_DATE,enddate=OPTI_END_DATE):
    startdate=split_date(OPTI_START_DATE)
    enddate=split_date(OPTI_END_DATE)

    count_months_year = (enddate[0]-startdate[0])*12
    count_months_months = (enddate[1]-startdate[1])
    total_months = count_months_year + count_months_months
    print('years in months',count_months_year)
    print('months diff between dates',count_months_months)
    print('Total months difference:',total_months)
    return total_months

def cast_list_to_string_date(date):
    return str(date[0]) + '.' + str(date[1]) + '.' + str(date[2])

def add_init_cuts(start=OPTI_START_DATE,brick=brick,brick_IS=brick_IS,brick_OOS=brick_OOS,brick_REAL=brick_REAL,end=OPTI_END_DATE):
    start = split_date(OPTI_START_DATE)
    def add_months_Start(brickindex=0,brickvalue=brick,date=start):
        bricks=brickvalue*brickindex
        brick_year = bricks // 12
        year_added = date[0] + brick_year
        brick_month = bricks % 12
        month_added = date[1] + brick_month
        date = [year_added,month_added,1]
        return date

    def add_months_IS(brickvalue=brick_IS, date=start):
        bricks = brickvalue
        brick_year = bricks // 12
        year_added = date[0] + brick_year
        brick_month = bricks % 12
        month_added = date[1] + brick_month
        if month_added >= 13:
            month_added = month_added - 12
            year_added = year_added + 1
        date = [year_added, month_added, 1]
        return date

    def add_months_OOS(brickvalue=brick_OOS, date=start):
        bricks = brickvalue
        brick_year = bricks // 12
        year_added = date[0] + brick_year
        brick_month = bricks % 12
        month_added = date[1] + brick_month
        if month_added >= 13:
            month_added = month_added - 12
            year_added = year_added + 1
        date = [year_added, month_added, 1]
        return date

    def add_months_Real(brickvalue=brick_OOS, date=start):
        bricks = brickvalue
        brick_year = bricks // 12
        year_added = date[0] + brick_year
        brick_month = bricks % 12
        month_added = date[1] + brick_month
        if month_added >= 13:
            month_added = month_added - 12
            year_added = year_added + 1
        date = [year_added, month_added, 1]
        return date

    iterator = 0
    max_iterations = (count_months()-brick_IS + brick)/brick
    print('This is the number of steps trought the WF: ',round(max_iterations,ndigits=0)-1)
    print('This is what the iteration list would look like:')

    lists = []
    for i in range(int(max_iterations-1)):
        iterator += 1
        the_start_date = add_months_Start(i)
        the_IS_date = add_months_IS(date=the_start_date)
        the_OOS_date = add_months_OOS(date=the_IS_date)
        lists.append([
            cast_list_to_string_date(the_start_date),
            cast_list_to_string_date(the_IS_date),
            cast_list_to_string_date(the_OOS_date),
        ])

        print('start date',the_start_date,'forward date ',the_IS_date,'end date ',the_OOS_date)
    print('Max iterations are', iterator)

    return lists
data = pd.read_csv(csv_dir, sep ='\t',index_col=['<DATE>','<TIME>'])
data2 = pd.read_csv(csv_dir_data2, sep ='\t',index_col=['<DATE>','<TIME>'])
print('Data 1,',len(data))
print('Data 2,',len(data2))

snipped_data = data[start_date:end_date]
snipped_data2 = data2[start_date:end_date]
#print('Snipped 1,',snipped_data)
#print('Snipped 2,',snipped_data2)

k = 1 #lag
series = snipped_data['<CLOSE>']
series2 = snipped_data2['<CLOSE>']

def calculate_adfuller(dfseries,maxlag=k):
    """Used to reject or not reject inicial hipothesis on wheter a price series is or not mean-reversion, checks for stationarity"""
    fuller , pfuller, lags, nobs, ptable, icbest = tsa.adfuller(dfseries,maxlag=k)
    print(ptable)
    print('This is ADF statistic:', fuller)
    #for i in ptable: falta terminar esto
        #print(i[-1])
        #if fuller < i:
            #print('ADF Statistic value',fuller, 'is significant with ',j,' confidence')
            #break
    return fuller

def calculate_hurst(dfseries):
    """if Hurst < 0.5 favors mean reversion, if Hurst >0.5 favors trend, Hurst ~ 0.5 means a random walk"""
    hurst, constant, vals = compute_Hc(dfseries,kind='price')
    print('Hurst Exponent value is : ', hurst)
    return hurst

def calculate_variance_ratio(dfseries):
    """Calculates the variance ratio to Asses that the Hurst exponent is correctly calculated"""
    log_series = np.log2(dfseries)
    vratio = VarianceRatio(log_series)
    print(vratio)
    return vratio

def Ornstein_Uhlenbeck(dfseries): #Parece terminado, however be suspicious, no hay forma facil de verificar
    """Ornstein-Uhlenbeck process for finding halflife, the amount of observations it takes to reach back the mean,
    the shorter the better for a mean reversion strategy."""
    series = dfseries.values
    z_lag = np.roll(series, 1)
    z_lag[0] = 0
    z_ret = series - z_lag
    z_ret[0] = 0
    # adds intercept terms to X variable for regression
    z_lag2 = sm.add_constant(z_lag)
    model = sm.OLS(z_ret, z_lag2)
    res = model.fit()
    halflife = -np.log2(2) / res.params[1]
    print('This should be halflife in candles:', halflife)
    return halflife

def calculate_cointegration(depseries, indepseries, k = 1):
    """Checks for cointegration between two data series"""
    coint_data = pd.concat([depseries,indepseries],axis=1,keys=['<DEPSERIES>','<INDEPSERIES>'],join='outer')
    coint_data = coint_data.dropna()
    coint_test = tsa.coint(y0=coint_data['<DEPSERIES>'], y1=coint_data['<INDEPSERIES>'],maxlag=k)
    linear_model = sm.OLS(coint_data['<DEPSERIES>'],coint_data['<INDEPSERIES>'])
    res = linear_model.fit()
    coint_data['hedged'] = coint_data['<DEPSERIES>'] - (res.params[0] * coint_data['<INDEPSERIES>'])
    print('Cointegration Test Results: \n', coint_test)
    for i,j in zip(coint_test[2],effective_list):
        if coint_test[0] < i:
            res.summary()
            print('CADF Statistic value',coint_test[0], 'is significant with ',j,' confidence')
            print('Hedge ratio is: ', res.params[0])
            simple_plot(coint_data)
            simple_plot(coint_data['hedged'])
            break
    simple_plot(coint_data)
    simple_plot(coint_data['hedged'])
    print('These Assets do NOT cointegrate with at least ', j, ' confidence')

def calculate_cointegration_johansen(depseries, indepseries, k = 1):
    """Checks for cointegration between two or MORE data series"""
    coint_data = pd.concat([depseries,indepseries],axis=1,keys=['<DEPSERIES>','<INDEPSERIES>'],join='outer')
    coint_data = coint_data.dropna()
    coint_data.to_csv(store_dir + '/' + 'cointdata.csv')
    johansen_test = coint_johansen(coint_data, det_order = 0, k_ar_diff = k)
    print('Johansen Test')
    print('Trace Stat: ',johansen_test.trace_stat)
    print('Trace critical values: \n ', johansen_test.trace_stat_crit_vals)
    print('Max EigenVectors : ', johansen_test.max_eig_stat )
    print('Max EigenVectors critical values: \n', johansen_test.max_eig_stat_crit_vals)

def calculate_cointegration_johansen2(dataframe, k = 1):
    """Checks for cointegration between two or MORE data series"""
    coint_data = dataframe
    #coint_data = coint_data.dropna()
    #coint_data.to_csv(store_dir + '/' + 'cointdata.csv')

    johansen_test = coint_johansen(coint_data, det_order = 0, k_ar_diff = k)
    print('Johansen Test')
    print('Trace Stat: ',johansen_test.trace_stat)
    print('Trace critical values: \n ', johansen_test.trace_stat_crit_vals)
    print('Max EigenVectors : ', johansen_test.max_eig_stat )
    print('Max EigenVectors critical values: \n', johansen_test.max_eig_stat_crit_vals)
    print('EigenValues: \n', johansen_test.eig)


#calculate_adfuller(dfseries=series)
#calculate_hurst(dfseries=series)
#calculate_variance_ratio(dfseries=series)
#calculate_cointegration(depseries=series,indepseries=series2)
#Ornstein_Uhlenbeck(dfseries=series)
#full = create_dataframe()
#full = full.dropna()
#full2 = full.values
#calculate_cointegration_johansen2(full2)

def test():
    for file in os.listdir(data_dir + '/'):
        print(file)
        filepath = data_dir + '/' + file
        datafile = pd.read_csv(filepath, sep='\t', index_col=['<DATE>', '<TIME>'])
        datafile = datafile[start_date:end_date]
        series = datafile['<CLOSE>']
        calculate_adfuller(series)
        calculate_hurst(series)
        #calculate_variance_ratio(series)
        Ornstein_Uhlenbeck(series)

#OU_list = pd.DataFrame(columns=timebrick_list)

test()