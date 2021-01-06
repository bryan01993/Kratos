import pandas as pd
import os

pairlist = ['GBPUSD', 'EURUSD', 'USDCAD', 'USDCHF', 'USDJPY', 'GBPJPY', 'EURAUD', 'EURGBP', 'EURJPY', 'EURCHF']

for pair in pairlist:
    dfback = pd.read_csv('C:/Users/bryan/OneDrive/Desktop/Prueba de Seleccion/OptiResults-Simple-EA-MeanReversal-{}-H4-Phase1.csv'.format(pair))
    dfforward = pd.read_csv('C:/Users/bryan/OneDrive/Desktop/Prueba de Seleccion/OptiResults-Simple-EA-MeanReversal-{}-H4-Phase1.forward.csv'.format(pair))

    dfbackpositive =dfback[dfback['Profit'] > 0]
    ProbISPositive = round((len(dfbackpositive)/ len(dfback))*100,2)

    dfforwardpositive = dfforward[dfforward['Profit'] > 0]
    ProbOOSPositive = round((len(dfforwardpositive)/ len(dfforward))*100,2)


    #print(dfback)
    print(pair)
    print(ProbISPositive,'% of IS is positive' )
    #print(dfforward)
    print(ProbOOSPositive,'% of OOS is positive' )