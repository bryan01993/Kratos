import pandas as pd

CSVTOTAL = 'C:/Users/bryan/OneDrive/Desktop/Prueba de Seleccion/TOTAL-OptiResults-EA-T1v2-EURGBP-M5-Phase1.csv'
dfback = pd.read_csv('C:/Users/bryan/OneDrive/Desktop/Prueba de Seleccion/OptiResults-EA-T1v2-EURGBP-M5-Phase1.csv')
dfforward = pd.read_csv('C:/Users/bryan/OneDrive/Desktop/Prueba de Seleccion/OptiResults-EA-T1v2-EURGBP-M5-Phase1.forward.csv')
dfback.sort_values(by=['Profit'],ascending=False,inplace=True)
dfback['Rank'] = range(0,len(dfback))
dfforward.sort_values(by=['Profit'],ascending=False,inplace=True)
dfforward['Rank Forward'] = range(0,len(dfforward))
dffull = dfback.join(dfforward,on=dfback['Pass'],rsuffix='Forward')
dffull['Total Score'] = dffull['Rank'] + (dffull['Rank Forward'])*3
dffull.to_csv(CSVTOTAL, sep=',', index=False)
selection = dffull['Total Score'].min()

newdf = dffull[(dffull['Total Score'] == dffull['Total Score'].min())]

print(newdf)

print(selection)
#print(dfback)
#print(dfforward)