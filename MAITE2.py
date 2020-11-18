import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.optimizers import SGD
import sklearn
from sklearn import preprocessing
import numpy as np
import pandas as pd

# Define el nombre del Portafolio y el Numero de estrategias que tiene
PortfolioName = 'AATROX'
TotalStrategies = 8
Base_Location_html = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/Portfolios/{}.html'.format(PortfolioName)
TradesFeed = pd.read_csv('C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/Portfolios/{}.csv'.format(PortfolioName))  # previo a anexar las ordenes de las estrategias es de 2080 x 21, deberia terminar siendo 2080 x (21 + TotalStrategies)

# Lee el HTML de la estrategia y el CSV con la lista de movimientos

# Transforma el HTML en DFs para poder manejarlos
dfs = pd.read_html(Base_Location_html, header=0)
df_Strats = dfs[0].copy(deep=True)
df_Strats = df_Strats.drop(df_Strats.index[TotalStrategies:])
df_Months = dfs[1].copy(deep=True)
df_PortfolioStats = dfs[2].copy(deep=True)
df_TradeStats = dfs[3].copy(deep=True)

# Obtiene el Net Profit del Portafolio antes de pasar por la IA
BasicBruteProfit = df_TradeStats.iloc[0,1].strip('$')
BasicBruteLoss =  df_TradeStats.iloc[0,3].strip('$')
BasicNetProfit = round(float(BasicBruteProfit) + float(BasicBruteLoss),ndigits=2)

# Define List of Strategy Numbers
StratInputNeuron = df_Strats['#']
StratInputNeuron = StratInputNeuron.str.strip('S')
StratInputNeuron = StratInputNeuron.tolist()
StratInputNeuron = [int(i) for i in StratInputNeuron]

# Define List of Strategy Normal Lot Sizes
StratLotSizeNeuron = df_Strats['#']

def Extract_Order(PortfolioName):
    AllTrades = TradesFeed.copy(deep=True)
    OrderedSXdf = pd.DataFrame()
    for row, index in df_Strats.iterrows():
        Strategy_csv = df_Strats['#'][row]
        StratFileName = str(PortfolioName + '-' + Strategy_csv)
        StratFilePath = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/Portfolios/{}.csv'.format(StratFileName)
        with open(StratFilePath) as f_csv:
            SXdf = pd.read_csv(StratFilePath)
            SXdf['StratOrder'] = Strategy_csv
            SXdf = SXdf.filter(items=['StratOrder', 'Order (Global)'])
            OrderedSXdf = pd.concat([OrderedSXdf, SXdf])
    OrderedSXdf.sort_values(by=['Order (Global)'], ascending=True, inplace=True)
    OrderedSXdf.reset_index(drop=True, inplace=True)
    #OrderedSXdf.to_csv('C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/Portfolios/Ordered-{}.csv'.format(PortfolioName), sep=',', index=False)
    AllTrades = AllTrades.join(OrderedSXdf, lsuffix='St', how='inner')
    AllTrades.to_csv('C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/Portfolios/Global-{}.csv'.format(PortfolioName),sep=',', index=False)
    return AllTrades

# Dictionary that goes as input layer
Input_Dict = {'S1':0.18,'S2':0.25,'S3':0.18,'S4': 0.19,'S5':0.23,'S6':0.13,'S7':0.18,'S8':0.16,'S9':0.14}
nein =1

#def reduce_lots(StratOrder,MinSize=0.01):
    #"""Reduce Lot size to 0.01 Lots"""
    #Xdf =

#print(StratInputNeuron)
def showalldfs():
    print('this is DF0:',dfs[0])
    print('this is DF1:',dfs[1])
    print('this is DF2:',dfs[2])
    print('this is DF3:',dfs[3])
    print('this is DF4:',dfs[4])
    print('this is DF5:',dfs[5])

#showalldfs()
#Extract_Order()


# Podria intentar crear un switch aqui, con funcion de Tanh -1 bajo el lotaje, entre -0.5 y 0.5 no hago nada y + de 0.5 devuelvo el lotaje al monto normal

    """List of Strategies that can be taken, total of 3."""
def ReduceLots(row,StratOrder,MinSize=0.01):
    """Reduce Lot size to 0.01 Lots"""
    RowFactor = 0.01/row['Size']
    print('Reduced Lot Size for:',StratOrder)
    return RowFactor

def DoNothing():
    """Does not change Lot Size"""
    print('No action taken')

def IncreaseLots(row,StratOrder,NormSize):
    """Returns Lot Size back to it's standard amount"""
    print('Lot Size returned to normality for:',StratOrder)



def ResizeLosses(SizeFactor,PL_Global):
    """Resizes Lossing trades profit"""
    ResizePL = SizeFactor * PL_Global
    return ResizePL

def MAITE():

    """Transforma todo el Dataframe a numeros para que sea mas facil procesarlo"""
    AllTradesFeed = pd.read_csv('C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/Portfolios/Global-{}.csv'.format(PortfolioName))
    Z = pd.DataFrame(AllTradesFeed)

    #print('Example of Dataframe:',Z.iloc[0,:])
    le = preprocessing.LabelEncoder()
    #Ticket = le.fit_transform(list(AllTradesFeed['Ticket']))
    Symbol = le.fit_transform(list(AllTradesFeed['Symbol']))
    Type = le.fit_transform(list(AllTradesFeed['Type']))
    Open_Time = le.fit_transform(list(AllTradesFeed['Open time']))
    #Order_Global = le.fit_transform(list(AllTradesFeed['Order (Global)St']))
    #Open_Price = le.fit_transform(list(AllTradesFeed['Open price']))
    #Size = le.fit_transform(list(AllTradesFeed['Size']))
    Close_Time = le.fit_transform(list(AllTradesFeed['Close time']))
    #Close_Price = le.fit_transform(list(AllTradesFeed['Close price']))
    Time_in_trade = le.fit_transform(list(AllTradesFeed['Time in trade']))
    #PL_Global = le.fit_transform(list(AllTradesFeed['Profit/Loss (Global)']))
    #Balance_Global = le.fit_transform(list(AllTradesFeed['Balance (Global)']))
    #Comm_Swap = le.fit_transform(list(AllTradesFeed['Comm/Swap']))
    #PL = le.fit_transform(list(AllTradesFeed['Profit/Loss ($)']))
    #Balance = le.fit_transform(list(AllTradesFeed['Balance ($)']))
    #PL_pips = le.fit_transform(list(AllTradesFeed['Profit/Loss (Pips)']))
    #Balance_pips = le.fit_transform(list(AllTradesFeed['Balance (Pips)']))
    #PL_percent = le.fit_transform(list(AllTradesFeed['Profit/Loss (%)']))
    #Balance_percent = le.fit_transform(list(AllTradesFeed['Balance (%)']))
    Comment = le.fit_transform(list(AllTradesFeed['Comment']))
    Sample_type = le.fit_transform(list(AllTradesFeed['Sample type']))
    StratOrder = le.fit_transform(list(AllTradesFeed['StratOrder']))
    #Order_Global = le.fit_transform(list(AllTradesFeed['Order (Global)']))

    #X= list(zip(Ticket,Symbol,Type,Open_Time,Order_Global,Open_Price,Size,Close_Time,Close_Price,Time_in_trade,PL_Global,Balance_Global,Comm_Swap,PL,Balance_pips,PL_percent,Balance_percent,Comment,Sample_type,StratOrder))
    X= list(zip(AllTradesFeed['Ticket'],Symbol,Type,Open_Time,AllTradesFeed['Order (Global)St'],AllTradesFeed['Open price'],
                AllTradesFeed['Size'],Close_Time,AllTradesFeed['Close price'],Time_in_trade,AllTradesFeed['Profit/Loss (Global)'],
                AllTradesFeed['Balance (Global)'],AllTradesFeed['Comm/Swap'],AllTradesFeed['Profit/Loss ($)'],AllTradesFeed['Balance ($)'],
                AllTradesFeed['Profit/Loss (Pips)'],AllTradesFeed['Balance (Pips)'],AllTradesFeed['Profit/Loss (%)'],AllTradesFeed['Balance (%)'],
                Comment,Sample_type,StratOrder,AllTradesFeed['Order (Global)']))
    #for i in AllTradesFeed['Balance (Global)']:
        #print(i)
    #Xdf = pd.DataFrame(data=X,columns=list(Z))
    Xdf = pd.DataFrame(data=X)
    #print(Xdf)
    Lcount = 0
    """Iteracion completa a traves del DataFrame, logrando ejecutar cambio de lotaje y recalculo."""
    #for index,row in Xdf.iterrows():       # ya itera por fila este modulo, faltaria agregar condicionales
        #if row['Profit/Loss ($)'] < int(0):
            #Factor = ReduceLots(row,row['StratOrder'])
            #ResizedLost = ResizeLosses(Factor,row['Profit/Loss (Global)'])
            #NewBalance = (row['Balance (Global)'] - row['Profit/Loss (Global)']) + ResizedLost
            #print('Here it reduced lots from Strategy #',int(row['StratOrder']),row['Size'],'to 0.01 Lots for a Factor of',ReduceLots(row,row['StratOrder']))
            #print('It took losses from',row['Profit/Loss (Global)'],'to ',ResizedLost)
            #print('Drove the Balance from',row['Balance (Global)'],'to ',NewBalance)
            #Lcount +=1
    #print(Lcount)

            #Xdf[i, 'Profit/Loss ($)'] > 0
            #print(Xdf.loc[i,'Profit/Loss ($)'])
        #print(i)

    """Comienza codigo de modelo de Actor/Critico """
    #ActionMap = [ReduceLots(),DoNothing(),IncreaseLots()]
    ActionMap = 3
    inputs = layers.Input(shape=Xdf,name='Input_Layer')
    common = layers.Dense(64,name='Hidden_Layer')(inputs)
    actor = layers.Dense(3,name='Actor_Layer')(common)
    critic = layers.Dense(1,name='Critic_Layer')(common)
    model = keras.Model(inputs=inputs,outputs=[actor,critic])
    optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False,name='Optimizer_SGD')
    model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy') #cambiar la loss function, puede ser una lista de varias funciones, colocar aqui comparacion de Portafolio actual vs el resultado del epoch
    model.fit(batch_size=1,steps_per_epoch=1)


    #print(inputs,common,actor,critic)
    model.summary()

    #model.fit()
Extract_Order('AATROX')
MAITE()

