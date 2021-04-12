import pandas as pd
import os
import pxl
import openpyxl as pxl
from openpyxl import load_workbook
from pandas import ExcelWriter

botname = 'TendencialNuevo'
pairlist = ['GBPUSD', 'EURUSD', 'USDCAD', 'USDCHF', 'USDJPY', 'GBPJPY', 'EURAUD', 'EURGBP', 'EURJPY', 'EURCHF']
timeframes = [ 'H4', 'H1', 'M30', 'M15','M5']

FOLDER_PATH = "C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028"
REPORT_PATH = os.path.join(FOLDER_PATH, 'reports')
savepath = pd.ExcelWriter('C:/Users/bryan/OneDrive/Desktop/Prueba de Seleccion/{}.xlsx'.format(botname))

def verify_file(file):
    if 'Complete' in file and 'Filtered' not in file:
        return file
    else:
        return None

def verify_date_range_in_dataframe(DateRange,dataframe):
    if DateRange not in list(dataframe):
        list(dataframe).append(DateRange)
        dataframe[DateRange] = 0
    else:
        pass

def pack_values(positiveIS,positiveOOS,positivebayes):
    packed = (positiveIS,positiveOOS,positivebayes)
    return packed

def all_sheets(Calculo, timeframes):
    ALL_SHEETS = []
    for calc in Calculo:
        for timeframe in timeframes:
            add_value = '{}-{}'.format(calc,timeframe)
            ALL_SHEETS.append(add_value)
    print(ALL_SHEETS)
    return ALL_SHEETS
def run_all():
    Calculos = ['Positivo IS', 'Positivo OOS', 'Positivo Bayes']

    Optisheet1 = pd.DataFrame(dtype='float',index=pairlist, columns=["Pair"])
    Optisheet1['Pair'] = pairlist
    Optisheet1.to_excel('C:/Users/bryan/OneDrive/Desktop/Prueba de Seleccion/{}.xlsx'.format(botname))
    Optisheet2 = Optisheet1.copy()
    Optisheet3 = Optisheet1.copy()
    for pair in pairlist:
        for timeframe in timeframes:
            filepath = os.path.join(REPORT_PATH, botname, pair, timeframe, 'WF_Report')
            try:
                for file in os.listdir(filepath):
                    selected_file = verify_file(file)
                    if selected_file != None:
                        file = os.path.join(filepath, selected_file)
                        df_complete = pd.read_csv(file)
                        file_period = selected_file[-30:-13]
                        verify_date_range_in_dataframe(dataframe=Optisheet1,DateRange=file_period)
                        df_complete.sort_values(by=['Result'], ascending=False, inplace=True)
                        df_backpositive = df_complete[df_complete['Profit'] > 0]
                        df_backpositiveresult = round((len(df_backpositive)/len(df_complete))* 100, ndigits=1)
                        df_forwardpositive = df_complete[df_complete['ProfitForward'] > 0]
                        df_fwdpositiveresult = round((len(df_forwardpositive) / len(df_complete)) * 100, ndigits=1)
                        bayespositive = round((len(df_backpositive)/len(df_complete))* 100, ndigits=1)          # no esta formulado correctamente, es meramente para propositos de organizar las cosas y hacer pruebas.
                        Optisheet1.at[pair,file_period] = df_backpositiveresult
                        Optisheet2.at[pair,file_period] = df_fwdpositiveresult
                        Optisheet3.at[pair,file_period] = bayespositive

                excel_book = pxl.load_workbook(savepath)
                with ExcelWriter(savepath, engine='openpyxl') as writer:
                    print('Saving Results on {}-{}'.format(pair,timeframe))
                    writer.book = excel_book
                    writer.sheets = {worksheet.title: worksheet for worksheet in excel_book.worksheets}
                    Optisheet1['mean'] = Optisheet1.mean(axis=1, numeric_only=True)
                    Optisheet2['mean'] = Optisheet2.mean(axis=1, numeric_only=True)
                    Optisheet3['mean'] = Optisheet3.mean(axis=1, numeric_only=True)
                    #period_mean1 = Optisheet1[file_period].mean()
                    #period_mean2 = Optisheet2[file_period].mean()
                    #period_mean3 = Optisheet3[file_period].mean()
                    #Optisheet1.at['Cross-Mean',file_period] = period_mean1
                    #Optisheet2.at['Cross-Mean', file_period] = period_mean2
                    #Optisheet3.at['Cross-Mean', file_period] = period_mean3
                    Optisheet1.at['Cross-Mean'] = Optisheet1.mean(axis=0)
                    Optisheet2.at['Cross-Mean'] = Optisheet2.mean(axis=0)
                    Optisheet3.at['Cross-Mean'] = Optisheet3.mean(axis=0)
                    #Optisheet1.describe()
                    Optisheet1.to_excel(writer,sheet_name=' {}-{}'.format(Calculos[0],timeframe))
                    Optisheet2.to_excel(writer, sheet_name=' {}-{}'.format(Calculos[1],timeframe))
                    Optisheet3.to_excel(writer, sheet_name=' {}-{}'.format(Calculos[2],timeframe))
                writer.save()
                writer.close()
            except FileNotFoundError:
                print('This pair: {} and timeframe {} has no file.'.format(pair,timeframe))



run_all()


"""PASOS
1) crear los 3 dataframes vacios y llenarles las columnas (periodos) y filas(pares) LISTO
1.5) 1er Dataframe para %positivoIS, 2do dataframe para #positivoOOS, 3er dataframe para %Bayespositivo top 10%. LISTO
 2) al iterar en un timeframe guardar todos los resultados en la pestana adecuada"""