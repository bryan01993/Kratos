import pandas as pd
import os.path
Meta5DataFolder= 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/Data'
DataCount=0
def CleanData(file):
 """Eliminates midnight candles were Errors were found."""
 df = pd.read_csv('C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/Data/{}'.format(file),delimiter='\t')
 df = df.drop(['<TICKVOL>'],axis=1)
 df['<VOL>'] = 1234
 df['<SPREAD>']= df['<SPREAD>']*1.30
 df = df[(df != '00:00:00')]
 df = df[df['<TIME>'].notna()]
 df.to_csv('C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/Moded Data/MOD{}'.format(file), sep=',', index=False)
 print('data for :',file,' cleansed')
for file in os.listdir(Meta5DataFolder):
 print('Processing Now:',file)
 CleanData(file)
 DataCount += 1
print('Done the Data', DataCount,' Files were processed.')
