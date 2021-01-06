import numpy as np
import pandas as pd
from scipy.fft import ifftn, fft,dct
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.rcParams['figure.figsize'] = [16,12]
plt.rcParams.update({'font.size':18})
# Import Data
df = pd.read_csv('C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/Moded Data/DayData.csv')
#df = pd.read_csv('C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/Data/EURUSDMT5.csv')

df['Audio'] = df['<CLOSE>'] - df['<OPEN>']
df['Audio+'],df['Audio-'] = df['Audio'],df['Audio']

df['HAclose'] = (df['<OPEN>'] + df['<HIGH>'] + df['<LOW>'] + df['<CLOSE>']) /4
df['HAopen'] = (df['<OPEN>'].shift(1,fill_value=0) + df['<CLOSE>'].shift(1,fill_value=0)) /2
df['HAaudio'] = df['HAclose'] - df['HAopen']
df['HAhigh'],df['HAlow'] = df['<HIGH>'] , df['<LOW>']
df['fechayhora'] = df['<DATE>'] + " " + df['<TIME>']
df['HAaudio+'],df['HAaudio-'] = df['HAaudio'],df['HAaudio']

df['CloseMA'] = (df['<CLOSE>']+ df['<CLOSE>'].shift(1) + df['<CLOSE>'].shift(2) + df['<CLOSE>'].shift(3) + df['<CLOSE>'].shift(4) + df['<CLOSE>'].shift(5) + df['<CLOSE>'].shift(6))/7
df = df.dropna()
df.loc[df['Audio+'] <0, 'Audio+'] = 0
df.loc[df['Audio-'] >0, 'Audio-'] = 0
df.loc[df['HAaudio+'] <0, 'HAaudio+'] = 0
df.loc[df['HAaudio-'] >0, 'HAaudio-'] = 0


df['index'] = df['fechayhora'].index
df.loc[df['HAaudio'] > 0.5, 'HAaudio'] = 0
df.loc[df['HAaudio+'] > 0.5, 'HAaudio+'] = 0
df.loc[df['HAaudio-'] < -0.5, 'HAaudio-'] = 0

def MaxClose(period):
    Serie = df['<CLOSE>']




ypoints= np.array(df['CloseMA'])
xpoints=np.array(df['fechayhora'])
xpointsindex = np.array(df['index'])
print(df)

df.to_csv('C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/CSV MODELOS/MODEURUSDMT5TRAINHA.csv', sep=',', index=False)

#Z = ifftn(ypoints)
#plt.plot(xpoints,ypoints)
#plt.show()

dt = 1
t = np.arange(0,len(xpointsindex),dt)
f = ypoints
f_clean = f


n = len(t)
fhat = np.fft.fft(f,n)                        # Compute the FFT
PSD = fhat * np.conj(fhat) / n                # Power Spectrum
freq = (1/(dt*n)) *np.arange(n)               # Create X axis for frequency
L = np.arange(1,np.floor(n/2),dtype='int')    #only plot the first half of ?????




indices = PSD > 0.01      #valor que le agregamos en las graficas del eje Y de la frecuencia
PSDclean = PSD * indices      #zero all other values
fhat = indices * fhat         #zero all small Fouriers coeff in Y
ffilt = np.fft.ifft(fhat)     #Inverse FFT for filtered time signal

fig,axs = plt.subplots(3,1)

plt.sca(axs[0])
plt.plot(t,f,color='c',LineWidth=1.5,label='Noisy')
plt.plot(t,f_clean,color='k',LineWidth=2,label='Clean')
plt.xlim(t[0],t[-1])
plt.legend()

plt.sca(axs[1])
plt.plot(t,ffilt,color='k',LineWidth=2,label='Filtered')
plt.xlim(t[0],t[-1])
plt.legend()

plt.sca(axs[2])
plt.plot(freq[L],PSD[L],color='c',LineWidth=2,label='Noisy')
plt.xlim(freq[L[0]],freq[L[-1]])
plt.legend()

plt.show()
