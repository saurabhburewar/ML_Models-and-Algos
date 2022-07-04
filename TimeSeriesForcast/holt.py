import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller


# # Read data
df = pd.read_csv('data.csv', parse_dates=True, index_col='Month')
df = df.rename(columns={
               "International airline passengers: monthly totals in thousands. Jan 49 ? Dec 60": "Value"})
df.dropna(inplace=True)

result = adfuller(df)


print("ADF test statistic: ", result[0])
print("p-value: ", result[1])
print("Number of lags used: ", result[2])
print("Number of obs used: ", result[3])

print(result)
print(df.head())

df.sort_index(inplace=True)

deresult = seasonal_decompose(df['Value'], model='multiplicative', period=1)
deresult.plot()
plt.savefig('holt1.png')

x = 12
alpha = 1/(2*x)

df['HWES1'] = SimpleExpSmoothing(df['Value']).fit(
    smoothing_level=alpha, optimized=False, use_brute=True).fittedvalues
df[['Value', 'HWES1']].plot(title='Holt_singlesmooth')
plt.savefig('Holt_singlesmooth.png')


df['HWES2add'] = ExponentialSmoothing(
    df['Value'], trend='add').fit().fittedvalues
df['HWES2mul'] = ExponentialSmoothing(
    df['Value'], trend='mul').fit().fittedvalues

df[['Value', 'HWES2add', 'HWES2mul']].plot(title='Holt_addandmul')
plt.savefig('Holt_addandmul.png')
