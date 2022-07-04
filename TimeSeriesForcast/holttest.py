import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing


data = pd.read_csv('data.csv', parse_dates=True, index_col='Month')
data = data.rename(columns={
    "International airline passengers: monthly totals in thousands. Jan 49 ? Dec 60": "Value"})
data.dropna(inplace=True)
data.index.freq = 'MS'

traindata = data[:120]
testdata = data[120:]

fitted = ExponentialSmoothing(
    traindata['Value'], trend='mul', seasonal='mul', seasonal_periods=12).fit()
pred = fitted.forecast(24)
# traindata['Value'].plot(legend=True, label='Train')
# testdata['Value'].plot(legend=True, label='Test', figsize=(6, 4))
# pred.plot(legend=True, label='Predictions')
# plt.title('Predicted values by Holt Winters')
# plt.savefig('holt_all.png')


testdata['Value'].plot(legend=True, label='Test', figsize=(9, 6))
pred.plot(legend=True, label='Predictions', xlim=['1959-01-01', '1961-01-01'])
plt.savefig('holt_predicted.png')
