import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def dataandmodel(x, df):

    trainsize = (int)(0.8 * len(df))

    dftrain = pd.DataFrame(df[0:trainsize])
    dftest = pd.DataFrame(df[trainsize:len(df)])

    dftrain2 = dftrain.dropna()
    Xtrain = dftrain2.iloc[:, 1:].values.reshape(-1, x)
    Ytrain = dftrain2.iloc[:, 0].values.reshape(-1, 1)

    linreg = LinearRegression()
    linreg.fit(Xtrain, Ytrain)

    theta = linreg.coef_.T
    intercept = linreg.intercept_
    dftrain2['Predicted'] = Xtrain.dot(
        linreg.coef_.T) + linreg.intercept_

    Xtest = dftest.iloc[:, 1:].values.reshape(-1, x)
    dftest['Predicted'] = Xtest.dot(linreg.coef_.T) + linreg.intercept_

    return [dftrain2, dftest, theta, intercept]


def AR(p, df):
    dftemp = df

    for i in range(1, p+1):
        dftemp['Shifted_values_%d' % i] = dftemp['Value'].shift(i)

    [dftrain2, dftest, theta, intercept] = dataandmodel(p, dftemp)
    # dftest[['Value', 'Predicted']].plot()

    RMSE = np.sqrt(mean_squared_error(
        dftest['Value'], dftest['Predicted']))

    print("THe RMSE is: ", RMSE, ", Value of p: ", p)

    return [dftrain2, dftest, theta, intercept, RMSE]


def MA(q, df):

    for i in range(1, q+1):
        df['Shifted_values_%d' % i] = df['Residuals'].shift(i)

    [dftrain2, dftest, theta, intercept] = dataandmodel(q, df)
    # dftest[['Residuals', 'Predicted']].plot()

    RMSE = np.sqrt(mean_squared_error(
        dftest['Residuals'], dftest['Predicted']))

    print("THe RMSE is: ", RMSE, ", Value of q: ", q)

    return [dftrain2, dftest, theta, intercept, RMSE]


# Read data
df = pd.read_csv('df.csv', parse_dates=True, index_col='date')
df = pd.DataFrame(df.groupby(df.index.strftime('%Y-%m')).sum()['amount'])
df.columns = ['Value']


# check if data is stationary
dftesting = pd.DataFrame(np.log(df.Value).diff().diff(12))

result = adfuller(dftesting.Value.dropna())
if result[1] <= 0.05:
    print("Data is stationary")
    stat = True
else:
    print("Data is not stationary")
    stat = False

# Differencing
dftesting = pd.DataFrame(df.Value)
d = 0
while not stat:
    dftesting = pd.DataFrame(dftesting.diff())
    d += 1

    dftesting = dftesting.Value.dropna()
    result = adfuller(dftesting)

    if result[1] <= 0.05:
        stat = True
    else:
        stat = False

print("ADF test statistic: ", result[0])
print("p-value: ", result[1])
print("Number of lags used: ", result[2])
print("Number of obs used: ", result[3])
# print("Degree of differencing: ", d)
# print("Data is now stationary")

ACF = plot_acf(dftesting.dropna(), lags=50)
# plt.show()


# AR model
bestE = 9999999999
bestp = -1
print("AR model - \n")
print(dftesting.head())
for i in range(1, 21):
    [artrain, artest, theta, intercept, RMSE] = AR(
        i, pd.DataFrame(dftesting.Value))
    if(RMSE < bestE):
        bestE = RMSE
        bestp = i


[artrain, artest, theta, intercept, RMSE] = AR(
    bestp, pd.DataFrame(dftesting.Value))

print("Best p-value from AR model: ", bestp)

dfcom = pd.concat([artrain, artest])
# dfcom[['Value', 'Predicted']].plot()

res = pd.DataFrame()
res['Residuals'] = dfcom.Value - dfcom.Predicted

# MA model
bestE = 9999999999
bestq = -1
print("MA model - \n")
for i in range(1, 13):
    [restrain, restest, theta, intercept, RMSE] = MA(
        i, pd.DataFrame(res.Residuals))
    if(RMSE < bestE):
        bestE = RMSE
        bestq = i


[restrain, restest, theta, intercept, RMSE] = MA(
    bestq, pd.DataFrame(res.Residuals))

print("Best q-value from MA model: ", bestq)

rescom = pd.concat([restrain, restest])
dfcom.Predicted += rescom.Predicted
# dfcom[['Value', 'Predicted_values']].plot()


# Getting original data
dfcom.Value += np.log(df).shift(1).Value
dfcom.Value += np.log(df).diff().shift(12).Value
dfcom.Predicted += np.log(df).shift(1).Value
dfcom.Predicted += np.log(df).diff().shift(12).Value
dfcom.Value = np.exp(dfcom.Value)
dfcom.Predicted = np.exp(dfcom.Predicted)

dfcom.iloc[30:, :][['Value', 'Predicted']].plot()
plt.savefig("arima_predicted.png")


# model = ARIMA(df.Value, order=(7, 2, 1))
# modelfit = model.fit()
# modelpredict = modelfit.predict()
# modelpredict.plot()
# df.plot()
# plt.savefig("arima_inbuilt.png")
# plt.show()
