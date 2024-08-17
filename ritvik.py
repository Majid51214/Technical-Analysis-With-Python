import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import date
import plotly.express as px
import math
import plotly.graph_objects as go
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from pandas.plotting import register_matplotlib_converters
# pd.options.mode.chained_assignment=None
from time import time
import warnings
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, plot_predict
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from pmdarima.arima import auto_arima
register_matplotlib_converters
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from pmdarima.model_selection import train_test_split


#dictionary
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from scipy.stats import normaltest
from statsmodels.tsa.stattools import acf,pacf
#from pmdarima.arima import auto_arima
import scipy.interpolate as sci
import scipy.optimize as sco
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import scipy.optimize as sco

# end_date = date.today()
# start_date = end_date - timedelta(days=730)
# tommorow = end_date + timedelta(days=1)
# tickersymbol = 'TME'
# data = yf.download(tickers=tickersymbol, start=start_date, end=end_date)
# prices = data.Close
# returns = prices.pct_change().dropna()

# model = ARIMA(prices, order=(2,1,2))
# fitted = model.fit()

# result = fitted.forecast(100)

# plt.plot(prices.index, prices, label='Historical Data')
# plt.plot(pd.date_range(prices.index[-1], periods=100), result, label='Forecast', color='red')
# plt.legend()
# plt.title("ARIMA Forecasting of Daily Female Births")
# plt.show() 


df1 = pd.read_csv(r"D:\python for finance\sp500\HBI.csv")
df2 = pd.read_csv(r"D:\python for finance\sp500\PM.csv")
for df in (df1, df2):
    df.set_index('Date', inplace=True)
    df.index = pd.to_datetime(df.index)


df = pd.DataFrame({'HBI':[], 'PM':[]})
symbols = ['HBI', 'PM']
df.HBI = df1.Close
df.PM = df2.Close
df.dropna()
# sns.set_theme(style='darkgrid')
# sns.lineplot(data=df)
# plt.grid(True)
# plt.xlabel('Date')
# plt.ylabel('USD')
# plt.title('Stock closing prices')
# plt.show()



# for i in symbols[0:]:
#     result = seasonal_decompose(df[i], model='multiplicative', period=30)
#     fig = plt.figure()  
#     fig = result.plot()  
#     plt.title(i) 
#     fig.set_size_inches(12, 8)
# plt.show()

# for i in symbols:
#     print([i])
#     result = adfuller(df[i], autolag='AIC')
#     print('adf result :%f' %result[0])
#     print('p-values: %f' % result[1])
#     print("critical values:")
#     for key, value in result[4].items():
#          print('\t%s: %.3f' % (key, value))
#     if result[0] < result[4]["5%"]:
#         print("reject null hypothesis. time series is not stationary")
#     else:
#         print("accept hypothesis. time series is stationary.")
#     print("\n")

# df = np.log(df/df.shift(1))
# df = df.dropna()
train_df = df['HBI'][:int(len(df['HBI']) * .8)]
test_df = df['HBI'][int(len(df) * .8):]
model = ARIMA(train_df, order=(2,1,2))
model = model.fit()

train_df2 = df['PM'][1:int(len(df['PM']) * .8)]
test_df2 = df['PM'][int(len(df) * .8):]
model2 = ARIMA(train_df2, order=(2,1,2))
model2 = model2.fit()

forecast1 = model.predict(n_periods=len(test_df))
forecast10 = model.forecast(len(test_df))
forecast2 = model2.predict(n_periods=len(test_df2))
plt.plot(train_df)
plt.plot(test_df)
plt.plot(forecast1)
plt.plot(pd.date_range(test_df.index[-1], periods=len(test_df)), forecast10, label='Forecast', color='red')
plt.show()
  #what is pd.date_range?  