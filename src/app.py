from utils import db_connect
engine = db_connect()

# your code here

# instalar carpetas:
%pip install -r ../requirements.txt
! pip install -q sklearn
! pip install -q seaborn
! pip install -q cufflinks
! pip install -q statsmodels
%%capture
!pip install fbprophet
!pip install prophet
!pip install pystan==2.19.1.1 prophet
!pip install --upgrade plotly
%%capture
!pip install pmdarima

# cargar las librerias: 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode()
%matplotlib inline
plt.style.use('fivethirtyeight')
from prophet import Prophet
from prophet.plot import plot_plotly

# Cargar la data:

# dataset a: 
data_train_a = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-train-a.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)
data_test_a = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-test-a.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)

# dataset b 
data_train_b = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-train-b.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)
data_test_b = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-test-b.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)

data_train_a.reset_index(inplace=True)

# renombrar las columnas:
data_train_a.rename(columns={'datetime': 'ds', 'cpu': 'y'}, inplace=True)

m = Prophet()
m.fit(data_train_a)

future = m.make_future_dataframe(periods=1)
future.tail()

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# Using matplotlib visualize the data.
m = Prophet(weekly_seasonality=False)
m.add_seasonality(name='hourly', period=60, fourier_order=5)
forecast = m.fit(data_train_a).predict(future)
fig = m.plot_components(forecast)

m = Prophet(changepoint_prior_scale=0.01).fit(data_train_a)
future = m.make_future_dataframe(periods=300, freq='1min')
fcst = m.predict(future)
fig = m.plot(fcst)

m =Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
m.fit(data_train_a)

# grafico diario por semana por año 
m=Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
m.fit(data_train_a)
future = m.make_future_dataframe(periods=300, freq='1min')
fcst = m.predict(future)
m.plot_components(fcst)

# Cargamos los datos de nuevo
data_train_a = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-train-a.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)
data_test_a = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-test-a.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)

# hacemos el index en los datos 
data_train_a.index = pd.to_datetime(data_train_a.index)

# Arima
#Use the ARIMA model to fit the data.
from pmdarima.arima import auto_arima
stepwise_model = auto_arima(data_train_a, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
print(stepwise_model.aic())

# Evaluate the performance
stepwise_model.fit(data_train_a)
stepwise_model.fit(data_train_a).plot_diagnostics(figsize=(15, 12))
plt.show()

future_forecast = stepwise_model.predict(n_periods=60)

future_forecast = pd.DataFrame(future_forecast,index = data_test_a.index,columns=['Prediction'])
pd.concat([data_test_a,future_forecast],axis=1).plot()

# Modelo b:
data_train_b.reset_index(inplace=True)

# renombrar las columnas:
data_train_b.rename(columns={'datetime': 'ds', 'cpu': 'y'}, inplace=True)

# modelo train_b
m = Prophet()
m.fit(data_train_b)

future = m.make_future_dataframe(periods=1)
future.tail()

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# Using matplotlib visualize the data.
m = Prophet(weekly_seasonality=False)
m.add_seasonality(name='hourly', period=60, fourier_order=5)
forecast = m.fit(data_train_b).predict(future)
fig = m.plot_components(forecast)

m = Prophet(changepoint_prior_scale=0.01).fit(data_train_b)
future = m.make_future_dataframe(periods=300, freq='1min')
fcst = m.predict(future)
fig = m.plot(fcst)

m =Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
m.fit(data_train_b)

# grafico diario por semana por año 
m=Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
m.fit(data_train_b)
future = m.make_future_dataframe(periods=300, freq='1min')
fcst = m.predict(future)
m.plot_components(fcst)

# cargamos la data de nuevo:
data_train_b = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-train-b.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)
data_test_b = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-test-b.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)

data_train_b.index = pd.to_datetime(data_train_b.index)

# Arima
#Use the ARIMA model to fit the data.
from pmdarima.arima import auto_arima
stepwise_model = auto_arima(data_train_b, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
print(stepwise_model.aic())

# Evaluate the performance
stepwise_model.fit(data_train_b)
stepwise_model.fit(data_train_b).plot_diagnostics(figsize=(15, 12))
plt.show()

future_forecast = stepwise_model.predict(n_periods=60)
future_forecast = pd.DataFrame(future_forecast,index = data_test_b.index,columns=['Prediction'])
pd.concat([data_test_b,future_forecast],axis=1).plot()




















