import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error
dates = pd.date_range(start='2021-01-01', periods=24, freq='M')  
sales = np.random.randint(100, 500, size=(24,))  
data = pd.DataFrame({'Date': dates, 'Sales': sales})
data.set_index('Date', inplace=True)
plt.figure(figsize=(10, 6))
data['Sales'].plot()
plt.title('Sales Data')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()
train = data['Sales'][:int(0.8*len(data))]
test = data['Sales'][int(0.8*len(data)):]
arima_model = ARIMA(train, order=(5, 1, 0))  
arima_model_fit = arima_model.fit()
arima_forecast = arima_model_fit.forecast(steps=len(test))
arima_mse = mean_squared_error(test, arima_forecast)
arima_rmse = arima_mse ** 0.5
print(f'ARIMA Model - RMSE: {arima_rmse}')
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label='Actual Sales')
plt.plot(test.index, arima_forecast, label='ARIMA Predicted Sales', color='orange')
plt.legend()
plt.title('ARIMA Forecast vs Actual Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()


data_prophet = data[['Sales']].reset_index()
data_prophet.columns = ['ds', 'y']  
prophet_model = Prophet()
prophet_model.fit(data_prophet)

future = prophet_model.make_future_dataframe(data_prophet, periods=12, freq='M')

prophet_forecast = prophet_model.predict(future)

prophet_model.plot(prophet_forecast)
plt.title('Sales Prediction using Prophet')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()
future_sales = prophet_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']] 
print(future_sales.tail(12))  
