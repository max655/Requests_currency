import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import sqlite3

conn = sqlite3.connect('exchange_rates.db')
cursor = conn.cursor()

cursor.execute("SELECT * FROM exchange_rates WHERE cc = 'USD' and exchange_date BETWEEN 20140101 AND 20161231 "
               "ORDER BY exchange_date ASC")
data = cursor.fetchall()

df = pd.DataFrame(data, columns=['r030', 'txt', 'rate', 'cc', 'exchange_date'])

df['exchange_date'] = pd.to_datetime(df['exchange_date'], format="%Y%m%d")

df.set_index('exchange_date', inplace=True)

df['rate'].plot(figsize=(12, 6))
plt.title('Time Series Plot')
plt.show()

model = ExponentialSmoothing(df['rate'], trend='add', seasonal='add', seasonal_periods=12)
results = model.fit()

forecast_steps = 365
forecast_values = results.forecast(steps=forecast_steps)

plt.plot(df.index, df['rate'], label='Actual Values')
plt.plot(pd.date_range(start=df.index[-1] + pd.DateOffset(1), periods=forecast_steps), forecast_values, label='Forecast Values')
plt.title('Exponential Smoothing Forecast')
plt.xlabel('Date')
plt.ylabel('Exchange Rate')
plt.legend()
plt.show()
