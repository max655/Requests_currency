import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import sqlite3

conn = sqlite3.connect('exchange_rates.db')
cursor = conn.cursor()

cursor.execute('SELECT * FROM exchange_rates')
rows = cursor.fetchall()

data = rows[-365:]

df = pd.DataFrame(data, columns=['r030', 'txt', 'rate', 'cc', 'exchange_date'])

df['exchange_date'] = pd.to_datetime(df['exchange_date'], format="%Y%m%d")

df.set_index('exchange_date', inplace=True)

df['rate'].plot(figsize=(12, 6))
plt.title('Time Series Plot')
plt.show()

model = LinearRegression()

df_train = df.head(300)

X = pd.to_numeric(df_train.index.values).reshape(-1, 1)
y = df_train['rate'].values

model.fit(X, y)

forecast_steps = 120
forecast_dates = pd.date_range(start=df_train.index[-1] + pd.DateOffset(1), periods=forecast_steps)
forecast_values = model.predict(pd.to_numeric(forecast_dates.values).reshape(-1, 1))

plt.plot(df.index, df['rate'], label='Actual Values')
plt.plot(forecast_dates, forecast_values, label='Forecast Values')
plt.title('Actual vs Forecasted Values using Linear Regression')
plt.xlabel('Date')
plt.ylabel('Exchange Rate')
plt.legend()
plt.show()
