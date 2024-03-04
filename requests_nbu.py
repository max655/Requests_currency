import pandas as pd
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, AutoDateFormatter
import sqlite3

conn = sqlite3.connect('exchange_rates.db')
cursor = conn.cursor()

url = 'https://bank.gov.ua/NBUStatService/v1/statdirectory/exchangenew?json'
start_date_input = input("Enter the start date (YYYYMMDD): ")
end_date_input = input("Enter the end date (YYYYMMDD): ")

start_date = datetime.strptime(start_date_input, "%Y%m%d")
end_date = datetime.strptime(end_date_input, "%Y%m%d")

currencies = []
response_currencies = requests.get(url)

if response_currencies.status_code == 200:
    data = response_currencies.json()
    if data:
        for i in range(len(data)):
            currencies.append(data[i]['cc'])

print(f"Available currencies: {currencies}")
currency = input("Input currency: ").upper()

cursor.execute('SELECT DISTINCT exchange_date, cc FROM exchange_rates')
existing_data = [(row[0], row[1]) for row in cursor.fetchall()]

dates_to_load = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]

dates_to_load_str = [date.strftime("%Y%m%d") for date in dates_to_load if (date.strftime("%Y%m%d"), currency)
                     not in existing_data]

dates = []
rates = []
full_data = []

for date_str in dates_to_load_str:
    print(date_str)
    date_json = f'&date={date_str}&valcode={currency}'
    response = requests.get(url + date_json, timeout=100)

    if response.status_code == 200:
        data = response.json()
        for entry in data:
            full_data.append(entry)
        if data:
            dates.append(date_str)
            rates.append(data[0]['rate'])
    else:
        print(f"Request failed: {requests.exceptions.RequestException}")



# cursor.execute('DROP TABLE IF EXISTS exchange_rates')
cursor.execute('''
CREATE TABLE IF NOT EXISTS exchange_rates (
r030 INTEGER,
txt TEXT,
rate REAL,
cc TEXT,
exchange_date TEXT 
)
''')

for i in range(len(full_data)):
    date_value = dates[i]
    rate_value = rates[i]
    cc_value = currency
    r030_value = full_data[0]['r030']
    txt_value = full_data[0]['txt']
    cursor.execute('''
    INSERT INTO exchange_rates (r030, txt, rate, cc, exchange_date)
    VALUES (?, ?, ?, ?, ?)   
    ''', (r030_value, txt_value, rate_value, cc_value, date_value))

cursor.execute('SELECT * FROM exchange_rates')
rows = cursor.fetchall()
for row in rows:
    print(row)

conn.commit()
conn.close()

print("Data has been successfully saved to the database.")

data = {'Date': dates, 'Rate': rates}
df = pd.DataFrame(data)

df['Date'] = pd.to_datetime(df['Date'])

df['Week'] = df['Date'].dt.to_period('W')
weekly_average = df.groupby('Week')['Rate'].mean()

weekly_dates = weekly_average.index.to_timestamp()

fig, ax = plt.subplots(figsize=(14, 10))
plt.plot(weekly_dates, weekly_average, label='Exchange Rate')

locator = AutoDateLocator()
ax.xaxis.set_major_locator(locator)

formatter = AutoDateFormatter(locator)
ax.xaxis.set_major_formatter(formatter)

plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Exchange Rate')
plt.title(f'Exchange Rate for {currency} ({start_date_input}-{end_date_input})')
plt.legend()
plt.show()
