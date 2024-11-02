### DEVELOPED BY: VINOD KUMAR S
### REGISTER NO: 212222240116
### DATE:
# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load the coffee sales data
data = pd.read_csv('coffeesales.csv')
data['datetime'] = pd.to_datetime(data['datetime'])

# Aggregate sales by date
data['date'] = data['datetime'].dt.date
daily_sales = data.groupby('date')['money'].sum().reset_index()

# Set date as the index
daily_sales['date'] = pd.to_datetime(daily_sales['date'])
daily_sales.set_index('date', inplace=True)

# Plot the daily coffee sales
plt.plot(daily_sales.index, daily_sales['money'])
plt.xlabel('Date')
plt.ylabel('Coffee Sales ($)')
plt.title('Coffee Sales Time Series')
plt.show()

# Check stationarity
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

check_stationarity(daily_sales['money'])

# Plot ACF and PACF
plot_acf(daily_sales['money'])
plt.show()
plot_pacf(daily_sales['money'])
plt.show()

# Split the data into training and testing sets
train_size = int(len(daily_sales) * 0.8)
train, test = daily_sales['money'][:train_size], daily_sales['money'][train_size:]

# Fit the SARIMA model
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

# Generate predictions
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Calculate RMSE
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

# Plot actual vs predicted sales
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Coffee Sales ($)')
plt.title('SARIMA Model Predictions for Coffee Sales')
plt.legend()
plt.xticks(rotation=45)
plt.show()


```

### OUTPUT:
![image](https://github.com/user-attachments/assets/07a02929-2686-427c-9b01-d8b8d7659eb6)

![image](https://github.com/user-attachments/assets/a8d14a1e-e0b9-483e-af70-4ab820c5e9a3)

![image](https://github.com/user-attachments/assets/b90f2660-b3b3-4ad0-a9b8-ccc8bd005835)

![image](https://github.com/user-attachments/assets/117dc49a-0925-44cc-9a49-21cdb1734335)





### RESULT:
Thus the program run successfully based on the SARIMA model.
