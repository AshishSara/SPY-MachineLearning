
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
np.random.seed(42)
tf.random.set_seed(42)
# Load the SPY.csv data
file_path = '/Users/ashishsaragadam/Downloads/SPY.csv'  # Update this path
spy_df = pd.read_csv(file_path)

# Prepare the data for Random Forest
data = spy_df['Close'].values.reshape(-1, 1)
train_size = int(len(data) * 0.8)
train_rf, test_rf = data[:train_size], data[train_size:]

# Create lagged dataset for Random Forest
X, y = [], []
lag = 5  # You can change the lag value
for i in range(lag, len(train_rf)):
    X.append(train_rf[i-lag:i].flatten())
    y.append(train_rf[i])

X, y = pd.DataFrame(X), pd.DataFrame(y)

# Train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y.values.ravel())

# Prepare test data
X_test, y_test = [], test_rf
for i in range(lag, len(test_rf)):
    X_test.append(test_rf[i-lag:i].flatten())

X_test = pd.DataFrame(X_test)

# Make predictions
predictions_rf = rf_model.predict(X_test)

# Calculate RMSE for Random Forest
rmse_rf = mean_squared_error(y_test[lag:], predictions_rf, squared=False)
print(f'Random Forest RMSE: {rmse_rf}')

# Initialize portfolio and variables
initial_investment = 1000  # $1000
cash = initial_investment
stock = 0
portfolio_rf = []

# Backtesting Random Forest
print("Backtesting Random Forest...")
for i in range(len(y_test) - lag):
    actual_price = y_test[i + lag]
    predicted_price = predictions_rf[i]

    if predicted_price > actual_price:  # Buy
        if cash > 0:
            stock += cash / actual_price
            cash = 0
    elif predicted_price < actual_price:  # Sell
        if stock > 0:
            cash += stock * actual_price
            stock = 0

    portfolio_value = cash + stock * actual_price
    portfolio_rf.append(portfolio_value)

# Plotting the Random Forest portfolio value
plt.figure()
plt.plot(portfolio_rf)
plt.title('Random Forest Portfolio Value Over Time')
plt.xlabel('Time')
plt.ylabel('Portfolio Value in $')
plt.show()

final_portfolio_value_rf = portfolio_rf[-1]
print(f"Final Random Forest Portfolio Value: ${final_portfolio_value_rf[0]}")
