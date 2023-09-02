
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

np.random.seed(42)
tf.random.set_seed(42)
# Load the SPY.csv data
file_path = '/Users/ashishsaragadam/Downloads/SPY.csv'  # Update this path
spy_df = pd.read_csv(file_path)


# Only keep 'Close' prices
data = spy_df.filter(['Close']).values.astype(float)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Prepare training data
train_data = scaled_data[:int(len(data) * 0.8)]
x_train, y_train = [], []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

# Convert to numpy arrays and reshape
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Prepare test data
test_data = scaled_data[len(train_data) - 60:, :]
x_test, y_test = [], data[len(train_data):]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

# Convert to numpy arrays and reshape
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(np.reshape(predictions, (-1, 1)))

# Plot the results
plt.figure(figsize=(16, 8))
plt.title('LSTM Prediction vs Actual Prices')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.plot(y_test, label='Actual Prices')
plt.plot(predictions, label='Predicted Prices')
plt.legend()
plt.show()
# Calculate RMSE and MAE
rmse_lstm = mean_squared_error(y_test, predictions, squared=False)
mae_lstm = mean_absolute_error(y_test, predictions)

print(f'LSTM RMSE: {rmse_lstm}')
print(f'LSTM MAE: {mae_lstm}')

# ... (Your existing LSTM code stays the same up to the RMSE and MAE calculations)

# Initialize portfolio and variables
initial_investment = 1000  # $1000
cash = initial_investment
stock = 0
portfolio = []

# Backtesting
print("Backtesting LSTM...")
for i in range(len(y_test)):
    actual_price = y_test[i]
    predicted_price = predictions[i]

    if predicted_price > actual_price:  # Buy
        if cash > 0:
            stock += cash / actual_price
            cash = 0
    elif predicted_price < actual_price:  # Sell
        if stock > 0:
            cash += stock * actual_price
            stock = 0

    portfolio_value = cash + stock * actual_price
    portfolio.append(portfolio_value)

# Plotting the portfolio value
plt.figure()
plt.plot(portfolio)
plt.title('LSTM Portfolio Value Over Time')
plt.xlabel('Time')
plt.ylabel('Portfolio Value in $')
plt.show()

final_portfolio_value = portfolio[-1]
print(f"Final LSTM Portfolio Value: ${final_portfolio_value}")
