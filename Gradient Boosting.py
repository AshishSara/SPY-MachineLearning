from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Note: Make sure to update this file path with the actual path to your SPY.csv file
file_path = '/Users/ashishsaragadam/Downloads/SPY.csv'
spy_df = pd.read_csv(file_path)
spy_df['Date'] = pd.to_datetime(spy_df['Date'])
spy_df.set_index('Date', inplace=True)

data = spy_df['Close'].values
train_size = int(len(data) * 0.8)
train_gb, test_gb = data[:train_size], data[train_size:]

def arima_predict(train, test):
    history = [x for x in train]
    predictions_arima = []
    for t in range(len(test)):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions_arima.append(yhat)
        history.append(test[t])
    return predictions_arima

def rf_predict(train, test):
    X, y = [], []
    lag = 5
    for i in range(lag, len(train)):
        X.append(train[i-lag:i])
        y.append(train[i])
    X, y = np.array(X), np.array(y)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    X_test = []
    for i in range(lag, len(test)):
        X_test.append(test[i-lag:i])
    X_test = np.array(X_test)
    predictions_rf = rf_model.predict(X_test)
    return predictions_rf

def lstm_predict(train, test):
    train_data = train.reshape(-1, 1)
    test_data = test.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train_data)
    x_train, y_train = [], []
    for i in range(60, len(train)):
        x_train.append(scaled_train[i-60:i, 0])
        y_train.append(scaled_train[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    test_data = scaler.transform(test_data)
    x_test = []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(np.reshape(predictions, (-1, 1)))
    return predictions.flatten()

# Generate predictions from individual models
predictions_arima = arima_predict(train_gb, test_gb)
predictions_rf = rf_predict(train_gb, test_gb)
predictions_lstm = lstm_predict(train_gb, test_gb)

# Trim or pad the predictions to have the same length
min_length = min(len(predictions_arima), len(predictions_rf), len(predictions_lstm))

predictions_arima = predictions_arima[:min_length]
predictions_rf = predictions_rf[:min_length]
predictions_lstm = predictions_lstm[:min_length]

# Combine model predictions into a DataFrame
train_predictions = pd.DataFrame({
    'arima': predictions_arima,
    'rf': predictions_rf,
    'lstm': predictions_lstm
})

# Trim the test set to match the length of the smallest predictions array
test_gb_trimmed = test_gb[5:5 + min_length]

# Hyperparameter grid for Gradient Boosting
param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 4, 5]
}

# Initialize Gradient Boosting model
gb_model = GradientBoostingRegressor()

# Initialize and train Gradient Boosting model
grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, cv=3)
grid_search.fit(train_predictions, test_gb_trimmed)

# Get the best model
best_gb_model = grid_search.best_estimator_

# Make final predictions
final_predictions = best_gb_model.predict(train_predictions)

# Initialize portfolio and variables
initial_investment = 1000  # $1000
cash = initial_investment
stock = 0
portfolio = []

# Backtesting
print("Backtesting...")
for i in range(min_length):
    actual_price = test_gb_trimmed[i]
    predicted_price = final_predictions[i]

    if predicted_price > actual_price:  # Buy
        if cash > 0:
            stock += cash / actual_price
            cash = 0
    elif predicted_price < actual_price:  # Sell
        if stock > 0:
            cash += stock * actual_price
            stock = 0

    # Calculate and store the total portfolio value
    portfolio_value = cash + stock * actual_price
    portfolio.append(portfolio_value)

# Print the final portfolio value
final_portfolio_value = portfolio[-1]
print(f"Final Portfolio Value: ${final_portfolio_value:.2f}")

# Optionally, you can plot the portfolio value over time
plt.figure()
plt.plot(portfolio)
plt.title('Portfolio Value Over Time')
plt.xlabel('Time')
plt.ylabel('Portfolio Value in $')
plt.show()

#test 1: Final Portfolio Value: $3322.10