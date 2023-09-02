import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the SPY.csv data
file_path = '/Users/ashishsaragadam/Downloads/SPY.csv'  # Update this path
spy_df = pd.read_csv(file_path)

# Convert the Date to datetime format and set it as the index
spy_df['Date'] = pd.to_datetime(spy_df['Date'])
spy_df.set_index('Date', inplace=True)

# Prepare data for ARIMA
close_prices = spy_df['Close'].values
train_size = int(len(close_prices) * 0.8)
train_arima, test_arima = close_prices[:train_size], close_prices[train_size:]
history = [x for x in train_arima]

# Assume close_prices is your actual SPY close prices
close_prices = spy_df['Close'].values
train_size = int(len(close_prices) * 0.8)
train_arima, test_arima = close_prices[:train_size], close_prices[train_size:]
history = [x for x in train_arima]

# Fit and predict using ARIMA
print("Fitting ARIMA model...")
predictions_arima = []
for t in range(len(test_arima)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    predictions_arima.append(yhat)
    history.append(test_arima[t])

# Calculate and print ARIMA RMSE
rmse_arima = mean_squared_error(test_arima, predictions_arima, squared=False)
print(f'ARIMA RMSE: {rmse_arima}')

# Plot ARIMA results
plt.figure()
plt.plot(test_arima, label='Actual')
plt.plot(predictions_arima, label='Predicted')
plt.title('ARIMA Model')
plt.legend()
plt.show()

# Initialize portfolio and variables
initial_investment = 1000  # $1000
cash = initial_investment
stock = 0
portfolio = []

# Backtesting
print("Backtesting...")
for i in range(len(test_arima)):
    actual_price = test_arima[i]
    predicted_price = predictions_arima[i]

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
plt.title('Portfolio Value Over Time')
plt.xlabel('Time')
plt.ylabel('Portfolio Value in $')
plt.show()

final_portfolio_value = portfolio[-1]
print(f"Final Portfolio Value: ${final_portfolio_value}")
