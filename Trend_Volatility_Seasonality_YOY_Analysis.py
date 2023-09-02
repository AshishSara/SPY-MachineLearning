# Re-loading the data and performing advanced data analysis to identify trends and patterns
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the CSV file into a DataFrame
file_path = '/Users/ashishsaragadam/Downloads/SPY.csv'
spy_df = pd.read_csv(file_path)

# Drop NaN values for advanced analysis
spy_df.dropna(inplace=True)

# Calculate moving averages for 50 and 200 days
spy_df['50_MA'] = spy_df['Close'].rolling(window=50).mean()
spy_df['200_MA'] = spy_df['Close'].rolling(window=200).mean()

# Calculate daily returns and rolling 21-day volatility
spy_df['Daily_Return'] = spy_df['Close'].pct_change()
spy_df['21_day_Volatility'] = spy_df['Daily_Return'].rolling(window=21).std()

# Linear Regression to identify the trendline
X = np.array(range(len(spy_df))).reshape(-1, 1)
y = spy_df['Close'].values
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
spy_df['Residuals'] = y - y_pred

# Summary of findings
ma_crossovers = len(spy_df[(spy_df['50_MA'] > spy_df['200_MA']) & (spy_df['50_MA'].shift(1) < spy_df['200_MA'].shift(1))])
high_volatility_periods = len(spy_df[spy_df['21_day_Volatility'] > spy_df['21_day_Volatility'].mean()])
trend_slope = model.coef_[0]

ma_crossovers, high_volatility_periods, trend_slope

# Import additional libraries for time-series decomposition and seasonality analysis
from statsmodels.tsa.seasonal import seasonal_decompose

# Convert the Date column to datetime format and set it as the index
spy_df['Date'] = pd.to_datetime(spy_df['Date'])
spy_df.set_index('Date', inplace=True)

# Time-series decomposition to analyze trend, seasonality, and residuals
decomposition = seasonal_decompose(spy_df['Close'], model='additive', period=252)  # Using a yearly frequency

trend = decomposition.trend.dropna()
seasonal = decomposition.seasonal.dropna()
residual = decomposition.resid.dropna()

# Calculate year-over-year change for trend component
yoy_change = trend.pct_change(252).dropna()

# Summary of additional findings
average_seasonal_effect = seasonal.mean()
yoy_growth_periods = len(yoy_change[yoy_change > 0])
yoy_decline_periods = len(yoy_change[yoy_change < 0])

average_seasonal_effect, yoy_growth_periods, yoy_decline_periods
print(average_seasonal_effect, yoy_growth_periods, yoy_decline_periods)

# Existing code remains the same up to the Linear Regression part
# ...

# Print Trend Analysis Results
print("Trend Analysis")
print(f"Slope of Trend Line: {trend_slope}")

# Print Volatility Analysis Results
average_volatility = spy_df['21_day_Volatility'].mean()
print("\nVolatility Analysis")
print(f"Average 21-Day Volatility: {average_volatility}")

# Existing code remains the same up to the time-series decomposition
# ...

# Print Seasonality Analysis Results
print("\nSeasonality Analysis")
print(f"Average Seasonal Effect: {average_seasonal_effect}")

# Print Year-over-Year Analysis Results
print("\nYear-over-Year Analysis")
print(f"Number of YoY Growth Periods: {yoy_growth_periods}")
print(f"Number of YoY Decline Periods: {yoy_decline_periods}")
