import yfinance as yf
import pandas as pd

# Define the ticker symbol for S&P 500 (SPY is an ETF that tracks the S&P 500)
ticker_symbol = 'SPY'

# Fetch historical data
sp500_data = yf.download(ticker_symbol, start='2020-10-10', end='2023-12-31')

# Save the data to a CSV file in the VS Code workspace
sp500_data.to_csv('SPY.csv')

print("SPY.csv file created successfully.")
