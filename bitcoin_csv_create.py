import yfinance as yf
import pandas as pd

# Define the ticker symbol for Bitcoin
ticker_symbol = 'BTC-USD'

# Fetch historical data
bitcoin_data = yf.download(ticker_symbol, start='2020-01-01', end='2023-12-31')

# Save the data to a CSV file in the VS Code workspace
bitcoin_data.to_csv('bitcoin.csv')

print("bitcoin.csv file created successfully.")
