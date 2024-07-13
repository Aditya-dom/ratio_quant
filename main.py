import pandas as pd
from datetime import datetime
import numpy as np
import random

# Load the dataset
file_path = 'testData.csv'
trades_df = pd.read_csv(file_path)

# Display the first few rows of the DataFrame to understand its structure
print(trades_df.head())

# Check for column names and ensure they match the required format
print(trades_df.columns)

# Standardize column names to lower case for consistency
trades_df.columns = trades_df.columns.str.lower()

# Display the updated columns to verify the changes
print(trades_df.columns)

# Preprocess the data to ensure it matches the required format
# Convert 'date' to datetime format if present
if 'date' in trades_df.columns:
    trades_df['date'] = pd.to_datetime(trades_df['date'])

# Ensure 'side' contains only 'buy' or 'sell' values if present
if 'side' in trades_df.columns:
    trades_df['side'] = trades_df['side'].str.lower()
    trades_df = trades_df[trades_df['side'].isin(['buy', 'sell'])]

# Ensure 'size' and 'price' are floats if present
if 'size' in trades_df.columns:
    trades_df['size'] = pd.to_numeric(trades_df['size'], errors='coerce')
if 'price' in trades_df.columns:
    trades_df['price'] = pd.to_numeric(trades_df['price'], errors='coerce')

# Drop rows with missing 'size' or 'price' only if both columns exist
if 'size' in trades_df.columns and 'price' in trades_df.columns:
    trades_df = trades_df.dropna(subset=['size', 'price'])

# Display the cleaned DataFrame to verify preprocessing
print(trades_df.head())

# Define the calculateTradePerformance function
def getTickerPrice(ticker: str, date: datetime) -> float:
    # Return a random price between $1 and $100 for demonstration purposes
    return random.uniform(1, 100)

def calculateTradePerformance(trades_df: pd.DataFrame) -> dict:
    """
    Calculate trade performance metrics from the given DataFrame.

    Args:
        trades_df (pd.DataFrame): Input DataFrame containing trade data

    Returns:
        A dictionary with calculated metrics (see below for details)
    """
    metrics = {}
    if trades_df.empty:
        return metrics

    required_columns = {'side', 'size', 'price'}
    if required_columns.issubset(trades_df.columns):
        trades_df = trades_df.dropna(subset=['side', 'size', 'price'])
        trades_df['size'] = pd.to_numeric(trades_df['size'], errors='coerce')
        trades_df['price'] = pd.to_numeric(trades_df['price'], errors='coerce')
        trades_df = trades_df.dropna(subset=['size', 'price'])

        total_profit_loss = (
            (trades_df[trades_df['side'] == 'sell']['size'] * trades_df[trades_df['side'] == 'sell']['price']).sum() -
            (trades_df[trades_df['side'] == 'buy']['size'] * trades_df[trades_df['side'] == 'buy']['price']).sum()
        )
        num_trades = len(trades_df)
        avg_profit_loss_per_trade = total_profit_loss / num_trades if num_trades > 0 else 0

        trades_df['return'] = trades_df['size'] * trades_df['price']
        daily_returns = trades_df['return']

        mean_return = daily_returns.mean()
        std_return = daily_returns.std()
        risk_free_rate = getTickerPrice(trades_df.iloc[0]['symbol'], trades_df.iloc[0]['date']) if 'symbol' in trades_df.columns and 'date' in trades_df.columns else 0
        sharpe_ratio = (mean_return - risk_free_rate) / std_return * np.sqrt(252) if std_return != 0 else 0

        downside_risk = daily_returns[daily_returns < 0].std() * np.sqrt(252)
        sortino_ratio = (mean_return - risk_free_rate) / downside_risk if downside_risk != 0 else 0

        var = daily_returns.quantile(0.05)
        expected_return = mean_return
        std_dev_returns = std_return

        cumulative_returns = daily_returns.cumsum()
        max_drawdown = cumulative_returns.min()
        calmar_ratio = expected_return / abs(max_drawdown) if max_drawdown != 0 else 0

        benchmark_return = risk_free_rate
        info_ratio = (mean_return - benchmark_return) / std_dev_returns if std_dev_returns != 0 else 0

        metrics.update({
            'total_profit_loss': total_profit_loss,
            'num_trades': num_trades,
            'avg_profit_loss_per_trade': avg_profit_loss_per_trade,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'var': var,
            'expected_return': expected_return,
            'std_dev_returns': std_dev_returns,
            'calmar_ratio': calmar_ratio,
            'info_ratio': info_ratio
        })

    return metrics

# Apply the trade performance calculation function
performance_metrics = calculateTradePerformance(trades_df)
print(performance_metrics)



