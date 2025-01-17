# Scalar_task

## 10 financial metrics used:

> 1. Total Profit/Loss: The sum of all profit/losses from trades.
> 2. Number of Trades: The total number of trades executed.
> 3. Average Profit/Loss per Trade: The average profit/loss per trade, calculated by dividing the total profit/loss by the number of trades.
> 4. Sharpe Ratio (annualized): A measure of risk-adjusted return, computed as the ratio of excess return to standard deviation of returns 
over a one-year period.
> 5. Sortino Ratio: Similar to Sharpe Ratio, but using trailing 12-month volatility instead of standard deviation.
> 6. Value-at-Risk (VaR): An estimate of potential losses within a confidence interval (e.g., 95%).
> 7. Expected Return: The average return expected from the strategy, calculated as the sum of profits/losses divided by the number of trades.
> 8. Standard Deviation of Returns: A measure of the volatility of returns.
> 9. Calmar Ratio: A measure of risk-adjusted return, computed as the ratio of excess return to maximum drawdown over a one-year period.
> 10. Information Ratio: A measure of active return (i.e., outperformance) relative to the standard deviation of returns


### Handling Missing Values:

__Dropped rows with missing values in critical columns. Ensured Size and Price are numeric and dropped rows where conversion fails.__

### Accurate Metric Calculations:

*Sharpe Ratio:* Used daily returns and annualized the standard deviation.
*Sortino Ratio:* Used downside risk for calculation.
*Value-at-Risk (VaR):* Calculated using quantile method.
*Calmar Ratio:* Simplified to use cumulative returns for maximum drawdown.
*Information Ratio:* Calculated against a risk-free benchmark return.

