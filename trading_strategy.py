# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 18:11:19 2025

@author: tomls
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import tensorflow as tf

def load_models(model_files=["st5_lstm_model", "st7_lstm_model", "st8_lstm_model"]):
    models = {}
    for model_base in model_files:
        try:
            weights_file = f"{model_base}.h5"
            json_file = f"{model_base}.json"
            
            print(f"Loading model from {weights_file} and {json_file}...")
            
            if not os.path.exists(weights_file) or not os.path.exists(json_file):
                print(f"  Error: Model files for {model_base} not found")
                continue
                
            with open(json_file, 'r') as file:
                model_json = file.read()
                
            model = tf.keras.models.model_from_json(model_json)
            
            model.load_weights(weights_file)
            
            model.compile(optimizer='adam', loss='mse')
            
            models[model_base] = model
            print(f"  Successfully loaded model from {model_base}")
        except Exception as e:
            print(f"  Error loading model from {model_base}: {str(e)}")
    
    return models

def load_stock_data(stock_numbers=[5, 7, 8]):
    stock_data = {}
    for stock_num in stock_numbers:
        file_name = f"st{stock_num}.csv"
        try:
            print(f"Loading stock data from {file_name}...")
            df = pd.read_csv(file_name)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            if 'Adj Close' not in df.columns and 'Close' in df.columns:
                df['Adj Close'] = df['Close']
                
            df = df.dropna(subset=['Adj Close'])
            
            stock_data[stock_num] = df
            print(f"  Successfully loaded data for stock {stock_num}")
        except Exception as e:
            print(f"  Error loading data for stock {stock_num}: {str(e)}")
    
    return stock_data

def backtest_strategy(models, stock_data, test_days=365, n_lags=10):
    model_to_stock = {
        "st5_lstm_model": 5,
        "st7_lstm_model": 7,
        "st8_lstm_model": 8
    }
    
    results = {}
    
    for model_name, model in models.items():
        stock_num = model_to_stock.get(model_name)
        if stock_num is None or stock_num not in stock_data:
            print(f"Warning: Cannot match model {model_name} to stock data")
            continue
        
        df = stock_data[stock_num]
        
        if len(df) <= test_days + n_lags:
            print(f"Warning: Not enough data for stock {stock_num} for backtesting")
            continue
        
        dates = df['Date'].values[-test_days:]
        actual_prices = df['Adj Close'].values[-test_days:]
        predicted_prices = np.zeros(test_days)
        
        print(f"Generating predictions for stock {stock_num}...")
        for i in range(test_days):
            feature_window = df['Adj Close'].values[-(test_days-i+n_lags):-(test_days-i)]
            
            lstm_input = feature_window.reshape(1, n_lags, 1)
            
            prediction = model.predict(lstm_input, verbose=0)[0][0]
            predicted_prices[i] = prediction
        
        results[stock_num] = {
            "dates": dates,
            "actual": actual_prices,
            "predicted": predicted_prices,
            "rmse": np.sqrt(np.mean((actual_prices - predicted_prices) ** 2)),
            "mape": np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100,
            "directional_accuracy": np.mean(
                np.sign(np.diff(actual_prices)) == np.sign(np.diff(predicted_prices))
            ) * 100
        }
        
        print(f"  RMSE: {results[stock_num]['rmse']:.4f}")
        print(f"  MAPE: {results[stock_num]['mape']:.2f}%")
        print(f"  Directional Accuracy: {results[stock_num]['directional_accuracy']:.2f}%")
    
    return results

def create_prediction_dataframe(backtest_results):
    first_stock = list(backtest_results.keys())[0]
    dates = backtest_results[first_stock]["dates"]
    
    df = pd.DataFrame({"Date": dates})
    
    for stock_num, results in backtest_results.items():
        df[f"actual_{stock_num}"] = results["actual"]
        df[f"predicted_{stock_num}"] = results["predicted"]
    
    return df

def plot_backtest_results(backtest_results):
    n_stocks = len(backtest_results)
    fig, axes = plt.subplots(n_stocks, 1, figsize=(12, 5*n_stocks))
    
    if n_stocks == 1:
        axes = [axes]
    
    for i, (stock_num, results) in enumerate(backtest_results.items()):
        ax = axes[i]
        dates = results["dates"]
        
        if isinstance(dates[0], (int, float)) and dates[0] > 86400:
            dates = [datetime.fromtimestamp(d) for d in dates]
        elif isinstance(dates[0], np.datetime64):
            dates = pd.to_datetime(dates)
        
        ax.plot(dates, results["actual"], label="Actual", color="blue")
        ax.plot(dates, results["predicted"], label="Predicted", color="red", linestyle="--")
        ax.set_title(f"Stock {stock_num} - RMSE: {results['rmse']:.4f}, DA: {results['directional_accuracy']:.2f}%")
        ax.legend()
        ax.grid(True, alpha=0.3, color='gray', linestyle='-', linewidth=0.5) 
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
    
    plt.tight_layout()
    plt.savefig("backtest_predictions.png")
    plt.show()


def trading_strategy(prediction_df):
    stock_nums = [7,5,8]
    # Initialize variables
    cash = 10_000  
    
    # Initialize positions (positive = long, negative = short)
    positions = {stock: 0 for stock in stock_nums}
    
    # Short position limits
    max_total_short = 10_000  # Maximum allowed total short value
    max_short_per_stock = 3_333  # Maximum allowed short value per stock
    
    # Cash allocation per stock
    cash_per_stock = {}
    base_allocation = float(np.floor(cash / len(stock_nums)))
    
    for i, stock in enumerate(stock_nums):
        if i < len(stock_nums) - 1:
            cash_per_stock[stock] = base_allocation
        else:
            cash_per_stock[stock] = cash - base_allocation * (len(stock_nums) - 1)
    
    days = len(prediction_df)
    portfolio_values = np.zeros(days)
    portfolio_values[0] = cash  
    
    # Transaction log
    transactions = []
    
    available_cash = cash
    
    for day in range(1, days):  
        current_short_values = {stock: max(0, -positions[stock] * prediction_df[f"actual_{stock}"].iloc[day]) 
                               for stock in stock_nums}
        total_short_value = sum(current_short_values.values())
        
        for stock in stock_nums:
            yesterday_pred = prediction_df[f"predicted_{stock}"].iloc[day-1]
            today_pred = prediction_df[f"predicted_{stock}"].iloc[day]
            today_actual = prediction_df[f"actual_{stock}"].iloc[day]
            yesterday_actual = prediction_df[f"actual_{stock}"].iloc[day-1]
            
            current_position = positions[stock]
            if today_pred > yesterday_pred:  # Prediction increased - BUY or COVER SHORT
                if current_position < 0:  # Have short position - COVER
                    shares_to_cover = int(abs(current_position))
                    if shares_to_cover > 0:
                        cost = shares_to_cover * yesterday_actual
                        
                        available_cash -= cost
                        positions[stock] += shares_to_cover
                        
                        transactions.append({
                            "day": day,
                            "stock": stock,
                            "action": "COVER",
                            "price": yesterday_actual,
                            "quantity": shares_to_cover,
                            "cash_after": available_cash
                        })
                
                else:  # No short position - BUY
                    max_buy_value = min(cash_per_stock[stock], available_cash)
                    shares_to_buy = int(max_buy_value / yesterday_actual)
                    
                    if shares_to_buy > 0:
                        cost = shares_to_buy * yesterday_actual
                        
                        available_cash -= cost
                        positions[stock] += shares_to_buy
                        
                        transactions.append({
                            "day": day,
                            "stock": stock,
                            "action": "BUY",
                            "price": yesterday_actual,
                            "quantity": shares_to_buy,
                            "cash_after": available_cash
                        })
            
            elif today_pred < yesterday_pred:  # Prediction decreased - SELL or SHORT
                if current_position > 0:  # Have long position - SELL
                    shares_to_sell = int(current_position)
                    
                    if shares_to_sell > 0:
                        proceeds = shares_to_sell * yesterday_actual
                        
                        available_cash += proceeds
                        positions[stock] -= shares_to_sell
                        
                        transactions.append({
                            "day": day,
                            "stock": stock,
                            "action": "SELL",
                            "price": yesterday_actual,
                            "quantity": shares_to_sell,
                            "cash_after": available_cash
                        })
                
                else:  # No long position - SHORT
                    current_short_value = current_short_values[stock]
                    room_for_short_stock = max_short_per_stock - current_short_value
                    room_for_short_total = max_total_short - total_short_value
                    room_for_short = min(room_for_short_stock, room_for_short_total)
                    
                    max_short_value = min(cash_per_stock[stock], room_for_short)
                    shares_to_short = int(max_short_value / yesterday_actual)
                    
                    if shares_to_short > 0:
                        proceeds = shares_to_short * yesterday_actual
                        
                        available_cash += proceeds
                        positions[stock] -= shares_to_short
                        
                        transactions.append({
                            "day": day,
                            "stock": stock,
                            "action": "SHORT",
                            "price": yesterday_actual,
                            "quantity": shares_to_short,
                            "cash_after": available_cash
                        })
            
            # If prediction unchanged, maintain current position
        
        portfolio_value = available_cash
        
        # Add value of all positions (long positions add value, short positions subtract)
        for stock in stock_nums:
            position = positions[stock]
            price = prediction_df[f"actual_{stock}"].iloc[day]
            
            if position > 0:  # Long position
                portfolio_value += position * price
            elif position < 0:  # Short position
                portfolio_value += position * price  # This subtracts the value since position is negative
        
        portfolio_values[day] = portfolio_value
    
    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_value) / initial_value * 100
    
    daily_returns = np.zeros(days-1)
    for i in range(1, days):
        daily_returns[i-1] = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
    
    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)  
    
    peak = portfolio_values[0]
    max_drawdown = 0
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    max_drawdown *= 100 
    
    print(f"\nStrategy Performance Summary:")
    print(f"Initial Portfolio Value: ${initial_value:.2f}")
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    print(f"Number of Transactions: {len(transactions)}")
    
    return {
        "portfolio_value": portfolio_values,
        "dates": prediction_df["Date"].values,
        "transactions": transactions,
        "metrics": {
            "initial_value": initial_value,
            "final_value": final_value,
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown
        }
    }

def plot_portfolio_performance(strategy_results):

    fig = plt.figure(figsize=(18, 12))
    
    transactions = strategy_results["transactions"]
    stock_nums = list(set([t["stock"] for t in transactions]))
    
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    
    ax_portfolio = fig.add_subplot(gs[0, :])
    
    ax_positions = [fig.add_subplot(gs[1, i]) for i in range(min(3, len(stock_nums)))]
    
    position_axes = {stock_nums[i]: ax_positions[i] for i in range(min(3, len(stock_nums)))}
    
    dates = strategy_results["dates"]
    portfolio_values = strategy_results["portfolio_value"]
    
    if isinstance(dates[0], np.datetime64):
        dates = pd.to_datetime(dates)
    
    ax_portfolio.plot(dates, portfolio_values, label="Portfolio Value", color="blue", linewidth=2)
    
    metrics = strategy_results["metrics"]
    total_return = metrics["total_return"]
    sharpe_ratio = metrics["sharpe_ratio"]
    max_drawdown = metrics["max_drawdown"]
    
    annotation_text = (
        f"Total Return: {total_return:.2f}%\n"
        f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
        f"Max Drawdown: {max_drawdown:.2f}%"
    )
    
    ax_portfolio.annotate(annotation_text, xy=(0.02, 0.95), xycoords='axes fraction', 
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                fontsize=10, verticalalignment='top')
    
    ax_portfolio.set_title("Portfolio Performance Over Time", fontsize=16)
    ax_portfolio.set_xlabel("Date", fontsize=12)
    ax_portfolio.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax_portfolio.grid(True, alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
    
    ax_portfolio.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x:,.2f}"))
    
    plt.setp(ax_portfolio.get_xticklabels(), rotation=45, ha='right')
    
    position_history = {stock: np.zeros(len(dates)) for stock in stock_nums}
    
    for stock in stock_nums:
        position_history[stock][0] = 0 
    
    for transaction in transactions:
        day = transaction["day"]
        stock = transaction["stock"]
        if day < len(dates):  
            if transaction["action"] == "BUY":
                for i in range(day, len(dates)):
                    position_history[stock][i] += transaction["quantity"]
            elif transaction["action"] == "SELL":
                for i in range(day, len(dates)):
                    position_history[stock][i] -= transaction["quantity"]
            elif transaction["action"] == "SHORT":
                for i in range(day, len(dates)):
                    position_history[stock][i] -= transaction["quantity"]
            elif transaction["action"] == "COVER":
                  for i in range(day, len(dates)):
                    position_history[stock][i] += transaction["quantity"]
    
    for stock, ax in position_axes.items():
        ax.plot(dates, position_history[stock], label=f"Position", linewidth=2, color='blue')
        
        ax.fill_between(dates, position_history[stock], 0, where=(position_history[stock] > 0), 
                         color='green', alpha=0.3, label="Long")
        ax.fill_between(dates, position_history[stock], 0, where=(position_history[stock] < 0), 
                         color='red', alpha=0.3, label="Short")
        
        ax.set_title(f"Stock {stock} Position History", fontsize=14)
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Shares", fontsize=10)
        ax.grid(True, alpha=0.3, color='gray', linestyle='-', linewidth=0.5)  
        ax.legend(loc='upper right')
        
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig("portfolio_and_positions.png", dpi=300)
    plt.show()


def all_stocks_equal_weight_strategy(prediction_df):
    print("Implementing equal-weight buy and hold strategy for all 10 stocks...")
    
    all_stock_data = {}
    for stock_num in range(1, 11):
        file_name = f"st{stock_num}.csv"
        try:
            df = pd.read_csv(file_name)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            if 'Adj Close' not in df.columns and 'Close' in df.columns:
                df['Adj Close'] = df['Close']
                
            df = df.dropna(subset=['Adj Close'])
            all_stock_data[stock_num] = df
            print(f"  Successfully loaded data for stock {stock_num}")
        except Exception as e:
            print(f"  Error loading data for stock {stock_num}: {str(e)}")
    
    initial_cash = 10_000
    cash_per_stock = initial_cash / 10
    
    dates = prediction_df["Date"].values
    days = len(dates)
    
    positions = {}
    for stock_num in range(1, 11):
        if stock_num in all_stock_data:
            stock_df = all_stock_data[stock_num]
            start_date = dates[0]
            
            price_on_date = stock_df[stock_df['Date'] == pd.Timestamp(start_date)]['Adj Close'].values
            if len(price_on_date) > 0:
                price_day_one = price_on_date[0]
                shares = int(cash_per_stock / price_day_one)
                positions[stock_num] = shares
            else:
                positions[stock_num] = 0
        else:
            positions[stock_num] = 0
    
    initial_investment = 0
    for stock_num, shares in positions.items():
        if stock_num in all_stock_data:
            stock_df = all_stock_data[stock_num]
            price_on_date = stock_df[stock_df['Date'] == pd.Timestamp(dates[0])]['Adj Close'].values
            if len(price_on_date) > 0:
                initial_investment += shares * price_on_date[0]
    
    remaining_cash = initial_cash - initial_investment
    
    portfolio_values = np.zeros(days)
    
    for day in range(days):
        current_date = dates[day]
        portfolio_values[day] = remaining_cash
        
        for stock_num, shares in positions.items():
            if stock_num in all_stock_data and shares > 0:
                stock_df = all_stock_data[stock_num]
                price_on_date = stock_df[stock_df['Date'] == pd.Timestamp(current_date)]['Adj Close'].values
                if len(price_on_date) > 0:
                    portfolio_values[day] += shares * price_on_date[0]
                else:
                    past_prices = stock_df[stock_df['Date'] <= pd.Timestamp(current_date)]['Adj Close'].values
                    if len(past_prices) > 0:
                        portfolio_values[day] += shares * past_prices[-1]
    
    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_value) / initial_value * 100
    
    daily_returns = np.zeros(days-1)
    for i in range(1, days):
        daily_returns[i-1] = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
    
    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
    
    peak = portfolio_values[0]
    max_drawdown = 0
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    max_drawdown *= 100
    
    print(f"\nAll Stocks (1-10) Buy and Hold Strategy Performance Summary:")
    print(f"Initial Portfolio Value: ${initial_value:.2f}")
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    
    return {
        "portfolio_value": portfolio_values,
        "dates": dates,
        "positions": positions,
        "metrics": {
            "initial_value": initial_value,
            "final_value": final_value,
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown
        }
    }

def compare_strategies(trading_results, all_stocks_results):
    plt.figure(figsize=(16, 10))
    ax = plt.gca()
    
    dates = trading_results["dates"]
    
    if isinstance(dates[0], np.datetime64):
        dates = pd.to_datetime(dates)
    
    ax.plot(dates, trading_results["portfolio_value"], label="Active Trading Strategy", color="blue", linewidth=2)
    ax.plot(dates, all_stocks_results["portfolio_value"], label="Equal-Weight Buy & Hold (1-10)", color="red", linewidth=2)
    
    trading_metrics = trading_results["metrics"]
    all_stocks_metrics = all_stocks_results["metrics"]
    
    trading_text = (
        f"Active Trading Strategy:\n"
        f"  Total Return: {trading_metrics['total_return']:.2f}%\n"
        f"  Sharpe Ratio: {trading_metrics['sharpe_ratio']:.2f}\n"
        f"  Max Drawdown: {trading_metrics['max_drawdown']:.2f}%"
    )
    
    all_stocks_text = (
        f"Buy & Hold Strategy (1-10):\n"
        f"  Total Return: {all_stocks_metrics['total_return']:.2f}%\n"
        f"  Sharpe Ratio: {all_stocks_metrics['sharpe_ratio']:.2f}\n"
        f"  Max Drawdown: {all_stocks_metrics['max_drawdown']:.2f}%"
    )
    
    return_diff = trading_metrics['total_return'] - all_stocks_metrics['total_return']
    sharpe_diff = trading_metrics['sharpe_ratio'] - all_stocks_metrics['sharpe_ratio']
    drawdown_diff = trading_metrics['max_drawdown'] - all_stocks_metrics['max_drawdown']
    
    comparison_text = (
        f"Strategy Comparison:\n"
        f"Active vs Buy&Hold (1-10):\n"
        f"  Return Difference: {return_diff:.2f}%\n"
        f"  Sharpe Ratio Difference: {sharpe_diff:.2f}\n"
        f"  Max Drawdown Difference: {drawdown_diff:.2f}%"
    )
    
    ax.annotate(trading_text, xy=(0.02, 0.96), xycoords='axes fraction', 
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                fontsize=10, verticalalignment='top')
    
    ax.annotate(all_stocks_text, xy=(0.35, 0.96), xycoords='axes fraction', 
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                fontsize=10, verticalalignment='top')
    
    ax.annotate(comparison_text, xy=(0.68, 0.96), xycoords='axes fraction', 
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                fontsize=10, verticalalignment='top')
    
    ax.set_title("Portfolio Performance Comparison", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax.grid(True, alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
    ax.legend(loc='lower right')
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x:,.2f}"))
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig("strategy_comparison.png", dpi=300)
    plt.show()

models = load_models(model_files=["st5_lstm_model", "st7_lstm_model", "st8_lstm_model"])

if len(models) == 0:
    print("Error: No models were loaded successfully.")

# Load stock data for stocks 5, 7, 8 which had the best MAPE
stock_data = load_stock_data(stock_numbers=[5, 7, 8])

if len(stock_data) == 0:
    print("Error: No stock data was loaded successfully.")

backtest_results = backtest_strategy(models, stock_data)

plot_backtest_results(backtest_results)

prediction_df = create_prediction_dataframe(backtest_results)

prediction_df.to_csv("stock_predictions.csv", index=False)
print("Saved predictions to stock_predictions.csv")

trading_results = trading_strategy(prediction_df)
plot_portfolio_performance(trading_results)

all_stocks_results = all_stocks_equal_weight_strategy(prediction_df)

compare_strategies(trading_results, all_stocks_results)

print("\nBacktesting complete!")