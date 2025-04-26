# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 18:11:19 2025

@author: tomls
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import glob
import os

np.random.seed(42)
tf.random.set_seed(42)

def load_data(csv_file):
    df = pd.read_csv(csv_file)
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
    
    if 'Adj Close' not in df.columns:
        if 'Close' in df.columns:
            df['Adj Close'] = df['Close']
        else:
            return None
    
    df = df.dropna(subset=['Adj Close'])
    
    return df

def create_features_and_labels(df, n_lags=10):
    X, y = [], []
    for i in range(n_lags, len(df)):
        X.append(df["Adj Close"].values[i - n_lags:i])
        y.append(df["Adj Close"].values[i])
    return np.array(X).reshape(-1, n_lags, 1), np.array(y)

def train_and_evaluate_model(df, test_days=365, n_lags=10, lstm_units=50, epochs=50, batch_size=32):
    if df is None or "Adj Close" not in df.columns:
        return None
        
    if df["Adj Close"].isnull().any() or len(df) <= test_days + n_lags:
        return None
    
    try:
        X, y = create_features_and_labels(df, n_lags)
        X_train, X_test = X[:-test_days], X[-test_days:]
        y_train, y_test = y[:-test_days], y[-test_days:]
        
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(lstm_units, input_shape=(n_lags, 1)),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        
        y_pred = model.predict(X_test).flatten()
        
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        
        directional_acc = np.mean(
            np.sign(np.diff(y_test)) == np.sign(np.diff(y_pred))
        ) * 100
        
        return {
            "metrics": {
                "rmse": rmse,
                "mape": mape,
                "directional_accuracy": directional_acc,
            },
            "predictions": {
                "true": y_test,
                "pred": y_pred,
                "dates": df.index[-test_days:]
            },
            "model": model
        }
    except Exception as e:
        print(f"Error in model training: {e}")
        return None
    
def find_best_stocks(file_pattern="st*.csv", n_best=3, error_metric="mape"):
    stock_files = glob.glob(file_pattern)
    
    if not stock_files:
        print(f"No files found matching pattern: {file_pattern}")
        return []
    
    print(f"Found {len(stock_files)} stock files")
    
    results = []
    
    for file in stock_files:
        try:
            stock_name = os.path.basename(file).replace('.csv', '')
            print(f"Processing {stock_name}...")
            
            df = load_data(file)
            
            if df is None:
                print(f"  Failed to process {stock_name}: Invalid data format")
                continue
                
            eval_results = train_and_evaluate_model(df)
            
            if eval_results is not None:
                results.append({
                    "stock": stock_name,
                    "metrics": eval_results["metrics"],
                    "predictions": eval_results["predictions"],
                    "model": eval_results["model"] 
                })
                print(f"  RMSE: {eval_results['metrics']['rmse']:.4f}")
                print(f"  MAPE: {eval_results['metrics']['mape']:.2f}%")
                print(f"  Directional Accuracy: {eval_results['metrics']['directional_accuracy']:.2f}%")
            else:
                print(f"  Failed to process {stock_name}: Model training failed")
        except Exception as e:
            print(f"  Error processing {os.path.basename(file)}: {str(e)}")
    
    if error_metric == "directional_accuracy":
        sorted_stocks = sorted(results, key=lambda x: x["metrics"][error_metric], reverse=True)
    else:
        sorted_stocks = sorted(results, key=lambda x: x["metrics"][error_metric])
    
    return sorted_stocks[:n_best]

def plot_predictions(best_stocks):
    """Plot actual vs predicted values for the best stocks"""
    if not best_stocks:
        print("No stocks to plot")
        return
    
    fig, axes = plt.subplots(len(best_stocks), 1, figsize=(12, 4*len(best_stocks)))
    
    if len(best_stocks) == 1:
        axes = [axes]  
    
    for i, stock in enumerate(best_stocks):
        ax = axes[i]
        
        pred_data = stock["predictions"]
        y_true = pred_data["true"]
        y_pred = pred_data["pred"]
        
        ax.plot(range(len(y_true)), y_true, label="Actual", color="blue")
        ax.plot(range(len(y_pred)), y_pred, label="Predicted", color="red", linestyle="--")
        ax.set_title(f"{stock['stock']} - RMSE: {stock['metrics']['rmse']:.4f}, MAPE: {stock['metrics']['mape']:.2f}%")
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

def save_models(best_stocks):
    saved_models = []
    
    for stock_info in best_stocks:
        stock_name = stock_info['stock']
        model = stock_info['model']
        
        print(f"Saving model for {stock_name}...")
        
        try:
            model_path = f"{stock_name}_lstm_model.h5"
            model.save_weights(model_path)
            
            model_json = model.to_json()
            json_path = f"{stock_name}_lstm_model.json"
            with open(json_path, "w") as json_file:
                json_file.write(model_json)
            
            saved_models.append({
                "stock_name": stock_name,
                "weights_path": model_path,
                "architecture_path": json_path
            })
            
            print(f"  Model weights saved as {model_path}")
            print(f"  Model architecture saved as {json_path}")
            
        except Exception as e:
            print(f"  Error saving model for {stock_name}: {str(e)}")
    
    return saved_models

def main():
    print("Finding top 3 stocks with minimal MAPE...")
    best_stocks = find_best_stocks(file_pattern="st*.csv", n_best=3, error_metric="mape")
    
    print("\nTop 3 stocks with minimal MAPE:")
    for i, stock in enumerate(best_stocks):
        print(f"{i+1}. {stock['stock']}:")
        print(f"   RMSE: {stock['metrics']['rmse']:.4f}")
        print(f"   MAPE: {stock['metrics']['mape']:.2f}%")
        print(f"   Directional Accuracy: {stock['metrics']['directional_accuracy']:.2f}%")
    
    plot_predictions(best_stocks)
    
    print("Saving LSTM models for the best stocks...")
    saved_models = save_models(best_stocks)
    
    print("\nSummary of saved models:")
    for model_info in saved_models:
        print(f"{model_info['stock_name']}: Model weights saved to {model_info['weights_path']}")
        print(f"{model_info['stock_name']}: Model architecture saved to {model_info['architecture_path']}")

if __name__ == "__main__":
    main()