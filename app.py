import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

st.title("Stock Price Prediction")

tickers = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'META', 'BRK.A', 'BRK.B', 'JNJ', 'V',
    'PG', 'JPM', 'TSLA', 'WMT', 'MA', 'KO', 'NVDA', 'T', 'HD', 'PFE',
    'DIS', 'BAC', 'VZ', 'ADBE', 'NFLX', 'IBM', 'INTC', 'UNH', 'CVX',
    'MCD', 'CSCO', 'CAT', 'AMGN', 'GS', 'MMM', 'ABBV', 'JCI', 'F', 'XOM',
    'BMY', 'BA', 'GE', 'GM', 'DD', 'CVS', 'COST', 'ABT', 'PEP', 'TGT', 'HD', 'MS', 'LLY'
]

# User input
trend_choice = st.selectbox("Select Trend", ["Bullish", "Bearish"])
ema_choice = st.selectbox("Select Price condition compared to the Exponential Moving Average (17)", ["Above", "Below"])
rsi_choice = st.selectbox("Select RSI Condition", ["Overbought", "Oversold", "Neither"])

if st.button("Submit"):

    result = []
    
    for ticker in tickers:
        try:
            data = yf.download(ticker, period="200d")
            data['PCT'] = data['Close'].pct_change()
    
            data['EMA17'] = data['Close'].rolling(window=17).mean()
            last_ema17 = data['EMA17'][-1]
            last_close = data['Close'][-1]
            if last_ema17 >= last_close:
                ema_condition = "Below"
            else:
                ema_condition = "Above"
    
            data['Trend'] = 'Bearish'
            data['EMA34'] = data['Close'].rolling(window=34).mean()
            data.loc[data['EMA17'] > data['EMA34'], 'Trend'] = 'Bullish'
            trend_condition = data['Trend'].tail(1).values[0]
    
            def calculate_rsi(data):
                delta = data['Close'].diff(1)
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
    
                avg_gain = gain.rolling(window=14, min_periods=1).mean()
                avg_loss = loss.rolling(window=14, min_periods=1).mean()
    
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
    
                return rsi
    
            data['RSI'] = calculate_rsi(data)
            last_rsi = data['RSI'].tail(1).values[0]
            overbought_threshold = 70
            oversold_threshold = 30
            rsi_condition = ""
            if last_rsi > overbought_threshold:
                rsi_condition = "Overbought"
            elif last_rsi < oversold_threshold:
                rsi_condition = "Oversold"
            else:
                rsi_condition = "Neither"
    
            if rsi_condition == rsi_choice and trend_condition == trend_choice and ema_condition == ema_choice:
                result.append(ticker)
    
        except:
            pass
    if not result:
        st.write(f"None of the following verified stocks meet the selected condition: {tickers}")
        st.write(tickers)
    else:
        for ticker in result:
            try:
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.title(f"Stock symbol: {ticker}")
                data = yf.download(ticker, period="800d")
                data = data[["Close"]]
        
                scaler = MinMaxScaler()
                data_scaled = scaler.fit_transform(data.values)
        
                train_size = int(len(data_scaled) * 0.8)
                train_data = data_scaled[:train_size]
                test_data = data_scaled[train_size:]
        
                n_steps = 7
                n_features = data.shape[1]
        
                X_train, y_train = [], []
                for i in range(n_steps, len(train_data)):
                    X_train.append(train_data[i - n_steps:i])
                    y_train.append(train_data[i, 0])
                X_train, y_train = np.array(X_train), np.array(y_train)
        
                X_test, y_test = [], []
                for i in range(n_steps, len(test_data)):
                    X_test.append(test_data[i - n_steps:i])
                    y_test.append(test_data[i, 0])
                X_test, y_test = np.array(X_test), np.array(y_test)
        
                model = Sequential()
                model.add(LSTM(units=50, activation='tanh', return_sequences=True, input_shape=(n_steps, n_features)))
                model.add(Dropout(0.2))
                model.add(LSTM(units=50, activation='tanh'))
                model.add(Dropout(0.2))
                model.add(Dense(units=1))
        
                model.compile(optimizer='adam', loss='mean_squared_error')
        
                model.fit(X_train, y_train, epochs=10, batch_size=32)
        
                predictions = model.predict(X_test)
        
                predictions_unscaled = scaler.inverse_transform(predictions)
        
                
                st.write("LSTM - Test result")
                plt.figure(figsize=(12, 6))
                plt.plot(data.index[train_size + n_steps:], data['Close'].values[train_size + n_steps:], label='Actual')
                plt.plot(data.index[train_size + n_steps:], predictions_unscaled, label='Predicted')
                plt.title(f'{ticker} - LSTM Test Result')
                plt.xlabel('Date')
                plt.ylabel('Closing Price')
                plt.legend()
                st.pyplot()
        
                last_sequence = X_train[-1]
        
                predicted_values = []
        
                days = 15
                for i in range(days):
                    input_sequence = last_sequence.reshape(1, n_steps, n_features)
                    predicted_value = model.predict(input_sequence)[0][0]
                    predicted_values.append(predicted_value)
                    last_sequence = np.append(last_sequence[1:], predicted_value)
        
                predicted_values_unscaled = scaler.inverse_transform(np.array(predicted_values).reshape(-1, 1))
                today = datetime.today()
        
                next_days = []
                for i in range(days):
                    day = today + timedelta(days=i)
                    day = day.strftime('%Y-%m-%d')
                    next_days.append(day)
        
                st.write(f"Predicted Prices for Next {days} Days")
                plt.figure(figsize=(12, 6))
                plt.plot(next_days, predicted_values_unscaled, label='Predicted')
                plt.title(f'{ticker} Price Prediction for Next {days} Days')
                plt.xlabel('Date')
                plt.ylabel('Closing Price')
                plt.legend()
                plt.xticks(rotation=45)
                st.pyplot()
            except:
                pass
