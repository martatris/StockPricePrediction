import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, Flatten

# Step 2: Define Tech Stocks
tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']

# Step 3: Download Historical Data
data_dict = {}
for ticker in tech_list:
    data = yf.download(ticker, start='2015-01-01', end='2025-01-01')
    data_dict[ticker] = data[['Close']]

sequence_length = 60


# Step 4: Preprocessing Function
def create_sequences(data, seq_len=60):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data.values)
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len:i])
        y.append(scaled[i, 0])
    return np.array(X), np.array(y), scaler


# Step 5: Build Deep Learning Models
def build_lstm(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def build_gru(input_shape):
    model = Sequential([
        GRU(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def build_cnn(input_shape):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Step 6: Evaluation Function (final day only)
def evaluate_and_print_final(actual, predictions, stock_name):
    print(f"\n=== Stock: {stock_name} ===\n")
    for model_name, pred in predictions.items():
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        last_actual = actual[-1][0]
        last_pred = pred[-1][0]
        direction = "increase" if last_pred > last_actual else "decrease"
        diff = last_pred - last_actual
        print(f"Model: {model_name}")
        print(f"  Last actual price: {last_actual:.2f}")
        print(f"  Last predicted price: {last_pred:.2f}")
        print(f"  Predicted direction: {direction}, difference: {diff:.2f}")
        print(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}\n")


# Step 7: Train and Predict
results = {}

for ticker in tech_list:
    df = data_dict[ticker]

    # Deep Learning Data
    X, y, scaler = create_sequences(df, sequence_length)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    X_train_dl = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_dl = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Deep Learning Models
    models_dl = {
        'LSTM': build_lstm((X_train_dl.shape[1], 1)),
        'GRU': build_gru((X_train_dl.shape[1], 1)),
        '1D_CNN': build_cnn((X_train_dl.shape[1], 1))
    }

    predictions = {}

    for name, model in models_dl.items():
        model.fit(X_train_dl, y_train, epochs=20, batch_size=32, verbose=0)
        pred = model.predict(X_test_dl)
        predictions[name] = scaler.inverse_transform(pred.reshape(-1, 1))

    # Traditional ML Models
    X_ml = np.array([X[:, -1, 0]]).T
    X_train_ml, X_test_ml = X_ml[:split], X_ml[split:]

    lr = LinearRegression().fit(X_train_ml, y_train)
    rf = RandomForestRegressor(n_estimators=100).fit(X_train_ml, y_train)
    xgb = XGBRegressor(n_estimators=100, verbosity=0).fit(X_train_ml, y_train)

    predictions['Linear_Regression'] = lr.predict(X_test_ml).reshape(-1, 1)
    predictions['Random_Forest'] = rf.predict(X_test_ml).reshape(-1, 1)
    predictions['XGBoost'] = xgb.predict(X_test_ml).reshape(-1, 1)

    results[ticker] = {'actual': scaler.inverse_transform(y_test.reshape(-1, 1)), 'predictions': predictions}

    # Step 8: Print final day numeric results
    evaluate_and_print_final(results[ticker]['actual'], results[ticker]['predictions'], ticker)

    # Step 9: Plot for visualization
    plt.figure(figsize=(12, 6))
    plt.plot(results[ticker]['actual'], label='Actual', color='black', linewidth=2)
    for model_name, pred in results[ticker]['predictions'].items():
        plt.plot(pred, label=model_name)
    plt.title(f'{ticker} Stock Price Prediction Comparison')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()