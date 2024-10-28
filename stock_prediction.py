import numpy as np
import pandas as pd
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Load Event Data
event_file_path = 'events_data.csv'  # Path to events_data file
event_data = pd.read_csv(event_file_path)

# Convert 'Date' to datetime
event_data['Date'] = pd.to_datetime(event_data['Date'], format='%d/%m/%Y')
event_data = pd.get_dummies(event_data, columns=['Event_Type'])

# Load Stock Data
COMPANY = 'CBA.AX'
TRAIN_START = '2020-01-01'
TRAIN_END = '2023-08-01'
data = yf.download(COMPANY, TRAIN_START, TRAIN_END)

# Prepare scalers
price_scaler = MinMaxScaler(feature_range=(0, 1))
combined_scaler = MinMaxScaler(feature_range=(0, 1))

# Scale stock prices
scaled_data = price_scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Merge stock prices with events data
event_data.set_index('Date', inplace=True)
combined_data = data[['Close']].join(event_data[['Sentiment', 'Intensity'] + [col for col in event_data if col.startswith('Event_Type')]], how='left').fillna(0)

# Scale combined data
scaled_combined_data = combined_scaler.fit_transform(combined_data)

# Prepare training data
PREDICTION_DAYS = 60
x_train, y_train = [], []
for x in range(PREDICTION_DAYS, len(scaled_combined_data)):
    x_train.append(scaled_combined_data[x - PREDICTION_DAYS:x])
    y_train.append(scaled_combined_data[x, 0])  # Predicting the scaled 'Close' price

x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape for LSTM
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))

# Build the LSTM model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(x_train, y_train, epochs=25, batch_size=32, validation_split=0.1)

# Test on unseen data
TEST_START = '2023-08-02'
TEST_END = '2024-07-02'
test_data = yf.download(COMPANY, TEST_START, TEST_END)
actual_prices = test_data['Close'].values
total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

# Prepare test data with event-driven combined data
model_inputs = combined_data[len(combined_data) - len(test_data) - PREDICTION_DAYS:]
model_inputs = combined_scaler.transform(model_inputs)

x_test = []
for x in range(PREDICTION_DAYS, len(model_inputs)):
    x_test.append(model_inputs[x - PREDICTION_DAYS:x])

x_test = np.array(x_test)

# Make predictions
predicted_prices = model.predict(x_test)
predicted_prices = price_scaler.inverse_transform(predicted_prices)

# Multistep prediction
def multistep_prediction(model, input_data, n_steps):
    predictions = []
    current_input = input_data[-PREDICTION_DAYS:]

    if len(current_input) < PREDICTION_DAYS:
        raise ValueError(f"Input data must have at least {PREDICTION_DAYS} entries.")

    for _ in range(n_steps):
        current_input = np.reshape(current_input, (1, PREDICTION_DAYS, combined_data.shape[1]))
        next_pred = model.predict(current_input)
        predictions.append(next_pred[0, 0])

        new_row = np.zeros(combined_data.shape[1])
        new_row[0] = next_pred[0, 0]
        current_input = np.append(current_input[0, 1:], [new_row], axis=0)

    predictions_with_sentiment = np.column_stack((predictions, np.zeros((len(predictions), combined_data.shape[1] - 1))))
    return combined_scaler.inverse_transform(predictions_with_sentiment)[:, 0]

# Plotting functions
def plot_predictions(actual_prices, predicted_prices):
    plt.figure(figsize=(14, 7))
    plt.plot(actual_prices, label='Actual Prices', color='blue')
    plt.plot(predicted_prices, label='Predicted Prices', color='orange')
    plt.title('Actual vs Predicted Prices')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

plot_predictions(actual_prices, predicted_prices)

# Predict next day
real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], combined_data.shape[1]))

prediction = model.predict(real_data)
prediction = price_scaler.inverse_transform(prediction)
print(f"Prediction for the next day: {prediction[0][0]}")

# Multistep predictions
n_steps = 5
multistep_preds = multistep_prediction(model, model_inputs, n_steps)
print(f"Multistep predictions for the next {n_steps} days: {multistep_preds.flatten()}")

# Candlestick and boxplot plotting
def plot_candlestick(data, n_days=1):
    resampled_data = data.resample(f'{n_days}D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    })
    resampled_data.dropna(inplace=True)
    mpf.plot(resampled_data, type='candle', style='charles', title='Candlestick Chart')

def plot_boxplot(data, n_days=1):
    resampled_data = data.resample(f'{n_days}D').agg({'Close': 'median'})
    resampled_data.dropna(inplace=True)
    resampled_data['Period'] = resampled_data.index.to_period('W').astype(str)

    plt.figure(figsize=(12, 6))
    plt.boxplot(
        [resampled_data['Close'][resampled_data['Period'] == period] for period in resampled_data['Period'].unique()],
        labels=resampled_data['Period'].unique())
    plt.title('Boxplot Chart')
    plt.xlabel('Period')
    plt.ylabel('Closing Price')
    plt.xticks(rotation=45)
    plt.show()

plot_candlestick(data, n_days=5)
plot_boxplot(data, n_days=5)

