import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from datetime import datetime

# Set title for the Streamlit app
st.title("Ripple Price Predictor")

# Define the function to predict future closing prices
def predict_future(no_of_days, previous_100, model, scaler):
    future_predictions = []
    for i in range(no_of_days):
        # Reshape previous_100 to match the expected input shape of the model
        previous_100_reshaped = np.reshape(previous_100, (1, 100, 1))
        
        # Predict the next day's value
        next_day = model.predict(previous_100_reshaped).flatten()[0]
        
        # Update previous_100 for the next prediction
        previous_100 = np.append(previous_100[1:], [[next_day]], axis=0)

        # Inverse transform the predicted value and append to future_predictions
        future_predictions.append(scaler.inverse_transform([[next_day]])[0][0])
    return future_predictions

# Load the pre-trained LSTM model
model = load_model("xrp_price_model.keras")

# Fetch historical data for XRP
stock = "XRP-USD"
end = datetime.now()
start = datetime(end.year - 10, end.month, end.day)
stock = st.text_input("Enter the Stock here: ", stock)
xrp_data = yf.download(stock, start, end)

# Display XRP data and plot the original closing prices
st.subheader("XRP Data")
st.write(xrp_data)
figsize = (15, 6)
fig = plt.figure(figsize=figsize)
plt.plot(xrp_data["Close"], 'b')
st.pyplot(fig)

# Extract test data and plot test closing prices
splitting_len = int(len(xrp_data) * 0.9)
x_test = pd.DataFrame(xrp_data.iloc[splitting_len:])
st.subheader('Test Close price')
st.write(x_test)
fig = plt.figure(figsize=figsize)
plt.plot(x_test["Close"], 'b')
st.pyplot(fig)

# Preprocess data for prediction
scaler = MinMaxScaler(feature_range=(0, 1))
scaler_data = scaler.fit_transform(x_test[['Close']].values)
x_data = []
y_data = []
for i in range(100, len(scaler_data)):
    x_data.append(scaler_data[i-100:i, 0])
    y_data.append(scaler_data[i, 0])

x_data, y_data = np.array(x_data), np.array(y_data)

# Make predictions
predictions = model.predict(x_data)
inverted_predictions = scaler.inverse_transform(predictions)
inverted_y_test = scaler.inverse_transform(y_data.reshape(-1, 1))

# Prepare data for plotting
plotting_data = pd.DataFrame({
    'original_test_data': inverted_y_test.reshape(-1),
    'predictions': inverted_predictions.reshape(-1)
}, 
    index = xrp_data.index[splitting_len + 100:]
)

# Plot original vs predicted values
st.subheader("Original vs Predicted value")
st.write(plotting_data)
fig = plt.figure(figsize=(15, 6))
plt.plot(pd.concat([xrp_data["Close"][:splitting_len+100], plotting_data], axis=0))
plt.legend(["Data- not used", "Original test data", "Predicted test data"])
st.pyplot(fig)

# Predict for tomorrow (next day) only
no_of_days = 3

# Predict future prices
last_100 = xrp_data['Close'].tail(100).values.reshape(-1, 1)
last_100 = scaler.fit_transform(last_100)
previous_100 = np.copy(last_100)
future_results = predict_future(no_of_days, previous_100, model, scaler)
future_results = np.array(future_results).reshape(-1, 1)

# Prepare future date for plotting
future_date = pd.date_range(start=xrp_data.index[-1], periods=no_of_days+1, closed='right')[1]

# Display predicted future price for tomorrow
future_data = pd.DataFrame({
    'Date': [future_date],
    'Predicted Close Price': future_results.reshape(-1)
})

st.subheader("Predicted Closing Price for Tomorrow")
st.write(future_data)

# Plot future predicted price for tomorrow
fig = plt.figure(figsize=(10, 6))
plt.plot([future_date], future_results, marker='o', color='red')
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.title('Predicted Closing price of XRP for Tomorrow')
plt.grid(True)
st.pyplot(fig)
