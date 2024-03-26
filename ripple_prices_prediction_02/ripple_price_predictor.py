import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from datetime import datetime

st.title("Ripple Price Predictor")

stock = "XRP-USD"

end = datetime.now()
start = datetime(end.year - 10, end.month, end.day)

stock = st.text_input("Enter the Stock here: ", stock)

xrp_data = yf.download(stock, start, end)

model = load_model("xrp_price_model.keras")
st.subheader("XRP Data")
st.write(xrp_data)

splitting_len = int(len(xrp_data) * 0.9)
x_test = pd.DataFrame(xrp_data.iloc[splitting_len:])

st.subheader("Original Close price")
figsize = (15, 6)
fig = plt.figure(figsize=figsize)
plt.plot(xrp_data["Close"], 'b')
st.pyplot(fig)

st.subheader('Test Close price')
st.write(x_test)

st.subheader("Test Close price")
figsize = (15, 6)
fig = plt.figure(figsize=figsize)
plt.plot(x_test["Close"],'b')
st.pyplot(fig)


# preprocessing data
scaler = MinMaxScaler(feature_range=(0, 1))
scaler_data = scaler.fit_transform(x_test[['Close']].values)

x_data = []
y_data = []
for i in range(100, len(scaler_data)):
    x_data.append(scaler_data[i-100:i, 0])
    y_data.append(scaler_data[i, 0])

x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)
inverted_predictions = scaler.inverse_transform(predictions)
inverted_y_test = scaler.inverse_transform(y_data.reshape(-1, 1))

ploting_data = pd.DataFrame({
    'original_test_data': inverted_y_test.reshape(-1),
    'predictions': inverted_predictions.reshape(-1)
}, 
    index = xrp_data.index[splitting_len + 100:]
)
st.subheader("Original vs Predicted value")
st.write(ploting_data)

st.subheader("Original close price vs Predicted close price")
fig = plt.figure(figsize=(15, 6))
plt.plot()

plt.plot(pd.concat([xrp_data["Close"][:splitting_len+100], ploting_data], axis=0))
plt.legend(["Data- not used", "Original test data", "Predicted test data"])
st.pyplot(fig)

st.subheader("Future price value")

# Correct the shape of last_100
last_100 = xrp_data[['Close']].tail(100)
last_100 = scaler.fit_transform(last_100['Close'].values.reshape(-1, 1))
previous_100 = np.copy(last_100)  # Remove the extra list here

def predict_future(no_of_days, previous_100, model, scaler):
    future_predictions = []
    for i in range(no_of_days):
        print("Previous 100 shape:", previous_100.shape)  # Debug statement
        # Reshape previous_100 to match the expected input shape of the model
        previous_100_reshaped = np.reshape(previous_100, (1, 100, 1))
        
        # Predict the next day's value
        next_day = model.predict(previous_100_reshaped).flatten()[0]
        print("Next day:", next_day)  # Debug statement
        
        # Update previous_100 for the next prediction
        previous_100 = np.append(previous_100[0][1:], [[next_day]], axis=0)
        print("Updated previous 100 shape:", previous_100.shape)  # Debug statement

        # Inverse transform the predicted value and append to future_predictions
        future_predictions.append(scaler.inverse_transform([[next_day]])[0][0])
    return future_predictions


no_of_days = int(st.text_input("Enter the number of days you want to predict: ", 10))
future_results = predict_future(no_of_days, previous_100, model, scaler)
future_results = np.array(future_results).reshape(-1, 1)
st.write(future_results)
fig = plt.figure(figsize=(15, 6))
plt.plot(pd.DataFrame(future_results), marker='o', color='red')
for i in range(len(future_results)):
    plt.text(i, future_results[i], int(future_results[i][0]))
plt.xlabel("Days")
plt.ylabel("Close Price")
plt.xticks(range(no_of_days))
plt.yticks(range(int(min(future_results)), int(max(future_results)), 100))
plt.title('Closing price of XRP')
st.pyplot(fig)
