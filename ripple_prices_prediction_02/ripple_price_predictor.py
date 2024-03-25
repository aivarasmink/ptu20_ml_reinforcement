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

model = load_model("xrp_price_predictor.keras")
st.subheader("XRP Data")
st.write(xrp_data)

splitting_len = int(len(xrp_data) * 0.9)
x_test = pd.DataFrame(xrp_data.iloc[splitting_len:])

st.subheader("Original Close price")
figsize = (15, 6)
fig = plt.figure(figsize=figsize)
plt.plot(xrp_data["Close"],'b')
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
inverted_y_test = scaler.inverse_transform(y_data)


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

plt.plot(pd.concat([xrp_data.close[:splitting_len+100], ploting_data], axis=0))
plt.legend(["Data- not used", "Original test data", "Predicted test data"])
st.pyplot(fig)

st.subheader("Future price value")
# st.write(ploting_data)

last_100 = xrp_data[['Close']].tail(100)
last_100 = scaler.fit_transform(last_100['close'].values.reshape(-1, 1)).reshape(1, -1, 1)
previuous_100 = np.copy(last_100).tolist()  

def predict_future(no_of_days, previous_100, model, scaler):
    future_predictions = []
    for i in range(no_of_days):
        next_day = model.predict(previous_100).tolist()
        
        # Update previous_100 for the next prediction
        previous_100 = np.append(previous_100, [[next_day[0]]], axis=1)
        previous_100 = previous_100[:, 1:]  # Drop the first column

        # Inverse transform the predicted value and append to future_predictions
        future_predictions.append(scaler.inverse_transform(np.array([next_day]).reshape(-1, 1)))  # Convert next_day to a 2D array
    return future_predictions

no_of_days = int(st.text_input("Enter the number of days you want to predict: ", 10))
future_results = predict_future(no_of_days, previuous_100)
future_results = np.array(future_results).reshape(-1, 1)
print(future_results)
fig = plt.figure(figsize=(15, 6))
plt.plot(pd.DataFrame(future_results), marker='o', color='red')
for i in range(len(future_results)):
    plt.text(i, future_results[i], int(future_results[i][0]))
plt.xlabel("Days")
plt.ylabel("Close Price")
plt.xticks(range(no_of_days))
plt.yticks(range(min(list(map(int, future_results))), max(list(map(int, future_results))), 100))
plt.title('Closing pricr of xrp')
st.pyplot(fig)