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

