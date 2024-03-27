import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from datetime import datetime

# Title of the Streamlit app
st.title('Ripple Price Predictor')

# Downloading Ripple (XRP) data using yfinance
end = datetime.now()
start = datetime(end.year - 7, end.month, end.day)
stock = "XRP-USD" 
xrp_data = yf.download(stock, start, end)

# Displaying Ripple data
st.subheader('Ripple Data')
st.write(xrp_data)

# Loading the pre-trained Keras model
model = load_model('xrp_price_model.keras')

# Splitting the data into train and test sets
splitting_len = int(len(xrp_data) * 0.9)
x_test = pd.DataFrame(xrp_data.Close[splitting_len: ])

# Displaying original close price
st.subheader('Original Close Price')
figsize = (15, 6)
fig = plt.figure(figsize=figsize)
plt.plot(xrp_data.Close, 'b')
st.pyplot(fig)

# Displaying test close price
st.subheader('Test Close Price')
st.write(x_test)

fig = plt.figure(figsize=figsize)
plt.plot(x_test, 'b')
st.pyplot(fig)

# Scaling the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaler_data = scaler.fit_transform(x_test[['Close']].values)

# Creating sequences for prediction
x_data = []
y_data = []

for i in range(100, len(scaler_data)):
    x_data.append(scaler_data[i-100:i])
    y_data.append(scaler_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Making predictions using the model
predictions = model.predict(x_data)
inverse_predictions = scaler.inverse_transform(predictions)
inverse_y_test = scaler.inverse_transform(y_data)

# Displaying original vs predicted Ripple price
plotting_data = pd.DataFrame({
    'original_test_data': inverse_y_test.reshape(-1),
    'predictions': inverse_predictions.reshape(-1),
}, index = xrp_data.index[splitting_len+100:])

st.subheader('Original vs Predicted Ripple Price')
st.write(plotting_data)

fig = plt.figure(figsize=figsize)
plt.plot(pd.concat([xrp_data.Close[:splitting_len+100], plotting_data], axis=0))
plt.legend(['Data-not used', "Original test data", "Predicted test data"])
st.pyplot(fig)

# Function to predict future close values
last_100 = xrp_data[['Close']].tail(100)
last_100 = scaler.fit_transform(last_100['Close'].values.reshape(-1,1)).reshape(1, -1, 1)
last_100 = last_100.tolist()

def predict_future(no_of_days, previous_100):
    future_predictions = []

    for i in range(no_of_days):
        next_day = model.predict(np.array(previous_100)).tolist()
        previous_100[0].append(next_day[0])
        previous_100 = [previous_100[0][1:]]
        future_predictions.append(scaler.inverse_transform(next_day))
    return future_predictions

# User input for the number of days to predict
no_of_days = int(st.text_input('Enter number of days to predict: ', "10"))
future_results = predict_future(no_of_days, last_100)

# Plotting future close values
future_results = np.array(future_results).reshape(-1, 1)
fig = plt.figure(figsize= (15, 6))
plt.plot(future_results, marker='o')
for i in range(len(future_results)):
    price = future_results[i][0]  
    plt.text(i, price, str(round(float(price), 2)), ha='center', va='bottom')
plt.xlabel('Future Days')
plt.ylabel('Close Price')
plt.title('Future Close Price of XRP')
st.pyplot(fig)
