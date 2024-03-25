import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
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