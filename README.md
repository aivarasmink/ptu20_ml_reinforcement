# Projects Overview

This repository contains three projects:

1. [Snake Game with Machine Learning](#snake_game_ml)
2. [Ripple Price Predictor from Yahoo Finance and Wikipedia](#ripple_price_prediction)
3. [Ripple Price Predictor from Yahoo Finance with Localhost](#ripple_price_prediction_localhost)

Each project has its own folder in the repository with detailed documentation and code.

## Snake Game with Machine Learning

The Snake Game with Machine Learning project is a Python implementation of the classic Snake game, where the snake is controlled by a neural network using Q-learning. The neural network learns to play the game by trial and error, improving its performance over time.

### Features

- Snake game with graphical interface using Pygame
- Q-learning algorithm for training the neural network
- Neural network predicts the next move based on the current game state
- Performance metrics and visualization of the training process

## Ripple Price Predictor from Yahoo Finance and Wikipedia

The Ripple Price Predictor project utilizes historical price data from Yahoo Finance and additional contextual information from Wikipedia to predict future prices of Ripple (XRP) cryptocurrency.

### Features

- Data retrieval from Yahoo Finance and Wikipedia APIs
- Data preprocessing and feature engineering
- Machine learning models for price prediction
- Evaluation metrics and visualization of prediction results

## Ripple Price Predictor from Yahoo Finance

This project utilizes historical price data from Yahoo Finance to predict future prices of Ripple (XRP) cryptocurrency. It's implemented as a Streamlit web application.

### Overview

The Ripple Price Predictor from Yahoo Finance is a web application built with Streamlit, a Python library for building interactive web apps. It leverages historical price data of Ripple (XRP) obtained from Yahoo Finance to predict future prices using a pre-trained machine learning model.

### Features

- Downloads historical price data of Ripple (XRP) from Yahoo Finance.
- Visualizes the original and test close prices.
- Makes predictions using a pre-trained machine learning model.
- Predicts future close values based on user input for the number of days.

### Installation and Usage

```bash
pip install -r requirements.txt
