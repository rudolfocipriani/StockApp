import requests
import json
import streamlit as st
from datetime import date, timedelta 
import pandas as pd
import yfinance as yf 
from prophet import Prophet 
from prophet.plot import plot_plotly 
from plotly import graph_objs as go 
import scipy
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import register_matplotlib_converters
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import register_matplotlib_converters
import itertools
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
register_matplotlib_converters()


API_KEY = 'YB3L9H497PDWJJ5K4'

st.title("💰🚀 Stock Analizer App 🚀💰")

# Set the minimum and maximum start date values
min_date = date(2010, 1, 1)
max_date = date.today() - timedelta(days=1)

# Get the start date from the user using a slider
START= st.slider(
    'Select a start date:',
    min_value=min_date,
    max_value=max_date,
    value=date(2015, 1, 1),
    format="YYYY-MM-DD"
)

# Print the selected start date
st.write('Start date selected:', START.strftime('%Y-%m-%d'))

TODAY = date.today()

def get_stock_ticker(company_name):
    url = f'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={company_name}&apikey={API_KEY}'
    response = requests.get(url)
    data = response.json()

    if 'bestMatches' not in data:
        print(f"No results found for '{company_name}'.")
        return None

    matches = data['bestMatches']
    if len(matches) == 0:
        print(f"No results found for '{company_name}'.")
        return None

    best_match = matches[0]
    ticker = best_match['1. symbol']
    name = best_match['2. name']
    print(f"Best Match for '{company_name}': {ticker} ({name})")

    return ticker

company_name = st.text_input("Enter company name", 'Apple Inc')
ticker = get_stock_ticker(company_name)

if ticker is not None:
    selected_stock = ticker
    st.write(f'Selected stock for prediction is {selected_stock}')
else:
    st.error(f"No results found for '{company_name}'.")

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365


@st.cache_data
def load_data(ticker, START): 
    data = yf.download(ticker, start=START, end=TODAY, repair=True)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text(f"Loading data for: {company_name} ({ticker}) from {START} to {TODAY}...")
data = load_data(ticker, START) 

if data.shape[0] < 2 or data.isna().sum().sum() > 0:
    st.error(f"Insufficient or invalid data for: {company_name} from {START} to {TODAY}. Please try another ticker or date range.")
else:
    data_load_state.text(f'Loading data for: {company_name} from {START} to {TODAY} is done!')
    # Get Ticker object for the given ticker
    ticker_info = yf.Ticker(ticker)

    # Fetch company summary
    company_summary = ticker_info.info['longBusinessSummary']

    # Display the company summary
    st.subheader('Company Summary:')
    st.write(company_summary)

    st.subheader('Raw Data')
    st.write(data.head())

    def plot_raw_data(): 
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
        fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
        
    plot_raw_data()


def calculate_moving_averages(data):
    """Calculate and plot the 50-day and 200-day moving averages."""
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()

    st.subheader('Moving Averages')
    fig, ax = plt.subplots(figsize=(16,9))
    
    ax.plot(data['Date'], data['Close'], label='Close Price')
    ax.plot(data['Date'], data['MA50'], label='50-day Moving Average')
    ax.plot(data['Date'], data['MA200'], label='200-day Moving Average')
    
    ax.legend(loc='best')
    ax.set_title('Moving Averages: 50-day & 200-day')
    st.pyplot(fig)

# After loading and plotting the raw data...
calculate_moving_averages(data)

def prepare_data(df):
    """Prepare the data for Random Forest Regressor."""
    df = df[['Date', 'Close']].copy()
    df.set_index('Date', inplace=True)

    for i in range(1, 6):
        df[f'Lag_{i}'] = df['Close'].shift(i)

    df.dropna(inplace=True)

    return df

def split_data(df):
    """Split the data into features (X) and target variable (y)."""
    X = df.drop('Close', axis=1)
    y = df['Close']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X, y):
    """Train the Random Forest Regressor model."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


def prepare_future_data(df, period):
    """Prepare lagged features for prediction."""
    last_date = df.index[-1]
    last_close = df['Close'].iloc[-1]
    lagged_features = []

    for i in range(1, 6):
        lagged_value = df.loc[last_date - pd.Timedelta(days=i)]['Close'] if last_date - pd.Timedelta(days=i) in df.index else last_close
        lagged_features.append(lagged_value)

    future_dates = pd.date_range(start=last_date, periods=period, freq='D')
    return future_dates, lagged_features

def plot_forecast(data, forecast):
    """Plot the forecast."""
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(data.index, data['Close'], label='Close Price')
    ax.plot(forecast['Date'], forecast['Close'], label='Forecast')
    ax.legend(loc='best')
    ax.set_title('Stock Price Forecast (Random Forest Regressor)')
    plt.show()

def make_predictions(model, future_dates, lagged_features, period):
    """Make predictions for the future dates."""
    predictions = []
    for _ in range(period):
        # Convert lagged_features to DataFrame
        input_data = pd.DataFrame([lagged_features], columns=[f'Lag_{i}' for i in range(1, 6)])

        prediction = model.predict(input_data)
        predictions.append(prediction[0])

        lagged_features.pop(0)
        lagged_features.append(prediction[0])

    return pd.DataFrame({'Date': future_dates, 'Close': predictions})


# Load and validate the data
data = load_data(ticker, START)
assert data.shape[0] > 2, f"Insufficient data for: {company_name} from {START} to {TODAY}. Please try another ticker or date range."
assert data.isna().sum().sum() == 0, f"Invalid data for: {company_name} from {START} to {TODAY}. Please try another ticker or date range."

# Prepare the data
prepared_data = prepare_data(data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = split_data(prepared_data)

# Train the Random Forest Regressor model
model = train_model(X_train, y_train)

# Evaluate the model on the testing set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"The Mean Squared Error of the model is: {mse}")

# Prepare future data for prediction
period = 10  # Number of days to forecast
future_dates, lagged_features = prepare_future_data(prepared_data, period)

# Make predictions for the future dates
forecast = make_predictions(model, future_dates, lagged_features, period)

# Plot the forecast
plot_forecast(prepared_data, forecast)


# Streamlit app starts here
st.subheader('Stock Price Forecast (Random Forest Regressor)')

if data.shape[0] > 2 and data.isna().sum().sum() == 0:

    # Prepare the data
    prepared_data = prepare_data(data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(prepared_data)

    # Train the Random Forest Regressor model
    model = train_model(X_train, y_train)

    # Evaluate the model on the testing set
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"The Mean Squared Error of the model is: {mse}")

    # Prepare future data for prediction
    period = st.number_input('Number of days to forecast:', min_value=1, max_value=365, value=10)
    future_dates, lagged_features = prepare_future_data(prepared_data, period)

    # Make predictions for the future dates
    forecast = make_predictions(model, future_dates, lagged_features, period)
    
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(prepared_data.index, prepared_data['Close'], label='Historical Close Price')
    ax.plot(forecast['Date'], forecast['Close'], label='Forecast Close Price')
    ax.legend(loc='best')
    ax.set_title('Stock Price Forecast (Random Forest Regressor)')
    st.pyplot(fig)

else:
    st.write(f"Insufficient or invalid data for: {ticker} from {START}. Please try another ticker or date range.")