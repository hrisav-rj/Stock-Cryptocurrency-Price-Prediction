import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import base64

START = "2008-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
st.title('Crypto Currency')
crypto = (
    'BTC-USD', 'ETH-USD', 'XRP-USD', 'LTC-USD', 'BCH-USD', 
    'ADA-USD', 'DOT-USD', 'LINK-USD', 'BNB-USD', 'DOGE-USD'
)
selected_crypto = st.selectbox('Select Crypto dataset for prediction', crypto, key="crypto_selectbox")

st.title('Stock Forecast App')
stocks = (
    'GOOG', 'AAPL', 'MSFT', 'GME', 'AMZN', 
    'TSLA', 'NFLX', 'FB', 'NVDA', 'INTC'
)
selected_stock = st.selectbox('Select Stock dataset for prediction', stocks, key="stocks_selectbox")

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

@st.cache_resource
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data_stock = load_data(selected_stock)
data_crypto = load_data(selected_crypto)
data_load_state.text('Loading data... done!')

st.subheader('Raw Stock Data')
st.write(data_stock)

st.subheader('Raw Crypto Data')
st.write(data_crypto)

# Plot raw data
def plot_raw_data(data, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close"))
    fig.layout.update(title_text=title, xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data(data_stock, 'Time Series Data for Stocks with Rangeslider')
plot_raw_data(data_crypto, 'Time Series Data for Crypto with Rangeslider')

# Predict forecast with Prophet.
def forecast_data(data):
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)
    return forecast, m

st.subheader('Stock Forecast Data')
forecast_stock, model_stock = forecast_data(data_stock)
st.write(forecast_stock)

st.subheader('Crypto Forecast Data')
forecast_crypto, model_crypto = forecast_data(data_crypto)
st.write(forecast_crypto)

st.write(f'Stock Forecast plot for {n_years} years')
fig1 = plot_plotly(model_stock, forecast_stock)
st.plotly_chart(fig1)

st.write(f'Crypto Forecast plot for {n_years} years')
fig2 = plot_plotly(model_crypto, forecast_crypto)
st.plotly_chart(fig2)

st.write("Stock Forecast components")
fig3 = model_stock.plot_components(forecast_stock)
st.write(fig3)

st.write("Crypto Forecast components")
fig4 = model_crypto.plot_components(forecast_crypto)
st.write(fig4)
