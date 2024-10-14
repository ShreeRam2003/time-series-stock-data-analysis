import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
import streamlit as st
import plotly.express as px

# Set a wider layout
st.set_page_config(layout="wide", page_title="Stock Price Analysis")

# Function to load data
def load_data(file):
    # Read CSV, ensuring the 'Date' column is parsed and set as the index
    df = pd.read_csv(file, parse_dates=['Date'], index_col='Date')
    
    # Remove timezone information (if present) and ensure proper conversion to DatetimeIndex
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
    
    # Remove unnecessary columns
    df.drop(columns=['Dividends', 'Stock Splits', 'Capital Gains'], errors='ignore', inplace=True)
    
    return df

# Sidebar content for uploading files
st.sidebar.title("Stock Data Analysis")
st.sidebar.markdown("Upload your stock data in CSV format to visualize and analyze it.")

uploaded_file = st.sidebar.file_uploader("Upload your stock data CSV", type=["csv"])

# Title and introduction
st.title("📈 Stock Price Analysis & Visualization")
st.markdown("""
This application allows you to upload your stock data and analyze it interactively.
Use the sidebar to upload your file, and the graphs below will provide detailed insights into the stock's performance.
""")

# Styling - Dark theme using Markdown
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
    }
    .css-18e3th9 {
        background-color: #283747;
        color: #fff;
    }
    .css-1d391kg {
        background-color: #283747;
        color: #fff;
    }
    </style>
    """, unsafe_allow_html=True)

# Main function for the Streamlit app
def main():
    if uploaded_file is not None:
        df = load_data(uploaded_file)

        # Get a list of unique tickers
        unique_tickers = df['Ticker'].unique()

        # Sidebar for Ticker selection
        st.sidebar.subheader("Select a Ticker (Company)")
        selected_ticker = st.sidebar.selectbox("Ticker", unique_tickers)

        # Filter data by the selected ticker
        df_filtered = df[df['Ticker'] == selected_ticker]

        # Sidebar for Column selection (excluding specific columns)
        columns_to_plot = df_filtered.columns.tolist()
        columns_to_plot = [col for col in columns_to_plot if col not in ['Brand_Name', 'Ticker', 'Industry_Tag', 'Country']]
        selected_column = st.sidebar.selectbox("Select column to plot", columns_to_plot, index=0)

        # Dataset Preview
        st.subheader(f"Dataset Preview for {selected_ticker} 📊")
        st.write("Here’s a preview of the first five rows of your data:")
        st.dataframe(df_filtered.head(), width=1000)

        # Time Series Plot for the selected column
        st.subheader(f"Stock {selected_column} Over Time for {selected_ticker} 📅")
        fig1 = px.line(df_filtered, x=df_filtered.index, y=selected_column, title=f"{selected_ticker} - {selected_column} Over Time",
                       labels={'Date': 'Date', selected_column: selected_column},
                       template='plotly_dark')
        st.plotly_chart(fig1, use_container_width=True)

        # Monthly Resampling for the selected column
        df_resampled = df_filtered.resample('M').mean(numeric_only=True)
        st.subheader(f"Monthly Resampling of {selected_column} for {selected_ticker} 🗓")
        fig2 = px.line(df_resampled, x=df_resampled.index, y=selected_column, title=f"{selected_ticker} - Monthly Average {selected_column} Over Time",
                       labels={'Date': 'Date (Monthly)', selected_column: selected_column},
                       template='plotly_dark')
        st.plotly_chart(fig2, use_container_width=True)

        # ACF Plot for Volume
        st.subheader(f"Autocorrelation of Volume for {selected_ticker} 🔄")
        fig3, ax = plt.subplots(figsize=(12, 6))
        plot_acf(df_filtered['Volume'], lags=40, ax=ax)
        plt.title(f'{selected_ticker} - Autocorrelation Function (ACF) Plot for Volume')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        st.pyplot(fig3)

        # ADF Test for Stationarity
        result = adfuller(df_filtered['High'])
        st.subheader(f"Stationarity Test for {selected_ticker} 📈")
        st.markdown(f"*ADF Statistic*: {result[0]:.5f}")
        st.markdown(f"*p-value*: {result[1]:.5f}")
        st.markdown(f"*Critical Values*: {result[4]}")

        # Differencing for High price
        df_filtered['high_diff'] = df_filtered['High'].diff()
        st.subheader(f"Price Differencing for {selected_ticker} 📉")
        fig4 = px.line(df_filtered, x=df_filtered.index, y=['High', 'high_diff'],
                       labels={'value': 'Price', 'Date': 'Date'},
                       title=f"{selected_ticker} - Original vs Differenced High Price",
                       template='plotly_dark')
        fig4.update_traces(line=dict(width=2))
        st.plotly_chart(fig4, use_container_width=True)

        # Moving Average for High price
        window_size = 120
        df_filtered['high_smoothed'] = df_filtered['High'].rolling(window=window_size).mean()
        st.subheader(f"Moving Average Smoothing for {selected_ticker} 🌐")
        fig5 = px.line(df_filtered, x=df_filtered.index, y=['High', 'high_smoothed'],
                       labels={'value': 'Price', 'Date': 'Date'},
                       title=f"{selected_ticker} - Original vs Moving Average (Window={window_size})",
                       template='plotly_dark')
        st.plotly_chart(fig5, use_container_width=True)

        # Combined DataFrame for comparison
        df_combined = pd.concat([df_filtered['High'], df_filtered['high_diff']], axis=1)
        st.subheader(f"Data Comparison for {selected_ticker} 📄")
        st.write("Here’s a side-by-side comparison of the original and differenced high prices:")
        st.dataframe(df_combined.head(), width=1000)

        # ADF Test on Differenced High
        result_diff = adfuller(df_filtered['high_diff'].dropna())
        st.subheader(f"Stationarity Test on Differenced Data for {selected_ticker} 📊")
        st.markdown(f"*ADF Statistic (Differenced)*: {result_diff[0]:.5f}")
        st.markdown(f"*p-value (Differenced)*: {result_diff[1]:.5f}")
        st.markdown(f"*Critical Values*: {result_diff[4]}")

# Run the app
if __name__ == "__main__":
    main()