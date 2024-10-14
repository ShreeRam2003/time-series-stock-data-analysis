# Stock Price Analysis & Visualization

This is a Streamlit-based web app for interactive stock price analysis and visualization. Upload stock data in CSV format, select a company by its ticker, and explore time series plots, resampled data, autocorrelation, stationarity tests, and more.

## Features
- Time series plots for stock prices
- Monthly resampling of data
- ADF test for stationarity
- Moving average smoothing
- Autocorrelation plot for stock volume

## How to Run the App Locally
1. Clone the repository: 
   ```bash
   git clone https://github.com/ShreeRam2003/time-series-stock-data-analysis.git
   ```
2. Install dependencies: 
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app: 
   ```bash
   streamlit run app.py
   ```