# stock_prediction_ml/main.py

import pandas as pd
import yfinance as yf
import ta
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import streamlit as st
import datetime

# 1. Download stock data
def download_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip() for col in data.columns.values]
    rename_map = {col: col.split('_')[0] for col in data.columns if '_' in col}
    data = data.rename(columns=rename_map)
    data = data.reset_index()
    return data

# 2. Compute technical indicators
def add_technical_indicators(df):
    close = df['Close'].squeeze()
    df['RSI'] = ta.momentum.RSIIndicator(close=close).rsi()
    df['MACD'] = ta.trend.MACD(close=close).macd()
    df['SMA'] = ta.trend.SMAIndicator(close=close, window=14).sma_indicator()
    df['EMA'] = ta.trend.EMAIndicator(close=close, window=14).ema_indicator()
    return df

# 3. Perform sentiment analysis
def get_sentiment_score(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)['compound']

def load_sentiment_data():
    news_data = pd.read_csv("data/sample_news.csv")
    news_data['Sentiment'] = news_data['Headline'].apply(get_sentiment_score)
    daily_sentiment = news_data.groupby('Date')['Sentiment'].mean()
    daily_sentiment.index = pd.to_datetime(daily_sentiment.index)
    return daily_sentiment

# 4. Merge stock with sentiment
def merge_data(df, sentiment):
    sentiment_df = sentiment.reset_index()
    sentiment_df.columns = ['Date', 'Sentiment']
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.merge(sentiment_df, on='Date', how='left')
    df['Sentiment'] = df['Sentiment'].ffill()
    return df

# 5. Label target
def create_target(df):
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df.dropna()

# 6. Train model
def train_model(df, model_type='XGBoost'):
    features = ['RSI', 'MACD', 'SMA', 'EMA', 'Sentiment']
    df = df.dropna(subset=features + ['Target'])
    X = df[features]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    if model_type == 'Random Forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = XGBClassifier(eval_metric='logloss')

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, X_test, y_test, preds

# 7. Dashboard only

def dashboard():
    st.title("Stock Market ML Dashboard")
    ticker = st.text_input("Enter Stock Ticker", "AAPL")
    start = st.date_input("Start Date", pd.to_datetime("2021-01-01"))
    end = st.date_input("End Date", datetime.date.today())
    model_type = st.selectbox("Select Model", ["XGBoost", "Random Forest"])

    df = download_stock_data(ticker, start, end)
    df = add_technical_indicators(df)
    sentiment = load_sentiment_data()
    df = merge_data(df, sentiment)
    df = create_target(df)
    model, X_test, y_test, preds = train_model(df, model_type)

    st.subheader("Prediction Results")
    result_df = pd.DataFrame({"Actual": y_test.reset_index(drop=True), "Predicted": preds})
    st.line_chart(result_df)

    st.subheader("Recent Features and Sentiment")
    preview_df = df.tail(10)[['Date', 'RSI', 'MACD', 'SMA', 'EMA', 'Sentiment']]
    st.dataframe(preview_df)

    st.subheader("Download Data")
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions as CSV", csv, f"{ticker}_predictions.csv", "text/csv")

    live_data = yf.download(ticker, period="1d", interval="1m")
    st.subheader("Live Market Feed")
    st.line_chart(live_data['Close'])

if __name__ == '__main__':
    dashboard()
