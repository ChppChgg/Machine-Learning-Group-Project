from sentiment_score import get_cached_sentiment
from data_prep import get_sp500_tickers
from datetime import datetime, timedelta

# Tickers to backfill
tickers = get_sp500_tickers()

# Set date range
end_date = datetime.today().date()
start_date = end_date - timedelta(days=7) 

# Backfill sentiment day by day
for ticker in tickers:
    print(f"\nBackfilling {ticker}...")
    for n in range((end_date - start_date).days + 1):
        day = start_date + timedelta(days=n)

        try:
            get_cached_sentiment(ticker, date=day)
            print(f"  {day} done")
        except Exception as e:
            print(f"  {day} failed: {e}")
