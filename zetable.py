import ccxt
import math
import logging
import numpy as np
import pandas as pd
import streamlit as st
from rich.console import Console  # Optional, for logging styling
from rich.table import Table
from rich import box

# Configure logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a set of stablecoins to exclude.
STABLECOINS = {
    'USDT', 'USDC', 'DAI', 'BUSD', 'TUSD', 'PAX', 'GUSD',
    'USDK', 'UST', 'USD', 'SUSD', 'FRAX', 'LUSD', 'MIM', 'USDQ', 'TBTC', 'WBTC',
    'EUL', 'EUR', 'EURT', 'USDS', 'USTS', 'USTC', 'USDR', 'PYUSD', 'EURR',
    'GBP', 'AUD', 'EURQ', 'T', 'USDG', 'WAXL', 'PAXG', 'IDEX', 'FIS', 'CSM',
    'POWR', 'ATLAS', 'XCN', 'BOBA', 'OXY', 'BNC', 'POLIS', 'AIR', 'C98', 'BODEN'
}

# Global dictionaries for volume delta tracking.
last_volume_dict = {}
avg_vol_delta_dict = {}
VOLUME_ALPHA = 0.1  # Smoothing factor for volume delta.

# ---------------- Helper Functions ----------------
def fetch_tickers_usd(exchange):
    """
    Fetch all tickers from the exchange and filter for those ending with '/USD',
    excluding any where the base currency is a stablecoin.
    """
    all_tickers = exchange.fetch_tickers()
    tickers_usd = {}
    for symbol, ticker in all_tickers.items():
        if not symbol.endswith("/USD"):
            continue
        try:
            base, quote = symbol.split("/")
        except Exception:
            continue
        if base.upper() in STABLECOINS:
            continue
        tickers_usd[symbol] = ticker
    return tickers_usd

def calibrate_ranking_beta(tickers):
    """
    Calibrate ranking_beta using differences between consecutive raw scores (percentage changes),
    assuming an exponential distribution. Cap the value at 10.
    """
    scores = [ticker.get("percentage", 0.0) for ticker in tickers]
    if len(scores) < 2:
        return 0.25
    scores_sorted = sorted(scores, reverse=True)
    diffs = [scores_sorted[i] - scores_sorted[i+1] for i in range(len(scores_sorted)-1)]
    positive_diffs = [d for d in diffs if d > 0]
    if not positive_diffs:
        return 0.25
    mean_diff = np.mean(positive_diffs)
    ideal_beta = 1.0 / mean_diff if mean_diff != 0 else 0.25
    return min(ideal_beta, 10)

def update_volume_delta(symbol, current_volume):
    """
    Update the exponential moving average of volume delta for a given ticker.
    """
    last_vol = last_volume_dict.get(symbol, current_volume)
    raw_delta = current_volume - last_vol
    prev_avg = avg_vol_delta_dict.get(symbol, raw_delta)
    avg_delta = VOLUME_ALPHA * raw_delta + (1 - VOLUME_ALPHA) * prev_avg
    avg_vol_delta_dict[symbol] = avg_delta
    last_volume_dict[symbol] = current_volume
    return avg_delta

def process_volume_delta(tickers):
    """
    Update the volume delta for each ticker using its 'quoteVolume'.
    """
    for ticker in tickers:
        symbol = ticker.get("symbol")
        current_volume = float(ticker.get("quoteVolume", 0.0))
        vol_delta = update_volume_delta(symbol, current_volume)
        ticker["vol_delta"] = vol_delta

def compute_scores(tickers, ranking_beta, drive, vol_scale):
    """
    For each ticker, compute:
      - weight = exp(normalized_exponent)
         where normalized_exponent = (ranking_beta * (N - rank) / (ranking_beta*(N-1)))*5.
         This linearly maps the raw exponent to [0, 5].
      - boost = weight * drive.
      - score = (percentage change) * (1 + boost * (1 + (vol_delta / vol_scale))).
    """
    sorted_tickers = sorted(tickers, key=lambda x: x.get("percentage", 0.0), reverse=True)
    N = len(sorted_tickers)
    for idx, ticker in enumerate(sorted_tickers):
        rank = idx + 1
        ticker["rank"] = rank
        raw_exponent = ranking_beta * (N - rank)
        max_exponent = ranking_beta * (N - 1) if N > 1 else 1
        normalized_exponent = (raw_exponent / max_exponent) * 5  # Map to [0,5]
        weight = math.exp(normalized_exponent)
        ticker["weight"] = weight
        boost = weight * drive
        ticker["boost"] = boost
        vol_delta = ticker.get("vol_delta", 0.0)
        vol_factor = vol_delta / vol_scale
        ticker["score"] = ticker.get("percentage", 0.0) * (1 + boost * (1 + vol_factor))
    return sorted_tickers

def calibrate_drive(tickers, ranking_beta, drive_candidates):
    """
    Calibrate drive by choosing the candidate that maximizes the standard deviation of final scores.
    Also cap drive at 5.
    """
    best_drive = drive_candidates[0]
    best_std = -np.inf
    for d in drive_candidates:
        tickers_scored = compute_scores(tickers.copy(), ranking_beta, d, vol_scale=1.0)
        scores = [ticker["score"] for ticker in tickers_scored]
        std = np.std(scores)
        if std > best_std:
            best_std = std
            best_drive = d
    return min(best_drive, 5)

def get_dataframe(tickers):
    """
    Convert the list of tickers (each a dict) into a pandas DataFrame.
    """
    df = pd.DataFrame(tickers)
    # Reorder columns for clarity.
    cols = ["symbol", "last", "percentage", "rank", "weight", "boost", "score"]
    df = df[cols]
    df.columns = ["Symbol", "Price", "24h %", "Rank", "Weight", "Boost", "Score"]
    return df

# ---------------- Streamlit App ----------------
def run_app():
    st.title("USD Ticker Ranking")
    st.write("This app fetches USD tickers from Kraken (excluding stablecoin pairs), calibrates dynamic ranking parameters, and displays the final ranking.")
    
    # Create an exchange instance.
    try:
        exchange = ccxt.kraken()
    except Exception as e:
        st.error(f"Error creating exchange instance: {e}")
        return
    
    try:
        tickers_dict = fetch_tickers_usd(exchange)
    except Exception as e:
        st.error(f"Error fetching tickers: {e}")
        return

    tickers_list = []
    for symbol, ticker in tickers_dict.items():
        ticker["symbol"] = symbol
        ticker["percentage"] = float(ticker.get("percentage") or 0.0)
        ticker["quoteVolume"] = float(ticker.get("quoteVolume") or 0.0)
        tickers_list.append(ticker)
    
    if not tickers_list:
        st.error("No USD tickers found.")
        return
    
    # Update volume delta.
    process_volume_delta(tickers_list)
    
    # Calibrate ranking_beta.
    ideal_ranking_beta = calibrate_ranking_beta(tickers_list)
    st.write(f"Calibrated ranking_beta: {ideal_ranking_beta:.4f}")
    
    # Calibrate drive.
    drive_candidates = [0.5 * i for i in range(1, 21)]
    ideal_drive = calibrate_drive(tickers_list, ideal_ranking_beta, drive_candidates)
    st.write(f"Calibrated drive: {ideal_drive:.4f}")
    
    # Compute dynamic vol_scale.
    volumes = [ticker["quoteVolume"] for ticker in tickers_list if ticker["quoteVolume"] > 0]
    vol_scale = np.median(volumes) if volumes else 1.0
    st.write(f"Dynamically computed vol_scale (median volume): {vol_scale:.2f}")
    
    final_tickers = compute_scores(tickers_list, ideal_ranking_beta, ideal_drive, vol_scale=vol_scale)
    df = get_dataframe(final_tickers)
    
    st.dataframe(df)


    def create_dataframe(tickers, perc_field, perc_label):
        data = []
        for ticker in tickers:
            data.append({
                "Symbol": ticker.get("symbol", "N/A"),
                "Price": f"{ticker.get('last', 0.0):.2f}",
                perc_label: f"{ticker.get(perc_field, 0.0):.2f}",
                "Rank": ticker.get("rank", 0),
                "Weight": f"{ticker.get('weight', 0.0):.2f}",
                "Boost": f"{ticker.get('boost', 0.0):.2f}",
                "Score": f"{ticker.get('score', 0.0):.2f}"
            })
        return pd.DataFrame(data)

# In your Streamlit app, you can display the DataFrame:
    df_24 = create_dataframe(final_tickers_24, perc_field="percentage", perc_label="24h %")
    df_7d = create_dataframe(final_tickers_7d, perc_field="percentage7d", perc_label="7d %")
    df_30 = create_dataframe(final_tickers_30, perc_field="percentage30d", perc_label="30d %")

# Option to display tables side by side using Streamlit columns:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("24h %")
        st.dataframe(df_24)
    with col2:
        st.subheader("7d %")
        st.dataframe(df_7d)
    with col3:
        st.subheader("30d %")
        st.dataframe(df_30)
if __name__ == "__main__":
    import streamlit as st
    import pandas as pd
    run_app()
