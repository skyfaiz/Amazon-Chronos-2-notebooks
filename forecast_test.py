import pandas as pd
import matplotlib.pyplot as plt
from binance.client import Client

# =========================================================
# ⚙️ CONFIGURATION: YOUR CONTROL PANEL
# =========================================================
SYMBOL = "BTCUSDT"

# 1. TIMEFRAME: Change this to alter the candle resolution
# Options: Client.KLINE_INTERVAL_5MINUTE, 15MINUTE, 1HOUR, 4HOUR, 1DAY
TIMEFRAME = Client.KLINE_INTERVAL_1HOUR 

# 2. MORE DATA: How far back should Binance fetch?
# Examples: "1 year ago UTC", "6 months ago UTC", "2 years ago UTC"
LOOKBACK = "2 month ago UTC"

# 3. THE CUTOFF: How many candles are we hiding for the "Ground Truth"?
# If Timeframe is 1HOUR and TEST_CANDLES is 72, you are predicting 3 days into the future.
TEST_CANDLES = 96

# =========================================================
# 1. FETCH & PREPARE DATA
# =========================================================
print(f"Fetching {LOOKBACK} of {TIMEFRAME} data for {SYMBOL}...")
client = Client()
klines = client.get_historical_klines(SYMBOL, TIMEFRAME, LOOKBACK)
df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])

df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df['price_close'] = df['close'].astype(float)

# Calculate EMAs
df['ema_9'] = df['price_close'].ewm(span=9, adjust=False).mean()
df['ema_21'] = df['price_close'].ewm(span=21, adjust=False).mean()
df['ema_50'] = df['price_close'].ewm(span=50, adjust=False).mean()

# Calculate Spreads (Mean Reversion Strategy)
baseline = df['ema_50']
df['Price_Spread'] = df['price_close'] - baseline
df['EMA_9_Spread'] = df['ema_9'] - baseline
df['EMA_21_Spread'] = df['ema_21'] - baseline
df['EMA_50_Raw'] = baseline 

df = df.dropna().reset_index(drop=True)
print(f"Total historical candles fetched: {len(df)}")

# =========================================================
# 2. THE CUTOFF (Train / Test Split)
# =========================================================
# This slices the dataframe based on your TEST_CANDLES setting
train_df = df.iloc[:-TEST_CANDLES].copy()  # Everything BEFORE the cutoff (The Past)
test_df = df.iloc[-TEST_CANDLES:].copy()   # Everything AFTER the cutoff (The Ground Truth)

print(f"Model will train on {len(train_df)} candles.")
print(f"Model will go blind and predict the final {len(test_df)} candles.")

def make_spread_long_format(data):
    series = [
        pd.DataFrame({'timestamp': data['timestamp'], 'id': 'BTC_EMA_50_Raw', 'value': data['EMA_50_Raw']}),
        pd.DataFrame({'timestamp': data['timestamp'], 'id': 'BTC_Price_Spread', 'value': data['Price_Spread']}),
        pd.DataFrame({'timestamp': data['timestamp'], 'id': 'BTC_EMA_9_Spread', 'value': data['EMA_9_Spread']}),
        pd.DataFrame({'timestamp': data['timestamp'], 'id': 'BTC_EMA_21_Spread', 'value': data['EMA_21_Spread']})
    ]
    return pd.concat(series).reset_index(drop=True)

context_long = make_spread_long_format(train_df)

# =========================================================
# 3. FINE-TUNING ON MASSIVE DATA
# =========================================================
print("Formatting data for fine-tuning...")
train_inputs = [{"target": group["value"].values} for _, group in context_long.groupby("id")]

print("Starting Fine-Tuning... (This will take longer with 1 year of data!)")
finetuned_pipeline = pipeline.fit(
    inputs=train_inputs,
    prediction_length=TEST_CANDLES, 
    num_steps=500,      # Increased to 1000 because we have 1 year of data to learn
    learning_rate=3e-5,  # Slightly higher learning rate for a larger dataset
    batch_size=8,       
    logging_steps=100
)

# =========================================================
# 4. PREDICTION & RECONSTRUCTION
# =========================================================
print(f"Predicting the hidden {TEST_CANDLES} candles...")
pred_df = finetuned_pipeline.predict_df(
    df=context_long,
    target="value",
    id_column="id",
    timestamp_column="timestamp",
    prediction_length=TEST_CANDLES,
    quantile_levels=[0.1, 0.5, 0.9],
    predict_batches_jointly=True,
    batch_size=4 
)

# Extract predictions
pred_ema50 = pred_df[pred_df['id'] == 'BTC_EMA_50_Raw'].set_index('timestamp')
pred_price_spread = pred_df[pred_df['id'] == 'BTC_Price_Spread'].set_index('timestamp')

# Reconstruct Price targets
recon_price = pred_ema50['predictions'] + pred_price_spread['predictions']
recon_price_p10 = pred_ema50['predictions'] + pred_price_spread['0.1']
recon_price_p90 = pred_ema50['predictions'] + pred_price_spread['0.9']

# =========================================================
# 5. VISUALIZATION
# =========================================================
plt.figure(figsize=(16, 7))

# Define how much history to display (e.g., 3x the test size so you can see the lead-up)
hist = train_df.tail(TEST_CANDLES * 3)

plt.plot(hist['timestamp'], hist['price_close'], color='black', linewidth=2.5, label='History (Price)')
plt.plot(test_df['timestamp'], test_df['price_close'], color='xkcd:grass green', linewidth=3, label='Actual Future (Ground Truth)')
plt.plot(recon_price.index, recon_price, color='xkcd:violet', linewidth=2.5, linestyle='--', label='Fine-Tuned Forecast')

plt.fill_between(recon_price.index, recon_price_p10, recon_price_p90, color='xkcd:light lavender', alpha=0.5, label='80% Interval')

plt.axvline(x=hist['timestamp'].iloc[-1], color='red', linestyle='--', alpha=0.7, label='The Cutoff (Model went blind here)')
plt.title(f"Chronos-2 Backtest: Predicting {TEST_CANDLES} steps on {TIMEFRAME} timeframe")
plt.xlabel("Time (UTC)")
plt.ylabel("Price (USDT)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
