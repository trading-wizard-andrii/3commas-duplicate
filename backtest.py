import os
import json
import pandas as pd
from flask import Flask, request, render_template, jsonify, send_from_directory
import pandas as pd
import csv
from collections import defaultdict
import numpy as np
import os
import ccxt
import pandas as pd
from flask import Flask, request, render_template, jsonify, Response, send_file
from datetime import datetime
from multiprocessing import Pool, Manager, cpu_count
from ta import momentum, trend, volatility
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
import shutil

from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from multiprocessing import Manager, Lock

app = Flask(__name__)
DATA_DIR = "static"
exchange = ccxt.binance()

ALL_TIMEFRAMES = [
    "1m", "3m", "5m", "15m", "1h", "4h", "1d"
]

@app.route('/get_all_strategies', methods=['GET'])
def get_all_strategies():
    ROOT_DIR = os.path.join("static", "backtest_results")
    all_rows = []

    if os.path.exists(ROOT_DIR):
        subfolders = [f.path for f in os.scandir(ROOT_DIR) if f.is_dir()]
        for folder in subfolders:
            summary_path = os.path.join(folder, "backtest_summary_metrics.csv")
            if os.path.exists(summary_path):
                try:
                    df = pd.read_csv(summary_path)
                    df["folder_path"] = folder
                    df["folder_name"] = os.path.basename(folder)
                    all_rows.append(df)
                except Exception as e:
                    print(f"Error reading {summary_path}: {e}")

    if not all_rows:
        return jsonify({"columns": [], "rows": []})

    df_all = pd.concat(all_rows, ignore_index=True)

    # Gather columns in the order they appear in df_all
    columns = df_all.columns.tolist()
    # Convert rows to records
    rows = df_all.to_dict(orient="records")

    # Return a dict with 'columns' and 'rows'
    return jsonify({
        "columns": columns,    # a list of column names in correct order
        "rows": rows           # the actual data
    })

@app.route('/delete_strategy_folder', methods=['POST'])
def delete_strategy_folder():
    """
    Expects JSON like: { "folder_path": "/full/path/to/folder" }
    We'll remove that folder entirely.
    """
    data = request.json
    folder_path = data.get("folder_path", "")
    if not folder_path:
        return jsonify({"status": "error", "message": "No folder_path provided"})

    try:
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            return jsonify({"status": "success", "message": f"Deleted folder {folder_path}"})
        else:
            return jsonify({"status": "error", "message": f"Folder does not exist: {folder_path}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/download_deals_csv', methods=['GET'])
def download_deals_csv():
    """
    Query param: ?folder=some/path
    We'll find 'all_trades_combined.csv' in that folder and return it as a download.
    """
    folder = request.args.get("folder", "")
    if not folder:
        return jsonify({"status": "error", "message": "No folder param provided"})

    deals_path = os.path.join(folder, "all_trades_combined.csv")
    if not os.path.exists(deals_path):
        return jsonify({"status": "error", "message": "File not found in that folder"})

    # 'send_file' or 'send_from_directory' usage:
    return send_file(deals_path, as_attachment=True, download_name="all_trades_combined.csv")

def fetch_entire_ohlcv(symbol, timeframe, start_ts, end_ts):
    """
    Fetch all OHLCV data for a given symbol/timeframe from start_ts to end_ts
    in a simple loop, returning a single list of [timestamp, open, high, low, close, volume].
    """
    all_data = []
    limit = 1000
    since_ts = start_ts

    while True:
        data = exchange.fetch_ohlcv(symbol, timeframe, since=since_ts, limit=limit)
        if not data:
            break
        all_data.extend(data)
        last_timestamp = data[-1][0]
        if last_timestamp >= end_ts:
            break
        since_ts = last_timestamp + 1  # move forward slightly

    return all_data

def calculate_indicators(df):
    # 1. RSI
    rsi_periods = [7, 14, 21, 28]
    for period in rsi_periods:
        df[f'RSI_{period}'] = momentum.RSIIndicator(close=df['close'], window=period).rsi()

    # 2. Heiken Ashi
    ha_df = df.copy()
    ha_df['ha_close'] = (ha_df['open'] + ha_df['high'] + ha_df['low'] + ha_df['close']) / 4
    ha_df['ha_open']  = (ha_df['open'].shift(1) + ha_df['close'].shift(1)) / 2
    ha_df['ha_high']  = ha_df[['high', 'ha_close', 'ha_open']].max(axis=1)
    ha_df['ha_low']   = ha_df[['low', 'ha_close', 'ha_open']].min(axis=1)
    df['HA_Open']  = ha_df['ha_open']
    df['HA_High']  = ha_df['ha_high']
    df['HA_Low']   = ha_df['ha_low']
    df['HA_Close'] = ha_df['ha_close']

    # 3. SMA & EMA
    ma_periods = [5, 10, 14, 20, 25, 30, 50, 75, 100, 150, 200, 250]
    for period in ma_periods:
        df[f'SMA_{period}'] = trend.SMAIndicator(close=df['close'], window=period).sma_indicator()
        df[f'EMA_{period}'] = trend.EMAIndicator(close=df['close'], window=period).ema_indicator()

    # 4. Bollinger Bands %B
    bb_windows = [14, 20, 50, 10, 100]     # 5 popular windows
    bb_devs    = [1, 1.5, 2, 2.5, 3]      # 5 popular devs

    for window in bb_windows:
        for dev in bb_devs:
            bb = volatility.BollingerBands(close=df['close'], window=window, window_dev=dev)
            col_name = f'BB_%B_{window}_{dev}'
            df[col_name] = bb.bollinger_pband()

    # 5. MACD
    macd_std = trend.MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD_12_26_9']           = macd_std.macd()
    df['MACD_Signal_12_26_9']    = macd_std.macd_signal()
    df['MACD_Histogram_12_26_9'] = macd_std.macd_diff()

    # Add 5 more popular combos: (fast, slow, signal)
    macd_configs = [
        (6, 20, 9),
        (9, 30, 9),
        (15, 35, 9),
        (18, 40, 9),
        (10, 26, 9),   # example extra
    ]
    for fast, slow, signal in macd_configs:
        macd_temp = trend.MACD(close=df['close'], window_slow=slow, window_fast=fast, window_sign=signal)
        prefix = f'MACD_{fast}_{slow}_{signal}'
        df[prefix]            = macd_temp.macd()
        df[f'{prefix}_Signal'] = macd_temp.macd_signal()
        df[f'{prefix}_Hist']   = macd_temp.macd_diff()

    # Example: We'll generate multiple combos (k_len, k_smooth, d_smooth).
    stoch_combos = [
        (14, 3, 3),
        (14, 3, 5),
        (20, 5, 5),
        (21, 7, 7),
        (28, 9, 9),
    ]

    for (k_len, k_smooth, d_smooth) in stoch_combos:
        # 1) Create the StochasticOscillator with K length = k_len, K smoothing = k_smooth.
        stoch_obj = momentum.StochasticOscillator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=k_len,          # "K length"
            smooth_window=k_smooth # first-level smoothing
        )

        # 2) K line = stoch() (already smoothed by k_smooth)
        k_line = stoch_obj.stoch()

        # 3) Base D line = stoch_signal() (one smoothing step)
        d_line_base = stoch_obj.stoch_signal()

        # 4) Apply an additional rolling mean if d_smooth > 1
        #    to get a "double-smoothed" D line.
        #    We can do that with pandas rolling or a separate SMAIndicator.
        if d_smooth > 1:
            d_line = d_line_base.rolling(d_smooth).mean()  # Simple moving average
        else:
            d_line = d_line_base  # no extra smoothing

        # 5) Store results in columns
        #    We'll name them with the pattern "Stochastic_K_{k_len}_{k_smooth}"
        #    and "Stochastic_D_{k_len}_{k_smooth}_{d_smooth}".
        k_col_name = f'Stochastic_K_{k_len}_{k_smooth}'
        d_col_name = f'Stochastic_D_{k_len}_{k_smooth}_{d_smooth}'

        df[k_col_name] = k_line
        df[d_col_name] = d_line


    # 7. Parabolic SAR
    psar_std = trend.PSARIndicator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        step=0.02,
        max_step=0.2
    )
    df['PSAR_AF_0.02_Max_0.2'] = psar_std.psar()

    # -- 1st extra combo --
    psar_2 = trend.PSARIndicator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        step=0.03,
        max_step=0.2
    )
    df['PSAR_AF_0.03_Max_0.2'] = psar_2.psar()

    # -- 2nd extra combo --
    psar_3 = trend.PSARIndicator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        step=0.04,
        max_step=0.3
    )
    df['PSAR_AF_0.04_Max_0.3'] = psar_3.psar()

    # -- 3rd extra combo --
    psar_4 = trend.PSARIndicator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        step=0.05,
        max_step=0.4
    )
    df['PSAR_AF_0.05_Max_0.4'] = psar_4.psar()

    # -- 4th extra combo --
    psar_5 = trend.PSARIndicator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        step=0.06,
        max_step=0.5
    )
    df['PSAR_AF_0.06_Max_0.5'] = psar_5.psar()

    return df

def map_rating(value):
    if value > 0.5:
        return "Strong Buy"
    elif value > 0.1:
        return "Buy"
    elif value >= -0.1:
        return "Neutral"
    elif value >= -0.5:
        return "Sell"
    else:
        return "Strong Sell"

def calculate_technical_ratings(df):
    indicators = {}

    if 'SMA_20' in df.columns:
        indicators['sma_signal'] = (df['close'] > df['SMA_20']).astype(int) - (df['close'] < df['SMA_20']).astype(int)
    else:
        indicators['sma_signal'] = 0

    if 'EMA_20' in df.columns:
        indicators['ema_signal'] = (df['close'] > df['EMA_20']).astype(int) - (df['close'] < df['EMA_20']).astype(int)
    else:
        indicators['ema_signal'] = 0

    # RSI
    if 'RSI_14' in df.columns:
        indicators['rsi_signal'] = df['RSI_14'].apply(
            lambda x: 1 if x < 30 else (-1 if x > 70 else 0)
        )
    else:
        indicators['rsi_signal'] = 0

    # Stochastic
    if 'Stochastic_K_14_3' in df.columns and 'Stochastic_D_14_3_3' in df.columns:
        indicators['stochastic_signal'] = df.apply(
            lambda row: 1 if (row['Stochastic_K_14_3'] < 20 and row['Stochastic_K_14_3'] > row['Stochastic_D_14_3_3'])
                        else (-1 if (row['Stochastic_K_14_3'] > 80 and row['Stochastic_K_14_3'] < row['Stochastic_D_14_3_3'])
                              else 0),
            axis=1
        )
    else:
        indicators['stochastic_signal'] = 0

    # MACD
    if 'MACD_12_26_9' in df.columns and 'MACD_Signal_12_26_9' in df.columns:
        indicators['macd_signal'] = df.apply(
            lambda row: 1 if row['MACD_12_26_9'] > row['MACD_Signal_12_26_9']
            else (-1 if row['MACD_12_26_9'] < row['MACD_Signal_12_26_9'] else 0),
            axis=1
        )
    else:
        indicators['macd_signal'] = 0

    # Parabolic SAR
    parabolic_cols = [c for c in df.columns if 'PSAR_AF' in c]
    if parabolic_cols:
        col = parabolic_cols[0]
        indicators['parabolic_sar_signal'] = np.where(
            df['close'] > df[col], 1,
            np.where(df['close'] < df[col], -1, 0)
        )
    else:
        indicators['parabolic_sar_signal'] = 0

    # Combine signals
    ma_signals = ['sma_signal', 'ema_signal']
    osc_signals = ['rsi_signal', 'stochastic_signal', 'macd_signal', 'parabolic_sar_signal']

    df['ma_rating']        = pd.DataFrame(indicators)[ma_signals].mean(axis=1)
    df['oscillator_rating'] = pd.DataFrame(indicators)[osc_signals].mean(axis=1)
    df['tv_tech_rate']     = df[['ma_rating', 'oscillator_rating']].mean(axis=1)
    df['tv_tech_label']    = df['tv_tech_rate'].apply(map_rating)

    # Store the raw signals in df
    for k, v in indicators.items():
        df[k] = v

    return df


@app.route('/load_data')
def index2():
    return render_template('index.html')

def fetch_timeframe(symbol, timeframe, start_ts, end_ts):
    """
    Fetch data, calculate indicators, and return the result DataFrame.
    """
    ohlcv = fetch_entire_ohlcv(symbol, timeframe, start_ts, end_ts)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.sort_values('timestamp', inplace=True)
    df.drop_duplicates(subset=['timestamp'], inplace=True)

    # Calculate indicators
    df = calculate_indicators(df)
    df = calculate_technical_ratings(df)
    df = df.drop(columns=[f"{col}_{suffix}" for col in [
        'sma_signal', 'ema_signal', 'rsi_signal', 'stochastic_signal',
        'macd_signal', 'parabolic_sar_signal', 'ma_rating',
        'oscillator_rating', 'tv_tech_rate'
    ] for suffix in ["_1m", "_3m", "_5m", "_15m", "_1h", "_4h", "_1d"]], errors='ignore')


    return df

@app.route('/fetch_data', methods=['GET'])
def fetch_data():
    """
    Fetch OHLCV data in parallel, process indicators, and merge into a single Parquet file.
    """
    symbol = request.args.get('symbol', 'BTC/USDT')
    start_date = '2024-08-01T00:00:00'
    end_date = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
    start_ts = exchange.parse8601(start_date)
    end_ts = exchange.parse8601(end_date)

    def generate_progress():
        try:
            yield f"data: Starting parallel fetch for {symbol}\n\n"
            timeframe_data = {}

            # Parallel processing with ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
                futures = {
                    executor.submit(fetch_timeframe, symbol, tf, start_ts, end_ts): tf
                    for tf in ALL_TIMEFRAMES
                }
                for future in as_completed(futures):
                    tf = futures[future]
                    try:
                        timeframe_data[tf] = future.result()
                        yield f"data: Finished {tf}\n\n"
                    except Exception as e:
                        yield f"data: Error processing {tf}: {e}\n\n"

            # Merge into a single DataFrame
            df_1m = timeframe_data['1m']
            df_1m.set_index('timestamp', inplace=True)
            for tf, df_tf in timeframe_data.items():
                if tf == '1m':
                    continue
                df_tf.set_index('timestamp', inplace=True)
                df_tf = df_tf.resample('1T').ffill()
                df_tf = df_tf.add_suffix(f"_{tf}")
                df_1m = df_1m.join(df_tf, how='outer')

            df_1m.sort_index(inplace=True)
            df_1m.reset_index(inplace=True)

            # Save as Parquet
            file_path = os.path.join("static", f"{symbol.replace('/', '_')}_all_tf_merged.parquet")
            df_1m.to_parquet(file_path, index=False, compression='snappy')
            yield f"data: Saved to {file_path}\n\n"

        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"

    return Response(generate_progress(), content_type='text/event-stream')

@app.route('/data_summary', methods=['GET'])
def data_summary():
    """
    Lists Parquet files in `static` folder, returning a brief summary
    (file name, row count, start date, end date).
    """
    data_summary = []
    if os.path.exists('static'):
        for file in os.listdir('static'):
            if file.endswith('.parquet'):
                path = os.path.join('static', file)
                df = pd.read_parquet(path)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    rows = len(df)
                    start_date = df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')
                    end_date   = df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
                else:
                    rows, start_date, end_date = 0, 'N/A', 'N/A'
                data_summary.append({
                    'file': file,
                    'rows': rows,
                    'start_date': start_date,
                    'end_date': end_date
                })
    return jsonify(data_summary)

@app.route('/delete_file', methods=['DELETE'])
def delete_file():
    filename = request.args.get('filename')
    if not filename:
        return jsonify({'status': 'error', 'message': 'No filename provided'}), 400

    file_path = os.path.join('static', filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            return jsonify({'status': 'ok'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500
    else:
        return jsonify({'status': 'error', 'message': 'File not found'}), 404


# ─────────────────────────────────────────────────────────────
# NEW: Helper to see if a timestamp matches the "close" of a given timeframe
# ─────────────────────────────────────────────────────────────
def is_bar_close(ts, timeframe):
    """
    Returns True if 'ts' (a pandas Timestamp) is the bar close for the specified timeframe.
    We handle these exact timeframes:
      1m, 3m, 5m, 15m, 1h, 4h, 1d

    Note: This assumes your data rows occur at exact minute boundaries
          (e.g. 00:00, 00:01, 00:02, etc.), with second=0 by design.
    """

    if timeframe == "1m":
        # 1m bar closes each minute at second=0, microsecond=0.
        # If your data is exactly at minute boundaries, we only check second/microsecond.
        return (ts.second == 0 and ts.microsecond == 0)

    elif timeframe == "3m":
        # 3-minute bar: close if minute % 3 == 0, second=0, microsecond=0
        if ts.second != 0 or ts.microsecond != 0:
            return False
        return (ts.minute % 3) == 0

    elif timeframe == "5m":
        # 5-minute bar: close if minute % 5 == 0, second=0, microsecond=0
        if ts.second != 0 or ts.microsecond != 0:
            return False
        return (ts.minute % 5) == 0

    elif timeframe == "15m":
        # 15-minute bar: minute % 15 == 0, second=0, microsecond=0
        if ts.second != 0 or ts.microsecond != 0:
            return False
        return (ts.minute % 15) == 0

    elif timeframe == "1h":
        # 1-hour bar: bar closes at minute=0, second=0, microsecond=0
        # e.g. 01:00, 02:00, 03:00, ...
        if ts.minute != 0 or ts.second != 0 or ts.microsecond != 0:
            return False
        return True  # every hour on the hour

    elif timeframe == "4h":
        # 4-hour bar: bar closes when hour % 4 == 0, minute=0, second=0, microsecond=0
        if (ts.minute != 0 or ts.second != 0 or ts.microsecond != 0):
            return False
        return (ts.hour % 4) == 0

    elif timeframe == "1d":
        # 1-day bar: typically closes at 00:00:00
        # i.e. hour=0, minute=0, second=0, microsecond=0
        # Adjust if your daily candle closes at e.g. 23:59:59
        if (ts.hour == 0 and ts.minute == 0 and ts.second == 0 and ts.microsecond == 0):
            return True
        return False

    else:
        # If the timeframe is something else or unexpected,
        # we'll default to True or False. Usually best to do a fallback = True for "1m" logic.
        # But to avoid confusion, let's just say we do not recognize it -> False
        return False

# ─────────────────────────────────────────────────────────────
# 1) HELPER: GET ALL USER PAYLOAD FIELDS
# ─────────────────────────────────────────────────────────────
def get_user_payload(data):
    """
    Extracts EVERY field from the user payload.
    Returns them in a dict, so we can easily pass them around.
    """
    return {
        "strategy_name": data.get('strategy_name', ''),
        "pairs": data.get('pairs', []),
        "max_active_deals": data.get('max_active_deals', 0),
        "trading_fee": data.get('trading_fee', 0.0),
        "base_order_size": data.get('base_order_size', 0.0),
        "start_date": data.get("start_date", ""),  # from HTML input
        "end_date": data.get("end_date", ""),      # from HTML input

        "entry_conditions": data.get('entry_conditions', []),
        "safety_order_toggle": data.get('safety_order_toggle', False),
        "safety_order_size": data.get('safety_order_size', 0.0),
        "price_deviation": data.get('price_deviation', 0.0),
        "max_safety_orders_count": data.get('max_safety_orders_count', 0),
        "safety_order_volume_scale": data.get('safety_order_volume_scale', 0.0),
        "safety_order_step_scale": data.get('safety_order_step_scale', 0.0),
        "safety_conditions": data.get('safety_conditions', []),

        "price_change_active": data.get('price_change_active', False),
        "conditions_active": data.get('conditions_active', False),
        "take_profit_type": data.get('take_profit_type', ''),
        "target_profit": data.get('target_profit', 0.0),

        "trailing_toggle": data.get('trailing_toggle', False),
        "trailing_deviation": data.get('trailing_deviation', 0.0),
        "exit_conditions": data.get('exit_conditions', []),
        "minprof_toggle": data.get('minprof_toggle', False),
        "minimal_profit": data.get('minimal_profit', 0),

        "reinvest_profit": data.get('reinvest_profit', 0.0),
        "stop_loss_toggle": data.get('stop_loss_toggle', False),
        "stop_loss_value": data.get('stop_loss_value', 0.0),
        "stop_loss_timeout": data.get('stop_loss_timeout', 0.0),
        "stop_loss_trailing": data.get('stop_loss_trailing', False),
        "risk_reduction": data.get('risk_reduction', 0.0),

        "min_daily_volume": data.get('min_daily_volume', 0.0),
        "cooldown_between_deals": data.get('cooldown_between_deals', 0)
    }

# ─────────────────────────────────────────────────────────────
# 2) HELPER: DETERMINE WHICH COLUMNS ARE NEEDED
# ─────────────────────────────────────────────────────────────
def gather_required_columns(entry_conditions, safety_conditions, exit_conditions):
    """
    Parse all user conditions to figure out which
    indicator columns we actually need from the Parquet.
    Also includes standard columns: "timestamp", "open", "high", "low", "close", "volume".
    """
    required = {"timestamp", "open", "high", "low", "close", "volume"}

    def parse_one(cond):
        indicator = cond.get("indicator", "")
        subs = cond.get("subfields", {})
        tf = subs.get("Timeframe", "1m")

        if indicator == "RSI":
            length = subs.get("RSI Length", 14)
            col = f"RSI_{length}"
            if tf != "1m":
                col += f"_{tf}"
            return [col]

        elif indicator == "MA":
            ma_type = subs.get("MA Type", "SMA")
            fast_ma = subs.get("Fast MA", 14)
            slow_ma = subs.get("Slow MA", 28)
            col_fast = f"{ma_type}_{fast_ma}"
            col_slow = f"{ma_type}_{slow_ma}"
            if tf != "1m":
                col_fast += f"_{tf}"
                col_slow += f"_{tf}"
            return [col_fast, col_slow]

        elif indicator == "BollingerBands":
            period = subs.get("BB% Period", 20)
            dev = subs.get("Deviation", 2)
            col = f"BB_%B_{period}_{dev}"
            if tf != "1m":
                col += f"_{tf}"
            return [col]

        elif indicator == "MACD":
            preset = subs.get("MACD Preset", "12,26,9")  # e.g. "12,26,9"
            fast_str, slow_str, sig_str = preset.split(',')

            # We know the data is stored as e.g. "MACD_12_26_9" and "MACD_Signal_12_26_9"
            main_col   = f"MACD_{fast_str}_{slow_str}_{sig_str}"
            signal_col = f"MACD_Signal_{fast_str}_{slow_str}_{sig_str}"

            # If the user picks a Timeframe != "1m", you might add suffix
            tf = subs.get("Timeframe", "1m")
            if tf != "1m":
                main_col   += f"_{tf}"
                signal_col += f"_{tf}"

            return [main_col, signal_col]

        elif indicator == "Stochastic":
            stoch_preset = subs.get("Stochastic Preset", "14,3,3")
            k_str, ksmooth_str, dsmooth_str = stoch_preset.split(',')
            k_col = f"Stochastic_K_{k_str}_{ksmooth_str}"
            d_col = f"Stochastic_D_{k_str}_{ksmooth_str}_{dsmooth_str}"
            if tf != "1m":
                k_col += f"_{tf}"
                d_col += f"_{tf}"
            return [k_col, d_col]

        elif indicator == "ParabolicSAR":
            psar_str = subs.get("PSAR Preset", "0.02,0.2")
            step_str, max_str = psar_str.split(',')
            col = f"PSAR_AF_{step_str}_Max_{max_str}"
            if tf != "1m":
                col += f"_{tf}"
            return [col]

        elif indicator == "TradingView":
            col = "tv_tech_label"
            if tf != "1m":
                col += f"_{tf}"
            return [col]

        elif indicator == "HeikenAshi":
            col = "HA_Close"
            if tf != "1m":
                col += f"_{tf}"
            return [col]

        else:
            return []

    for c in entry_conditions:
        for col in parse_one(c):
            required.add(col)
    for c in safety_conditions:
        for col in parse_one(c):
            required.add(col)
    for c in exit_conditions:
        for col in parse_one(c):
            required.add(col)

    return list(required)

# ─────────────────────────────────────────────────────────────
# 3) HELPER: LOAD PARQUETS IN PARALLEL (ONLY NEEDED COLUMNS)
# ─────────────────────────────────────────────────────────────
def load_parquets_in_parallel(pairs, required_cols):
    """
    Loads each pair’s Parquet in parallel, using only 'required_cols'.
    Returns a dict: { "SOL/USDT": df, "BTC/USDT": df, ... }
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = {}

    def load_one(pair):
        file_path = f'static/{pair.replace("/", "_")}_all_tf_merged.parquet'
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Parquet file not found for {pair}: {file_path}")

        df = pd.read_parquet(file_path, columns=required_cols)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.sort_values('timestamp').reset_index(drop=True)
        df['symbol'] = pair
        return pair, df

    with ThreadPoolExecutor(max_workers=4) as executor:
        fut_map = {executor.submit(load_one, p): p for p in pairs}
        for fut in as_completed(fut_map):
            pair = fut_map[fut]
            try:
                p, df = fut.result()
                results[p] = df
            except Exception as e:
                print(f"Error loading {pair} in parallel: {e}")
                raise

    return results

# ─────────────────────────────────────────────────────────────
# 4) HELPER: RUN BUY/SELL FOR A SINGLE SYMBOL (PROCESS-BASED)
# ─────────────────────────────────────────────────────────────
def run_symbol_backtest(
    symbol_df,
    symbol,
    manager_dict,
    lock,
    payload,
    check_all_user_conditions,
    required_cols=None
):

    import pandas as pd

    # --- Extract relevant user settings ----------------------------------
    base_order_size       = payload.get("base_order_size", 0.0)  # in dollars, but not used for $ logic here
    entry_conditions      = payload.get("entry_conditions", [])
    exit_conditions       = payload.get("exit_conditions", [])
    safety_order_toggle   = payload.get("safety_order_toggle", False)
    safety_order_size     = payload.get("safety_order_size", 0.0)   # in dollars, not used for $ logic
    price_deviation       = payload.get("price_deviation", 1.0)     # e.g. 1 => 1%
    max_safety_orders     = payload.get("max_safety_orders_count", 0)
    safety_step_scale     = payload.get("safety_order_step_scale", 1.0)
    safety_vol_scale      = payload.get("safety_order_volume_scale", 1.0)
    safety_conditions     = payload.get("safety_conditions", [])

    stop_loss_toggle      = payload.get("stop_loss_toggle", False)
    stop_loss_value       = payload.get("stop_loss_value", 0.0)  # e.g. 5 => 5%

    # Price-change partial exit logic
    price_change_active   = payload.get("price_change_active", False)
    target_profit = payload.get("target_profit", 0.0)
    target_volume = payload.get("target_volume", 0.0)
    take_profit_type      = payload.get("take_profit_type", "percentage-total")  # not used for real $ logic

    conditions_active     = payload.get("conditions_active", False)

    minprof_toggle        = payload.get("minprof_toggle", False)
    minimal_profit        = payload.get("minimal_profit", 0)/100

    # --- Date filtering --------------------------------------------------
    start_date_str = payload.get("start_date", "")
    end_date_str   = payload.get("end_date", "")
    start_date = pd.to_datetime(start_date_str, errors="coerce") if start_date_str else None
    end_date   = pd.to_datetime(end_date_str, errors="coerce")   if end_date_str else None

    if start_date is not None and not pd.isnull(start_date):
        symbol_df = symbol_df[symbol_df["timestamp"] >= start_date]
    if end_date is not None and not pd.isnull(end_date):
        symbol_df = symbol_df[symbol_df["timestamp"] <= end_date]

    # Convert to records for iteration
    records = symbol_df.to_dict("records")

    # --- We'll track an "open trade" with placeholder quantity only ------
    open_trade    = None
    trade_details = []
    trade_ID = 0

    def record_trade(action, row, trade_price, quantity,amount,total_amount,profit_percent,move_from_entry, comment=""):
        """
        Creates a minimal dict for a trade record with:
          - timestamp, symbol, action, price, quantity, trade_comment
          - plus optional columns if in required_cols
        """
        trade_dict = {
            "timestamp":    row["timestamp"],
            "symbol":       symbol,
            "action":       action,
            "price":        trade_price,
            "quantity":     quantity,
            "amount":       amount,
            "total_amount": total_amount,
            "move_from_entry": move_from_entry,
            "profit_percent": profit_percent,
            "trade_comment": comment,
            "trade_id": f"{trade_ID}-{symbol}",
        }
        # If user wants to keep indicator columns in the CSV
        if required_cols:
            for c in required_cols:
                if c not in trade_dict:
                    trade_dict[c] = row.get(c, None)

        trade_details.append(trade_dict)

    # --- Main event loop over bars ---------------------------------------
    for i in range(len(records)):
        row = records[i]
        prev_row = records[i-1] if i>0 else None
        close_px = row["close"]
        move_from_entry = 0

        # 1) If we do NOT have an open trade, see if we open one
        if not open_trade:
            # If entry conditions pass => open a base trade
            if check_all_user_conditions(row, entry_conditions, prev_row):
                # quantity is a placeholder => base_order_size / close_px
                # though no $ logic, we do keep a quantity for partial-exit math
                qty = (base_order_size / close_px) if close_px>1e-12 else 0.0
                amount = close_px * qty
                total_amount = amount
                profit_percent = ""
                trade_ID += 1

                open_trade = {
                    "quantity":    qty,
                    "initial_quantity": qty,
                    "placed_so_count": 0,          # how many safety orders used
                    "last_so_price":  close_px,
                    "last_so_size":   safety_order_size,  # next safety order in $ (placeholder)
                    "so_dev_factor":  price_deviation,
                    "partial_tp_track": [],
                    "entry_price":  close_px,  # for partial-exit reference or stop-loss
                    "total_amount": total_amount,
                    "move_from_entry": move_from_entry,
                    "profit_percent": profit_percent,
                    "next_so_price": None
                }
                move_from_entry = (close_px - open_trade["entry_price"])/open_trade["entry_price"]

                # If user wants safety orders => define next so threshold
                if safety_order_toggle and max_safety_orders>0:
                    dev_frac = price_deviation / 100.0
                    open_trade["next_so_price"] = close_px*(1.0 - dev_frac)

                # If stop-loss => define threshold
                if stop_loss_toggle and stop_loss_value>0:
                    sl_frac = stop_loss_value/100.0
                    # e.g. if price is 100, stop_loss=5 => threshold=95
                    open_trade["stop_loss_threshold"] = close_px*(1.0 - sl_frac)
                else:
                    open_trade["stop_loss_threshold"] = None

                # If stop-loss => define threshold
                if target_profit > 0:
                    tp_frac = target_profit/100.0
                    open_trade["take_profit_threshold"] = close_px*(1.0 + tp_frac)
                else:
                    open_trade["take_profit_threshold"] = None

                # record the "BUY"
                record_trade(
                    action="BUY",
                    row=row,
                    trade_price=close_px,
                    quantity=qty,
                    amount = amount,
                    total_amount = open_trade["total_amount"],
                    profit_percent = "",
                    move_from_entry = move_from_entry,
                    comment="Condition-based Entry"
                )
            continue  # skip to next bar if still no open trade

        # 2) If we DO have an open trade, handle stop-loss / partial-exits / condition-exit / safeties

        # 2a) Stop-Loss first
        if stop_loss_toggle and open_trade.get("stop_loss_threshold") is not None:
            if close_px <= open_trade["stop_loss_threshold"]:
                qty2sell = open_trade["quantity"]  # full close
                amount = open_trade["stop_loss_threshold"] * qty2sell
                profit_percent = (amount - open_trade["total_amount"])/open_trade["total_amount"]
                move_from_entry = (open_trade["stop_loss_threshold"] - open_trade["entry_price"])/open_trade["entry_price"]
                record_trade(
                    action="Stop Loss EXIT",
                    row=row,
                    trade_price=open_trade["stop_loss_threshold"],
                    quantity=qty2sell,
                    amount = amount,
                    total_amount = open_trade["total_amount"],
                    profit_percent = profit_percent,
                    move_from_entry = move_from_entry,
                    comment=f"Stop loss triggered at {stop_loss_value}%"
                )
                open_trade = None
                continue

        # 2b) Condition-based exit if conditions_active => exit_conditions
        #     => full close
        has_exited = False
        if conditions_active:
            if check_all_user_conditions(row, exit_conditions, prev_row):
                qty2sell = open_trade["quantity"]
                amount = close_px * qty2sell
                profit_percent = (amount - open_trade["total_amount"])/open_trade["total_amount"]
                move_from_entry = (close_px - open_trade["entry_price"])/open_trade["entry_price"]

                # IF user wants minimal profit => check that too
                if minprof_toggle:
                    # Only exit if we meet or exceed minimal_profit
                    if profit_percent >= minimal_profit:
                        record_trade(
                            action="SELL",
                            row=row,
                            trade_price=close_px,
                            quantity=qty2sell,
                            amount=amount,
                            total_amount=open_trade["total_amount"],
                            profit_percent=profit_percent,
                            move_from_entry=move_from_entry,
                            comment="Exit triggered by conditions + min profit"
                        )
                        open_trade = None
                        has_exited = True
                else:
                    # minprof_toggle == False => exit on conditions alone
                    record_trade(
                        action="SELL",
                        row=row,
                        trade_price=close_px,
                        quantity=qty2sell,
                        amount=amount,
                        total_amount=open_trade["total_amount"],
                        profit_percent=profit_percent,
                        move_from_entry=move_from_entry,
                        comment="Exit triggered by conditions"
                    )
                    open_trade = None
                    has_exited = True

        if has_exited:
            continue

        # 2c) Price-change partial exit => check price target

        if price_change_active and open_trade.get("take_profit_threshold") is not None:
            if take_profit_type == "percentage-base":
                if close_px >= open_trade["take_profit_threshold"]:
                    qty2sell = open_trade["quantity"]  # full close
                    amount = open_trade["take_profit_threshold"] * qty2sell
                    profit_percent = (amount - open_trade["total_amount"])/open_trade["total_amount"]
                    move_from_entry = (open_trade["take_profit_threshold"] - open_trade["entry_price"])/open_trade["entry_price"]
                    record_trade(
                        action="Take Profit EXIT",
                        row=row,
                        trade_price=open_trade["take_profit_threshold"],
                        quantity=qty2sell,
                        amount=amount,
                        total_amount = open_trade["total_amount"],
                        profit_percent = profit_percent,
                        move_from_entry = move_from_entry,
                        comment=f"Take profit triggered at {target_profit}%"
                    )
                    open_trade = None

            elif take_profit_type == "percentage-total":
                if close_px >= open_trade["take_profit_threshold"]:
                    qty2sell = open_trade["quantity"]  # full close
                    amount = open_trade["take_profit_threshold"] * qty2sell
                    profit_percent = (amount - open_trade["total_amount"])/open_trade["total_amount"]
                    move_from_entry = (open_trade["take_profit_threshold"] - open_trade["entry_price"])/open_trade["entry_price"]
                    record_trade(
                        action="Take Profit EXIT",
                        row=row,
                        trade_price=open_trade["take_profit_threshold"],
                        quantity=qty2sell,
                        amount=amount,
                        total_amount = open_trade["total_amount"],
                        profit_percent = profit_percent,
                        move_from_entry = move_from_entry,
                        comment=f"Take profit triggered at {target_profit}%"
                    )
                    open_trade = None

        # 2d) If still open => check safety orders
        if open_trade:
            if safety_order_toggle and open_trade["placed_so_count"]< max_safety_orders:
                while True:
                    next_so_price = open_trade.get("next_so_price", None)
                    if next_so_price is None:
                        break
                    if open_trade["placed_so_count"]>= max_safety_orders:
                        break

                    if close_px< next_so_price:
                        # check safety conditions
                        if (not safety_conditions) or check_all_user_conditions(row, safety_conditions, prev_row):
                            so_size  = open_trade["last_so_size"]
                            so_qty   = (so_size / next_so_price) if close_px>1e-12 else 0.0
                            open_trade["placed_so_count"] += 1
                            open_trade["quantity"]       += so_qty
                            amount = next_so_price * so_qty
                            open_trade["total_amount"]       += amount
                            move_from_entry = (next_so_price - open_trade["entry_price"])/open_trade["entry_price"]
                            if take_profit_type == "percentage-total":
                                open_trade["take_profit_threshold"] = (open_trade["total_amount"] / open_trade["quantity"]) *(1+target_profit/100)


                            so_num = open_trade["placed_so_count"]
                            record_trade(
                                action=f"Safety Order #{so_num}",
                                row=row,
                                trade_price=next_so_price,
                                quantity=so_qty,
                                amount=amount,
                                total_amount = open_trade["total_amount"],
                                profit_percent = "",
                                move_from_entry = move_from_entry,
                                comment=f"Added safety order #{so_num}"
                            )

                            open_trade["last_so_price"] = close_px
                            open_trade["so_dev_factor"] *= safety_step_scale
                            dev_frac = (price_deviation* open_trade["so_dev_factor"])/100.0
                            open_trade["next_so_price"]  = open_trade["next_so_price"]*(1.0 - dev_frac)
                            open_trade["last_so_size"]  *= safety_vol_scale

                        else:
                            break
                    else:
                        break
    return trade_details

@app.route('/save_period', methods=['POST'])
def save_period():
    try:
        data = request.json
        period_name = data.get('period_name', '').strip()
        start_date  = data.get('start_date', '').strip()
        end_date    = data.get('end_date', '').strip()

        if not period_name or not start_date or not end_date:
            return jsonify({"status": "error", "message": "Missing period_name or dates."})

        # Append to CSV
        csv_file = 'saved_periods.csv'
        header   = ['period_name','start_date','end_date']

        # Check if file exists to see if we need a header
        import os
        file_exists = os.path.exists(csv_file)

        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'period_name': period_name,
                'start_date':  start_date,
                'end_date':    end_date
            })

        return jsonify({"status": "success", "message": "Period saved."})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/get_periods', methods=['GET'])
def get_periods():
    try:
        csv_file = 'saved_periods.csv'
        results = []
        if not os.path.exists(csv_file):
            # No file => return empty
            return jsonify(results)

        with open(csv_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # row is like {"period_name":..., "start_date":..., "end_date":...}
                results.append({
                    "period_name": row.get("period_name",""),
                    "start_date":  row.get("start_date",""),
                    "end_date":    row.get("end_date","")
                })

        return jsonify(results)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/delete_period', methods=['POST'])
def delete_period():
    try:
        data = request.json
        period_name = data.get('period_name', '').strip()
        if not period_name:
            return jsonify({"status":"error", "message":"No period_name provided."})

        csv_file = 'saved_periods.csv'
        import os
        if not os.path.exists(csv_file):
            return jsonify({"status":"error","message":"No saved_periods.csv found."})

        # 1) Read all rows
        import csv
        with open(csv_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # 2) Filter out the one we're deleting
        updated = [r for r in rows if r.get('period_name','').strip() != period_name]

        # If length is the same => means not found
        if len(updated) == len(rows):
            return jsonify({"status":"error","message":f'Period "{period_name}" not found.'})

        # 3) Write back the updated list
        header = ['period_name','start_date','end_date']
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for row in updated:
                writer.writerow(row)

        return jsonify({"status":"success","message":f'Period "{period_name}" deleted.'})

    except Exception as e:
        return jsonify({"status":"error","message":str(e)})

# ─────────────────────────────────────────────────────────────
# 5) CHECK CONDITIONS LOGIC, with new bar-close logic
# ─────────────────────────────────────────────────────────────
def check_all_user_conditions(row, conditions, prev_row=None):
    """
    Return True if *all* conditions in 'conditions' are satisfied for this 'row'.
    Now also requires that row['timestamp'] is the bar close for each condition's timeframe.
    If that condition's timeframe does not align with row['timestamp'], we immediately fail.
    """

    if not conditions:
        return True

    for cond in conditions:
        indicator = cond.get("indicator", "")
        subs = cond.get("subfields", {})

        # e.g. '1m', '5m', '4h', '1d'
        tf = subs.get("Timeframe", "1m")

        # 1) Check bar-close alignment for the chosen timeframe
        if not is_bar_close(row['timestamp'], tf):
            return False

        operator = subs.get("Condition", "")
        value = subs.get("Signal Value", None)

        # Now pick out the indicator columns from row (and possibly prev_row)
        if indicator == "RSI":
            length = subs.get("RSI Length", 14)
            col = f"RSI_{length}"
            if tf != "1m":
                col += f"_{tf}"
            row_val = row.get(col, None)
            if row_val is None:
                return False

            if operator == "Less Than":
                if value is None or not (row_val < value):
                    return False
            elif operator == "Greater Than":
                if value is None or not (row_val > value):
                    return False
            elif operator == "Crossing Down":
                # row_val was previously >= value, now < value
                if value is None or prev_row is None:
                    return False
                prev_val = prev_row.get(col, None)
                if prev_val is None or not (prev_val >= value and row_val < value):
                    return False
            elif operator == "Crossing Up":
                # row_val was previously <= value, now > value
                if value is None or prev_row is None:
                    return False
                prev_val = prev_row.get(col, None)
                if prev_val is None or not (prev_val <= value and row_val > value):
                    return False
            else:
                # If user sets e.g. "Signal Value" as a string => direct match, or skip
                if isinstance(value, str):
                    # Then we do direct string match
                    if row_val != value:
                        return False
                else:
                    return False

        elif indicator == "TradingView":
            col = "tv_tech_label"
            if tf != "1m":
                col += f"_{tf}"
            row_val = row.get(col, None)
            if row_val is None:
                return False

            # Usually TradingView signals are strings like "Buy", "Strong Buy", etc.
            # If the operator is e.g. "==" => we do row_val == value
            # If operator is "Less Than" => might not be meaningful for strings,
            # so adapt as you see fit.
            if isinstance(value, str):
                if row_val != value:
                    return False
            else:
                # Possibly user wants "Less Than" something => not typical for strings
                return False

        elif indicator == "HeikenAshi":
            # e.g. "Candles in a Row"? => we only check HA_Close, or your logic
            col = "HA_Close"
            if tf != "1m":
                col += f"_{tf}"
            row_val = row.get(col, None)
            if row_val is None:
                return False

            # Then do operator checks if you want. Or e.g. "Candles in a Row" might require more logic.
            # We'll do a simple numeric compare if "operator" is e.g. "Greater Than".
            if operator == "Greater Than":
                if value is None or not (row_val > value):
                    return False
            elif operator == "Less Than":
                if value is None or not (row_val < value):
                    return False
            # etc.

        elif indicator == "MA":
            ma_type = subs.get("MA Type", "SMA")  # 'SMA' or 'EMA'
            fast_ma = subs.get("Fast MA", 14)
            slow_ma = subs.get("Slow MA", 28)
            col_fast = f"{ma_type}_{fast_ma}"
            col_slow = f"{ma_type}_{slow_ma}"
            if tf != "1m":
                col_fast += f"_{tf}"
                col_slow += f"_{tf}"
            val_fast = row.get(col_fast, None)
            val_slow = row.get(col_slow, None)
            if val_fast is None or val_slow is None:
                return False

            if operator == "Less Than":
                if not (val_fast < val_slow):
                    return False
            elif operator == "Greater Than":
                if not (val_fast > val_slow):
                    return False
            elif operator == "Crossing Down":
                if prev_row is None:
                    return False
                prev_fast = prev_row.get(col_fast, None)
                prev_slow = prev_row.get(col_slow, None)
                if prev_fast is None or prev_slow is None:
                    return False
                if not (prev_fast >= prev_slow and val_fast < val_slow):
                    return False
            elif operator == "Crossing Up":
                if prev_row is None:
                    return False
                prev_fast = prev_row.get(col_fast, None)
                prev_slow = prev_row.get(col_slow, None)
                if prev_fast is None or prev_slow is None:
                    return False
                if not (prev_fast <= prev_slow and val_fast > val_slow):
                    return False
            else:
                return False

        elif indicator == "BollingerBands":
            period = subs.get("BB% Period", 20)
            dev = subs.get("Deviation", 2)
            col = f"BB_%B_{period}_{dev}"
            if tf != "1m":
                col += f"_{tf}"
            row_val = row.get(col, None)
            if row_val is None:
                return False

            if operator == "Less Than":
                if value is None or not (row_val < value):
                    return False
            elif operator == "Greater Than":
                if value is None or not (row_val > value):
                    return False
            elif operator == "Crossing Down":
                if value is None or prev_row is None:
                    return False
                prev_val = prev_row.get(col, None)
                if prev_val is None or not (prev_val >= value and row_val < value):
                    return False
            elif operator == "Crossing Up":
                if value is None or prev_row is None:
                    return False
                prev_val = prev_row.get(col, None)
                if prev_val is None or not (prev_val <= value and row_val > value):
                    return False
            else:
                return False

        elif indicator == "MACD":
            macd_preset = subs.get("MACD Preset", "12,26,9")
            fast_str, slow_str, sig_str = macd_preset.split(',')

            # We know from data that the main line is "MACD_12_26_9"
            # and the signal line is "MACD_Signal_12_26_9"
            main_name   = f"MACD_{fast_str}_{slow_str}_{sig_str}"
            signal_name = f"MACD_Signal_{fast_str}_{slow_str}_{sig_str}"

            tf = subs.get("Timeframe", "1m")
            if tf != "1m":
                main_name   += f"_{tf}"
                signal_name += f"_{tf}"

            main_val   = row.get(main_name, None)
            signal_val = row.get(signal_name, None)
            if main_val is None or signal_val is None:
                return False

            # Now check the "MACD Trigger" crossing logic
            macd_trigger = subs.get("MACD Trigger", "")  # e.g. "Crossing Up" or "Crossing Down"
            if macd_trigger == "Crossing Up":
                if prev_row is None:
                    return False
                prev_main = prev_row.get(main_name, None)
                prev_sig  = prev_row.get(signal_name, None)
                if prev_main is None or prev_sig is None:
                    return False
                # crossing up => prev main <= prev signal AND current main > current signal
                if not (prev_main <= prev_sig and main_val > signal_val):
                    return False

            elif macd_trigger == "Crossing Down":
                if prev_row is None:
                    return False
                prev_main = prev_row.get(main_name, None)
                prev_sig  = prev_row.get(signal_name, None)
                if prev_main is None or prev_sig is None:
                    return False
                if not (prev_main >= prev_sig and main_val < signal_val):
                    return False

            # "Line Trigger" => e.g. "Less Than 0" or "Greater Than 0"
            line_trigger = subs.get("Line Trigger", "")
            if line_trigger == "Less Than 0":
                if main_val >= 0:
                    return False
            elif line_trigger == "Greater Than 0":
                if main_val <= 0:
                    return False

        elif indicator == "Stochastic":
            stoch_preset = subs.get("Stochastic Preset", "14,3,3")
            k_str, ksmooth_str, dsmooth_str = stoch_preset.split(',')
            k_col = f"Stochastic_K_{k_str}_{ksmooth_str}"
            d_col = f"Stochastic_D_{k_str}_{ksmooth_str}_{dsmooth_str}"
            if tf != "1m":
                k_col += f"_{tf}"
                d_col += f"_{tf}"
            k_val = row.get(k_col, None)
            d_val = row.get(d_col, None)
            if k_val is None:
                return False

            k_cond = subs.get("K Condition", "")
            k_sig_val = subs.get("K Signal Value", None)
            if k_cond == "Less Than":
                if k_sig_val is None or not (k_val < k_sig_val):
                    return False
            elif k_cond == "Greater Than":
                if k_sig_val is None or not (k_val > k_sig_val):
                    return False
            elif k_cond == "Crossing Down":
                if prev_row is None or k_sig_val is None:
                    return False
                prev_k = prev_row.get(k_col, None)
                if prev_k is None or not (prev_k >= k_sig_val and k_val < k_sig_val):
                    return False
            elif k_cond == "Crossing Up":
                if prev_row is None or k_sig_val is None:
                    return False
                prev_k = prev_row.get(k_col, None)
                if prev_k is None or not (prev_k <= k_sig_val and k_val > k_sig_val):
                    return False

            main_condition = subs.get("Condition", "")
            if main_condition == "K Crossing Up D":
                if prev_row is None or d_val is None:
                    return False
                prev_k = prev_row.get(k_col, None)
                prev_d = prev_row.get(d_col, None)
                if prev_k is None or prev_d is None or not (prev_k <= prev_d and k_val > d_val):
                    return False
            elif main_condition == "K Crossing Down D":
                if prev_row is None or d_val is None:
                    return False
                prev_k = prev_row.get(k_col, None)
                prev_d = prev_row.get(d_col, None)
                if prev_k is None or prev_d is None or not (prev_k >= prev_d and k_val < d_val):
                    return False

        elif indicator == "ParabolicSAR":
            psar_str = subs.get("PSAR Preset", "0.02,0.2")
            step_str, max_str = psar_str.split(',')
            col = f"PSAR_AF_{step_str}_Max_{max_str}"
            if tf != "1m":
                col += f"_{tf}"
            row_val = row.get(col, None)
            if row_val is None:
                return False

            if operator in ["Crossing (Long)", "Crossing (Short)"]:
                if prev_row is None:
                    return False
                prev_val = prev_row.get(col, None)
                if prev_val is None:
                    return False
                close_now = row.get('close', None)
                close_prev = prev_row.get('close', None)
                if close_now is None or close_prev is None:
                    return False
                if operator == "Crossing (Long)":
                    if not (close_prev <= prev_val and close_now > row_val):
                        return False
                elif operator == "Crossing (Short)":
                    if not (close_prev >= prev_val and close_now < row_val):
                        return False
            else:
                # Possibly user wants e.g. "Less Than" => row_val < X, etc.
                if operator == "Less Than":
                    if value is None or not (row_val < value):
                        return False
                elif operator == "Greater Than":
                    if value is None or not (row_val > value):
                        return False
                # etc.
                else:
                    return False

        else:
            # Unknown indicator => automatically fail
            return False

        # If we never returned False within this condition block => condition passes
        # Move on to next condition

    # If we never fail any condition => pass
    return True

# ─────────────────────────────────────────────────────────────

def compute_metrics(df_out, initial_balance):
    """
    Given the final trades DataFrame (df_out) and the initial_balance,
    compute various performance metrics:
      - net profit
      - average daily profit
      - max deal duration, average deal duration
      - yearly return
      - profit factor
      - Sharpe ratio
      - Sortino ratio
      - total trades
      - win rate
      - average profit per trade
      - risk-reward ratio
      - exposure time
      - value at risk (VaR)
    Returns a dict of these metrics.
    """

    # 1) Net Profit
    # last row's "realized_balance" minus initial, / initial => decimal
    last_realized = df_out["realized_balance"].iloc[-1]
    net_profit = (last_realized - initial_balance) / initial_balance

    # 2) We figure out "average daily profit" => net profit / # of days
    #    We'll compute the total time from first timestamp to last timestamp
    df_out["timestamp"] = pd.to_datetime(df_out["timestamp"])
    start_ts = df_out["timestamp"].iloc[0]
    end_ts   = df_out["timestamp"].iloc[-1]
    total_minutes = (end_ts - start_ts).total_seconds() / 60.0
    total_days = total_minutes / (60.0*24.0)
    if total_days <= 0:
        average_daily_profit = 0.0
    else:
        average_daily_profit = net_profit / total_days

    # 3) Max deal duration & average deal duration
    #    We'll define a "deal" from the first BUY to the EXIT for each trade_id
    #    We'll store durations in a list
    deal_durations = []
    # We'll track {trade_id: open_timestamp} upon a buy, then compute close
    open_time = {}

    for i, row in df_out.iterrows():
        t_id = row["trade_id"]
        action_lc = str(row["action"]).lower()
        ts = row["timestamp"]

        if "buy" in action_lc or "safety" in action_lc:
            # if not open_time.get(t_id), store
            if t_id not in open_time:
                open_time[t_id] = ts

        elif "sell" in action_lc or "exit" in action_lc:
            if t_id in open_time:
                duration = ts - open_time[t_id]
                deal_durations.append(duration)
                # remove from open_time
                del open_time[t_id]

    if len(deal_durations)==0:
        max_deal_duration = "0 days"
        avg_deal_duration = "0 days"
    else:
        max_dur = max(deal_durations)
        avg_dur = sum(deal_durations, pd.Timedelta(0)) / len(deal_durations)

        # We'll define a helper to format a Timedelta => "X days, HH:MM:SS"
        def fmt_td(td):
            # total secs
            secs = int(td.total_seconds())
            days, secs = divmod(secs, 86400)
            hours, secs = divmod(secs, 3600)
            minutes, secs = divmod(secs, 60)
            out = f"{days} days, {hours} hours, {minutes} minutes"
            return out

        max_deal_duration = fmt_td(max_dur)
        avg_deal_duration = fmt_td(avg_dur)

    # 4) Yearly return => net_profit scaled to 1 year
    #    Let's define total_years = total_minutes / (525600.0)
    total_years = total_minutes / 525600.0
    if total_years>0:
        yearly_return = (1 + net_profit)**(1/total_years) - 1
    else:
        yearly_return = 0.0

    # 5) Profit factor => gross profit / gross loss
    #    We'll parse each exit row's profit_loss
    #    If it's >0 => goes to gross_profit, <0 => absolute value to gross_loss
    gross_profit = 0.0
    gross_loss   = 0.0
    if "profit_loss" in df_out.columns:
        # We'll consider any row that is an exit (or "sell") => parse profit_loss
        for i, row2 in df_out.iterrows():
            act_lc = str(row2["action"]).lower()
            pl = row2["profit_loss"]
            if (("sell" in act_lc) or ("exit" in act_lc)) and (pl != 0):
                if pl>0:
                    gross_profit += pl
                else:
                    gross_loss += abs(pl)
    if gross_loss>0:
        profit_factor = gross_profit/gross_loss
    else:
        profit_factor = float("inf") if gross_profit>0 else 1.0

    # 6) Sharpe Ratio => (mean(daily returns) - riskfree)/ std(daily returns)
    #    We'll do a quick hack: we can parse daily "realized_balance" => compute daily returns
    #    For a real approach, you'd do something more robust
    #    We'll assume riskFree=0 for simplicity
    daily_bal = df_out.resample("1D", on="timestamp")["realized_balance"].last().ffill()
    daily_ret = daily_bal.pct_change().dropna()
    if len(daily_ret)>1:
        sharpe_ratio = daily_ret.mean()/daily_ret.std() * np.sqrt(252)  # sqrt(252 trading days)
    else:
        sharpe_ratio = 0.0

    # 7) Sortino Ratio => like Sharpe but only downside stdev
    #    We'll define negative returns
    neg_ret = daily_ret[daily_ret<0]
    if len(neg_ret)>0:
        downside_std = neg_ret.std()
        sortino_ratio= daily_ret.mean()/downside_std * np.sqrt(252)
    else:
        sortino_ratio= 0.0

    # 8) Total Trades => number of executed trades
    #    We'll count how many times we see an EXIT or SELL
    total_trades = 0
    # 9) Win Rate => how many trades ended with profit>0
    wins = 0

    if "profit_loss" in df_out.columns:
        # each time we see an exit => increment total_trades
        # if pl>0 => increment wins
        for i, row3 in df_out.iterrows():
            act_lc = str(row3["action"]).lower()
            pl = row3["profit_loss"]
            if ("sell" in act_lc) or ("exit" in act_lc):
                total_trades +=1
                if pl>0:
                    wins+=1

    if total_trades>0:
        win_rate = wins/ total_trades
    else:
        win_rate = 0.0

    # 10) Average Profit per Trade => sum of all (pl) / total trades
    if total_trades>0:
        avg_profit_per_trade = (gross_profit - gross_loss)/ total_trades
    else:
        avg_profit_per_trade = 0.0

    # 11) Risk-Reward => average profit vs average loss
    #    average profit => gross_profit / (#winning trades)
    #    average loss   => gross_loss / (#losing trades)
    #    ratio => averageProfit / averageLoss
    num_wins    = (wins)
    num_losses  = total_trades - wins
    if num_wins>0:
        avg_win_amt  = gross_profit/ num_wins
    else:
        avg_win_amt  = 0.0

    if num_losses>0:
        avg_loss_amt = gross_loss/ num_losses
    else:
        avg_loss_amt = 1.0  # avoid zero

    risk_reward_ratio = (avg_win_amt/ avg_loss_amt) if avg_loss_amt>0 else float("inf")

    # 12) Exposure Time => % of time the strategy was in a position
    #    We'll do a simple approach => if positions sum>0 => in position
    #    We'll parse row-> row+1. This is an approximation for exposure time
    #    You can do more robust approach with each trade's in/out
    # For simplicity: if sum(positions_by_symbol) >0 => in position => we skip for next row
    # We'll do a rough approach => we do a step by step
    # This is an approximation
    # For a robust approach, track each trade's start->end and union all intervals
    # We'll do quick version:
    in_position_minutes = 0.0
    for i in range(len(df_out)-1):
        sumpos = 0
        # We can parse if "position_held" is in the df
        sumpos = df_out["position_held"].iloc[i]
        t1 = df_out["timestamp"].iloc[i]
        t2 = df_out["timestamp"].iloc[i+1]
        delta = (t2 - t1).total_seconds()/60.0
        if sumpos>0:
            in_position_minutes += delta

    exposure_time_frac = 0.0
    if total_minutes>0:
        exposure_time_frac = in_position_minutes/ total_minutes

    # 13) Value-at-Risk => advanced, we can do e.g. daily returns at 5% quantile
    #    We'll do quick approach => VaR= negative of daily_ret.quantile(0.05)
    #    means 5% chance daily return < that
    #    This is simplistic
    var_95 = 0.0
    if len(daily_ret)>0:
        var_95 = -daily_ret.quantile(0.05)

    metrics = {
        "net_profit": net_profit,
        "average_daily_profit": average_daily_profit,
        "max_deal_duration": max_deal_duration,
        "avg_deal_duration": avg_deal_duration,
        "yearly_return": yearly_return,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_profit_per_trade": avg_profit_per_trade,
        "risk_reward_ratio": risk_reward_ratio,
        "exposure_time_frac": exposure_time_frac,
        "var_95": var_95
    }
    return metrics

# 6) RUN_BACKTEST ROUTE (SAME CODE, NO CHANGES)
# ─────────────────────────────────────────────────────────────
@app.route('/run_backtest', methods=['POST'])
def run_backtest():
    from multiprocessing import Manager, Lock
    from datetime import datetime
    import json

    try:
        # 1) Parse JSON input
        data = request.json
        print("========== USER INPUT PAYLOAD ==========")
        print(json.dumps(data, indent=2))
        print("========================================")

        # 2) Extract ALL payload fields
        payload = get_user_payload(data)  # as defined in your snippet

        # Basic fields we use repeatedly
        pairs           = payload.get("pairs", [])
        strategy_name   = payload.get("strategy_name", '')
        max_active_deals= payload.get("max_active_deals", 0)
        initial_balance = payload.get("initial_balance", 10000.0)  # you can default here if you like

        # If no pairs, fail early
        if not pairs:
            return jsonify({"status": "error", "message": "No pairs selected."})

        # Additional numeric fields
        trading_fee          = payload.get("trading_fee", 0.0)
        base_order_size      = payload.get("base_order_size", 0.0)
        safety_order_size    = payload.get("safety_order_size", 0.0)
        risk_reduction       = payload.get("risk_reduction", 0.0)
        reinvest_profit      = payload.get("reinvest_profit", 0.0)

        # Derived (fractions, etc.)
        base_order_frac      = (base_order_size / initial_balance) if initial_balance else 0.0
        safety_order_frac    = (safety_order_size / initial_balance) if initial_balance else 0.0
        fee_rate             = (trading_fee / 100.0)

        # 3) Gather needed columns
        req_cols = gather_required_columns(
            payload["entry_conditions"],
            payload["safety_conditions"],
            payload["exit_conditions"]
        )

        # 4) Load data in parallel
        dfs_map = load_parquets_in_parallel(pairs, req_cols)

        manager = Manager()
        lock    = manager.Lock()
        manager_dict = manager.dict()
        manager_dict["global_open_trades"] = 0

        # Prepare directories
        BACKTEST_RESULTS_DIR = os.path.join(DATA_DIR, "backtest_results", strategy_name)
        os.makedirs(BACKTEST_RESULTS_DIR, exist_ok=True)

        all_symbol_trades = []

        # 5) Run each symbol's backtest in parallel => get trades
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor() as executor:
            future_map = {}
            for sym in pairs:
                df_sym = dfs_map[sym]
                fut = executor.submit(
                    run_symbol_backtest,
                    df_sym,
                    sym,
                    manager_dict,
                    lock,
                    payload,
                    check_all_user_conditions,
                    req_cols
                )
                future_map[fut] = sym

            for fut in as_completed(future_map):
                symbol = future_map[fut]
                try:
                    trades = fut.result()  # list of dicts
                    if trades:
                        df_trades = pd.DataFrame(trades).sort_values("timestamp")
                        out_name  = f"trades_{symbol.replace('/','_')}.csv"
                        out_path  = os.path.join(BACKTEST_RESULTS_DIR, out_name)
                        df_trades.to_csv(out_path, index=False)
                        print(f"Symbol {symbol}: {len(trades)} trades => saved {out_name}")
                        all_symbol_trades.extend(trades)
                    else:
                        print(f"No trades for {symbol}")
                except Exception as e:
                    msg = f"Error in {symbol}: {e}"
                    print(msg)
                    return jsonify({"status": "error", "message": msg})

        if not all_symbol_trades:
            # No trades => no metrics
            return jsonify({
                "status": "success",
                "message": "No trades generated => cannot display metrics.",
                "results_directory": BACKTEST_RESULTS_DIR
            })

        # 6) Sort by time => apply concurrency filter
        df_all = pd.DataFrame(all_symbol_trades).sort_values("timestamp").reset_index(drop=True)

        active_deals_count = 0
        symbol_is_open     = {p: False for p in pairs}
        skipped_trade_ids  = set()

        def is_buy_or_entry(action_text):
            text = str(action_text).lower()
            return ("buy" in text) or ("safety" in text)

        def is_exit_action(action_text):
            text = str(action_text).lower()
            return ("sell" in text) or ("exit" in text)

        filtered_rows = []
        for _, row in df_all.iterrows():
            t_id = row.get("trade_id", "")
            sym  = row["symbol"]
            act  = row["action"]

            if t_id in skipped_trade_ids:
                continue

            if is_buy_or_entry(act):
                # if symbol not open, we can open it if concurrency allows
                if not symbol_is_open[sym]:
                    if active_deals_count < max_active_deals:
                        symbol_is_open[sym] = True
                        active_deals_count += 1
                        filtered_rows.append(row)
                    else:
                        skipped_trade_ids.add(t_id)
                else:
                    filtered_rows.append(row)

            elif is_exit_action(act):
                if symbol_is_open[sym]:
                    symbol_is_open[sym] = False
                    active_deals_count -= 1
                    filtered_rows.append(row)
                else:
                    filtered_rows.append(row)
            else:
                # e.g. not recognized => keep it
                filtered_rows.append(row)

        if not filtered_rows:
            return jsonify({
                "status": "success",
                "message": "All trades were skipped => no final CSV => no metrics.",
                "results_directory": BACKTEST_RESULTS_DIR
            })

        # 7) Build final trades DataFrame with realized_balance, etc.
        df_filtered = pd.DataFrame(filtered_rows).sort_values("timestamp").reset_index(drop=True)

        balance       = initial_balance
        real_balance  = initial_balance
        free_cash     = initial_balance

        positions_by_symbol = defaultdict(float)
        last_close = defaultdict(float)
        trade_accum = {}
        out_records = []

        max_balance_so_far  = initial_balance
        max_drawdown_so_far = 0.0

        for row in df_filtered.to_dict("records"):
            sym   = row["symbol"]
            act   = str(row["action"]).lower()
            tid   = row.get("trade_id", "")
            px    = float(row["price"])
            amt   = float(row["amount"])
            tot   = float(row["total_amount"])
            ppct_ = row.get("profit_percent", "")
            ppct  = float(ppct_) if ppct_ != "" else 0.0

            # Always update last_close
            last_close[sym] = px

            position        = 0.0
            order_size      = 0.0
            profit_loss     = 0.0
            position_change = 0.0

            if tid not in trade_accum:
                trade_accum[tid] = {
                    "position": 0.0,
                    "trade_size": 0.0,
                    "fraction": (real_balance / initial_balance) if initial_balance else 0.0
                }

            # Check if buy / safety
            if is_buy_or_entry(act):
                fraction   = trade_accum[tid]["fraction"]
                order_size = fraction * amt
                position   = order_size / px if px > 0 else 0.0

                positions_by_symbol[sym] += position
                position_change           = position
                free_cash                -= order_size

                trade_accum[tid]["position"]   += position
                trade_accum[tid]["trade_size"] += order_size

            # Check if exit
            elif is_exit_action(act):
                position = trade_accum[tid]["position"]
                if position < 0:
                    position = 0.0

                order_size  = position * px
                profit_loss = order_size - trade_accum[tid]["trade_size"]

                positions_by_symbol[sym] -= position
                position_change = -position
                free_cash      += order_size

                # reduce or add to 'balance' based on profit
                if profit_loss < 0:
                    balance += (profit_loss * (risk_reduction / 100.0))  # if risk_reduction was %
                elif profit_loss > 0:
                    balance += (profit_loss * (reinvest_profit / 100.0)) # if reinvest_profit was %

                # Mark trade closed
                trade_accum[tid]["position"]   = 0.0
                trade_accum[tid]["trade_size"] = 0.0

                real_balance += profit_loss

            # Recompute realized_balance
            realized_balance = free_cash
            for s, q in positions_by_symbol.items():
                realized_balance += q * last_close[s]

            # Track drawdowns
            if realized_balance > max_balance_so_far:
                max_balance_so_far = realized_balance

            current_drawdown = 0.0
            if max_balance_so_far > 0:
                current_drawdown = (max_balance_so_far - realized_balance) / max_balance_so_far
            if current_drawdown > max_drawdown_so_far:
                max_drawdown_so_far = current_drawdown

            out_rec = {
                "timestamp":         row["timestamp"],
                "symbol":            sym,
                "action":            row["action"],
                "price":             round(px, 4),
                "amount":            round(amt, 2),
                "total_amount":      round(tot, 2),
                "trade_comment":     row.get("trade_comment", ""),
                "trade_id":          tid,
                "profit_percent":    ppct,

                "position":          round(trade_accum[tid]["position"], 6),
                "order_size":        round(order_size, 2),
                "trade_size":        round(trade_accum[tid]["trade_size"], 2),
                "profit_loss":       round(profit_loss, 2),
                "balance":           round(balance, 2),
                "real_balance":      round(real_balance, 2),
                "free_cash":         round(free_cash, 2),
                "position_change":   round(position_change, 6),
                "position_held":     round(positions_by_symbol[sym], 6),
                "realized_balance":  round(realized_balance, 2),

                "drawdown":          round(current_drawdown, 4),
                "max_drawdown":      round(max_drawdown_so_far, 4),
            }
            out_records.append(out_rec)

        df_out = pd.DataFrame(out_records, columns=[
            "timestamp","symbol","action","price","amount","total_amount",
            "trade_comment","trade_id","profit_percent",
            "position","order_size","trade_size","profit_loss","balance",
            "real_balance","free_cash","position_change","position_held",
            "realized_balance","drawdown","max_drawdown"
        ])

        # Write final trades CSV
        out_final = os.path.join(BACKTEST_RESULTS_DIR, "all_trades_combined.csv")
        df_out.to_csv(out_final, index=False)
        print(f"Final backtest => {out_final}")

        # -------------------------------------------
        # 8) Compute metrics
        # -------------------------------------------
        if df_out.empty:
            return jsonify({"status": "error", "message": "No trade data available."})

        metrics = compute_metrics(df_out, initial_balance)
        print("Metrics:", metrics)

        # -------------------------------------------
        # 9) Write summary CSV (payload + metrics)
        # -------------------------------------------
        summary_csv_path = os.path.join(BACKTEST_RESULTS_DIR, "backtest_summary_metrics.csv")

        # Convert arrays / booleans to JSON strings so they fit in CSV
        summary_data = {
            # Basic info about this run
            "timestamp_run":           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "strategy_name":           payload.get("strategy_name", ""),
            "pairs":                   json.dumps(pairs),  # store as JSON text
            "initial_balance":         initial_balance,
            "max_active_deals":        max_active_deals,
            "trading_fee":             payload.get("trading_fee", 0.0),
            "base_order_size":         base_order_size,
            "start_date":              payload.get("start_date", ""),
            "end_date":                payload.get("end_date", ""),

            # Conditions
            "entry_conditions":        json.dumps(payload.get("entry_conditions", [])),
            "safety_order_toggle":     payload.get("safety_order_toggle", False),
            "safety_order_size":       safety_order_size,
            "price_deviation":         payload.get("price_deviation", 0.0),
            "max_safety_orders_count": payload.get("max_safety_orders_count", 0),
            "max_active_safety_orders_count": payload.get("max_active_safety_orders_count", 0),
            "safety_order_volume_scale": payload.get("safety_order_volume_scale", 0.0),
            "safety_order_step_scale":   payload.get("safety_order_step_scale", 0.0),
            "safety_conditions":       json.dumps(payload.get("safety_conditions", [])),
            "price_change_active":     payload.get("price_change_active", False),
            "conditions_active":       payload.get("conditions_active", False),
            "take_profit_type":        payload.get("take_profit_type", ""),
            "target_profit":           payload.get("target_profit", 0.0),
            "trailing_toggle":         payload.get("trailing_toggle", False),
            "trailing_deviation":      payload.get("trailing_deviation", 0.0),
            "exit_conditions":         json.dumps(payload.get("exit_conditions", [])),
            "minprof_toggle":          payload.get("minprof_toggle", False),
            "minimal_profit":          payload.get("minimal_profit", 0),
            "reinvest_profit":         payload.get("reinvest_profit", 0.0),
            "stop_loss_toggle":        payload.get("stop_loss_toggle", False),
            "stop_loss_value":         payload.get("stop_loss_value", 0.0),
            "stop_loss_timeout":       payload.get("stop_loss_timeout", 0.0),
            "stop_loss_trailing":      payload.get("stop_loss_trailing", False),
            "risk_reduction":          payload.get("risk_reduction", 0.0),
            "min_daily_volume":        payload.get("min_daily_volume", 0.0),
            "cooldown_between_deals":  payload.get("cooldown_between_deals", 0),

            # Derived fields
            "base_order_frac":         base_order_frac,
            "safety_order_frac":       safety_order_frac,
            "fee_rate":                fee_rate,

            # Metrics
            "net_profit":              metrics.get("net_profit", 0),
            "average_daily_profit":    metrics.get("average_daily_profit", 0),
            "max_deal_duration":       metrics.get("max_deal_duration", ""),
            "avg_deal_duration":       metrics.get("avg_deal_duration", ""),
            "yearly_return":           metrics.get("yearly_return", 0),
            "profit_factor":           metrics.get("profit_factor", 0),
            "sharpe_ratio":            metrics.get("sharpe_ratio", 0),
            "sortino_ratio":           metrics.get("sortino_ratio", 0),
            "total_trades":            metrics.get("total_trades", 0),
            "win_rate":                metrics.get("win_rate", 0),
            "avg_profit_per_trade":    metrics.get("avg_profit_per_trade", 0),
            "risk_reward_ratio":       metrics.get("risk_reward_ratio", 0),
            "exposure_time_frac":      metrics.get("exposure_time_frac", 0),
            "var_95":                  metrics.get("var_95", 0),
        }

        df_summary = pd.DataFrame([summary_data])

        # Append or create new summary file
        if not os.path.exists(summary_csv_path):
            df_summary.to_csv(summary_csv_path, index=False)
        else:
            df_summary.to_csv(summary_csv_path, mode='a', header=False, index=False)

        print(f"Appended run summary to {summary_csv_path}")

        # -------------------------------------------
        # 10) Build chart_data for Realized Balance
        # -------------------------------------------
        chart_data = {
            "timestamps": df_out["timestamp"].astype(str).tolist(),
            "realized_balance": df_out["realized_balance"].tolist(),
        }

        # -------------------------------------------
        # 11) Compute BUY & HOLD line
        # -------------------------------------------
        start_ts = df_out["timestamp"].min()
        end_ts   = df_out["timestamp"].max()

        unique_symbols = df_out["symbol"].unique().tolist()
        num_symbols    = len(unique_symbols)
        if num_symbols == 0:
            # no symbols => can't do B/H
            return render_template("metrics_results.html", metrics=metrics, chartData=chart_data)

        per_symbol_balance = initial_balance / num_symbols if num_symbols else 0.0
        full_index = pd.date_range(start=start_ts, end=end_ts, freq='1T')
        df_bh = pd.DataFrame(index=full_index)

        for sym in unique_symbols:
            file_path = os.path.join(DATA_DIR, f"{sym.replace('/','_')}_all_tf_merged.parquet")
            if not os.path.exists(file_path):
                continue

            df_sym = pd.read_parquet(file_path, columns=["timestamp","close"])
            df_sym["timestamp"] = pd.to_datetime(df_sym["timestamp"])
            df_sym.set_index("timestamp", inplace=True)
            df_sym = df_sym.resample("1T").ffill()

            df_sym_range = df_sym.loc[start_ts:end_ts]
            if df_sym_range.empty:
                continue

            first_close = df_sym_range["close"].iloc[0]
            if first_close <= 0:
                continue

            coins_held = per_symbol_balance / first_close if first_close else 0.0
            df_sym["value_"+sym] = df_sym["close"] * coins_held

            df_bh = df_bh.join(df_sym["value_"+sym], how='left')

        value_cols = [c for c in df_bh.columns if c.startswith("value_")]
        if len(value_cols) == 0:
            return render_template("metrics_results.html", metrics=metrics, chartData=chart_data)

        df_bh["bh_balance"] = df_bh[value_cols].sum(axis=1).ffill()
        df_bh.reset_index(inplace=True)
        df_bh.rename(columns={"index": "timestamp"}, inplace=True)

        chart_data["bh_timestamps"] = df_bh["timestamp"].astype(str).tolist()
        chart_data["bh_balance"]    = df_bh["bh_balance"].tolist()

        # -------------------------------------------
        # 12) Render template with metrics + chart data
        # -------------------------------------------
        return render_template("metrics_results.html", metrics=metrics, chartData=chart_data)

    except Exception as e:
        print("Exception in run_backtest:", e)
        return jsonify({"status": "error", "message": str(e)})



# ─────────────────────────────────────────────────────────────
# 7) OPTIONAL: your other routes (index, get_crypto_pairs, etc.)
# ─────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('backtest.html')

@app.route('/templates/css/<path:filename>')
def css(filename):
    return send_from_directory('templates/css', filename)

@app.route('/templates/js/<path:filename>')
def js(filename):
    return send_from_directory('templates/js', filename)

@app.route('/get_crypto_pairs', methods=['GET'])
def get_crypto_pairs():
    pairs = []
    try:
        for file in os.listdir(DATA_DIR):
            if file.endswith('.parquet'):
                pair = file.split('_')[0] + '/' + file.split('_')[1]
                pairs.append(pair)
    except Exception as e:
        print("Error reading pairs:", e)
    return jsonify(pairs)


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    app.run(host="0.0.0.0", port=5012)
