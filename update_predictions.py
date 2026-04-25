"""
Grant's Kronos demo — probabilistic BTC/USDT forecast.

Adapted from shiyu-coder/Kronos-demo. Changes vs. the upstream demo:
  - Model: Kronos-Base (102M params) instead of Kronos-mini (4M)
  - Interval: 10-min candles (resampled from Binance 5m — Binance has no native 10m)
  - Context: capped at 512 candles (~85h) — Kronos-Base's max_context limit
  - Horizon: 24h = 144 10-min candles
  - Git push + scheduler disabled for local testing (re-enable when deploying to a fork)
"""

import gc
import os
import re
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from binance.client import Client

from model import KronosTokenizer, Kronos, KronosPredictor

# --- Configuration ---
Config = {
    "REPO_PATH": Path(__file__).parent.resolve(),
    "MODEL_PATH": "./Kronos_model",
    "SYMBOLS": ['BTCUSD'],
    "FETCH_INTERVAL": '5m',    # Binance has no 10m — fetch 5m and resample 2-into-1.
    "INTERVAL_MIN": 10,        # target candle interval in minutes.
    "HIST_POINTS": 512,        # Kronos-Base max_context cap (~85h at 10-min).
    "PRED_HORIZON": 144,       # 24h forecast × 6 candles/hr at 10-min.
    "N_PREDICTIONS": 10,       # Monte Carlo sample count.
    "VOL_WINDOW": 144,         # last 24h of candles for historical vol baseline.
}


def load_model():
    """Loads Kronos-Base + matching tokenizer from Hugging Face Hub."""
    print("Loading Kronos-Base (102M params)...")
    tokenizer = KronosTokenizer.from_pretrained(
        "NeoQuasar/Kronos-Tokenizer-base", cache_dir=Config["MODEL_PATH"]
    )
    model = Kronos.from_pretrained(
        "NeoQuasar/Kronos-base", cache_dir=Config["MODEL_PATH"]
    )
    tokenizer.eval()
    model.eval()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
    print("Model loaded successfully.")
    return predictor


def fetch_binance_data(symbol):
    """Fetch 5-min K-lines from Binance and resample to 10-min candles."""
    fetch_interval = Config["FETCH_INTERVAL"]
    interval_min = Config["INTERVAL_MIN"]

    target_bars = Config["HIST_POINTS"] + Config["VOL_WINDOW"] + 10
    hours_back = int((target_bars * interval_min) / 60) + 4

    print(f"Fetching ~{hours_back}h of {symbol} {fetch_interval} data from Binance.US...")
    client = Client(tld='us')   # Binance.com geo-blocks US data centers (incl. GitHub Actions); use Binance.US
    klines = client.get_historical_klines(
        symbol=symbol,
        interval=fetch_interval,
        start_str=f"{hours_back} hours ago UTC",
    )

    cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
            'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume', 'ignore']
    df = pd.DataFrame(klines, columns=cols)
    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']]
    df.rename(columns={'quote_asset_volume': 'amount', 'open_time': 'timestamps'}, inplace=True)
    df['timestamps'] = pd.to_datetime(df['timestamps'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        df[col] = pd.to_numeric(df[col])

    df = (
        df.set_index('timestamps')
          .resample(f'{interval_min}min')
          .agg({'open': 'first', 'high': 'max', 'low': 'min',
                'close': 'last', 'volume': 'sum', 'amount': 'sum'})
          .dropna()
          .reset_index()
    )

    print(f"Fetched + resampled → {len(df)} {interval_min}-min candles for {symbol}.")
    return df


def make_prediction(df, predictor):
    """Run Kronos with N Monte Carlo samples — returns per-path forecasts."""
    interval_min = Config["INTERVAL_MIN"]
    last_timestamp = df['timestamps'].max()
    start_new_range = last_timestamp + pd.Timedelta(minutes=interval_min)
    new_timestamps_index = pd.date_range(
        start=start_new_range,
        periods=Config["PRED_HORIZON"],
        freq=f'{interval_min}min',
    )
    y_timestamp = pd.Series(new_timestamps_index, name='y_timestamp')
    x_timestamp = df['timestamps']
    x_df = df[['open', 'high', 'low', 'close', 'volume', 'amount']]

    with torch.no_grad():
        print(f"Running {Config['N_PREDICTIONS']} Monte Carlo samples (T=1.0)...")
        t0 = time.time()
        close_preds, volume_preds = predictor.predict(
            df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
            pred_len=Config["PRED_HORIZON"], T=1.0, top_p=0.95,
            sample_count=Config["N_PREDICTIONS"], verbose=True,
        )
        print(f"Prediction completed in {time.time() - t0:.1f}s.")

    return close_preds, volume_preds, close_preds


def calculate_metrics(hist_df, close_preds_df, v_close_preds_df):
    """Upside probability and volatility amplification probability."""
    last_close = hist_df['close'].iloc[-1]

    final_step_preds = close_preds_df.iloc[-1]
    upside_prob = (final_step_preds > last_close).mean()

    hist_log_returns = np.log(hist_df['close'] / hist_df['close'].shift(1))
    historical_vol = hist_log_returns.iloc[-Config["VOL_WINDOW"]:].std()

    amplification_count = 0
    for col in v_close_preds_df.columns:
        full_sequence = pd.concat(
            [pd.Series([last_close]), v_close_preds_df[col]]
        ).reset_index(drop=True)
        pred_log_returns = np.log(full_sequence / full_sequence.shift(1))
        predicted_vol = pred_log_returns.std()
        if predicted_vol > historical_vol:
            amplification_count += 1

    vol_amp_prob = amplification_count / len(v_close_preds_df.columns)

    print(f"Upside Probability (24h): {upside_prob:.2%}, "
          f"Volatility Amplification Probability: {vol_amp_prob:.2%}")
    return upside_prob, vol_amp_prob


def create_plot(symbol, hist_df, close_preds_df, volume_preds_df):
    """Two-panel chart: price (history + mean + full-range band) and volume."""
    interval_min = Config["INTERVAL_MIN"]
    horizon_hours = Config["PRED_HORIZON"] * interval_min // 60

    print(f"Generating forecast chart for {symbol}...")
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 10), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]},
    )

    hist_time = hist_df['timestamps']
    last_hist_time = hist_time.iloc[-1]
    pred_time = pd.to_datetime([
        last_hist_time + timedelta(minutes=interval_min * (i + 1))
        for i in range(len(close_preds_df))
    ])

    ax1.plot(hist_time, hist_df['close'], color='royalblue',
             label='Historical Price', linewidth=1.5)
    mean_preds = close_preds_df.mean(axis=1)
    ax1.plot(pred_time, mean_preds, color='darkorange', linestyle='-',
             label='Mean Forecast')
    ax1.fill_between(
        pred_time, close_preds_df.min(axis=1), close_preds_df.max(axis=1),
        color='darkorange', alpha=0.2, label='Forecast Range (Min-Max)',
    )
    ax1.set_title(
        f'{symbol} Probabilistic Price & Volume Forecast '
        f'(Next {horizon_hours}h, {interval_min}-min candles)',
        fontsize=16, weight='bold',
    )
    ax1.set_ylabel('Price (USDT)')
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    bar_width = (interval_min / (24 * 60)) * 0.9

    ax2.bar(hist_time, hist_df['volume'], color='skyblue',
            label='Historical Volume', width=bar_width)
    ax2.bar(pred_time, volume_preds_df.mean(axis=1), color='sandybrown',
            label='Mean Forecasted Volume', width=bar_width)
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Time (UTC)')
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    separator_time = hist_time.iloc[-1] + timedelta(minutes=interval_min / 2)
    for ax in [ax1, ax2]:
        ax.axvline(x=separator_time, color='red', linestyle='--', linewidth=1.5,
                   label='_nolegend_')
        ax.tick_params(axis='x', rotation=30)

    fig.tight_layout()
    chart_path = Config["REPO_PATH"] / f'prediction_chart_{symbol.lower()}.png'
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)
    print(f"Chart saved to: {chart_path}")


def update_html(results):
    """
    Inject per-symbol metrics and update-time into index.html.
    `results` is a list of dicts: [{'symbol': 'BTCUSDT', 'upside': 0.467, 'vol_amp': 0.933}, ...]
    """
    print("Updating index.html...")
    html_path = Config["REPO_PATH"] / 'index.html'
    now_pst_str = datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M:%S')

    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()

    content = re.sub(
        r'(<strong id="update-time">).*?(</strong>)',
        lambda m: f'{m.group(1)}{now_pst_str}{m.group(2)}', content,
    )

    for r in results:
        sym = r['symbol'].lower()
        content = re.sub(
            rf'(<p class="metric-value" id="upside-prob-{sym}">).*?(</p>)',
            lambda m: f'{m.group(1)}{r["upside"]:.1%}{m.group(2)}', content,
        )
        content = re.sub(
            rf'(<p class="metric-value" id="vol-amp-prob-{sym}">).*?(</p>)',
            lambda m: f'{m.group(1)}{r["vol_amp"]:.1%}{m.group(2)}', content,
        )

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("HTML file updated successfully.")


def git_commit_and_push(commit_message):
    """Re-enable this when deploying to a GitHub-Pages-backed fork."""
    print("Performing Git operations...")
    try:
        os.chdir(Config["REPO_PATH"])
        files_to_add = ['index.html'] + [
            f'prediction_chart_{s.lower()}.png' for s in Config["SYMBOLS"]
        ]
        subprocess.run(['git', 'add', *files_to_add],
                       check=True, capture_output=True, text=True)
        commit_result = subprocess.run(['git', 'commit', '-m', commit_message],
                                       check=True, capture_output=True, text=True)
        print(commit_result.stdout)
        push_result = subprocess.run(['git', 'push'],
                                     check=True, capture_output=True, text=True)
        print(push_result.stdout)
        print("Git push successful.")
    except subprocess.CalledProcessError as e:
        output = e.stdout if e.stdout else e.stderr
        if "nothing to commit" in output or "Your branch is up to date" in output:
            print("No new changes to commit or push.")
        else:
            print(f"A Git error occurred:\n--- STDOUT ---\n{e.stdout}"
                  f"\n--- STDERR ---\n{e.stderr}")


def main_task(model):
    """One full update cycle: for each symbol, fetch → predict → metrics → chart."""
    print("\n" + "=" * 60)
    print(f"Starting update task at {datetime.now(timezone.utc)}")
    print("=" * 60)

    results = []
    for symbol in Config["SYMBOLS"]:
        print(f"\n--- Processing {symbol} ---")
        df_full = fetch_binance_data(symbol)
        df_for_model = df_full.iloc[:-1]

        close_preds, volume_preds, v_close_preds = make_prediction(df_for_model, model)

        hist_df_for_plot = df_for_model.tail(Config["HIST_POINTS"])
        hist_df_for_metrics = df_for_model.tail(Config["VOL_WINDOW"])

        upside_prob, vol_amp_prob = calculate_metrics(
            hist_df_for_metrics, close_preds, v_close_preds,
        )
        create_plot(symbol, hist_df_for_plot, close_preds, volume_preds)
        results.append({
            'symbol': symbol,
            'upside': upside_prob,
            'vol_amp': vol_amp_prob,
        })

        del df_full, df_for_model, close_preds, volume_preds, v_close_preds
        del hist_df_for_plot, hist_df_for_metrics
        gc.collect()

    update_html(results)

    # Disabled for local testing. Re-enable once this folder lives in a GitHub Pages repo.
    # commit_message = f"Auto-update forecast for {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC"
    # git_commit_and_push(commit_message)

    print("-" * 60)
    print("--- Task completed successfully ---")
    print("-" * 60 + "\n")


def run_scheduler(model):
    """Hourly scheduler. Off by default — flip the __main__ block to enable."""
    while True:
        now = datetime.now(timezone.utc)
        next_run_time = (now + timedelta(hours=1)).replace(
            minute=0, second=5, microsecond=0,
        )
        sleep_seconds = (next_run_time - now).total_seconds()

        if sleep_seconds > 0:
            print(f"Current time: {now:%Y-%m-%d %H:%M:%S UTC}.")
            print(f"Next run at: {next_run_time:%Y-%m-%d %H:%M:%S UTC}. "
                  f"Waiting for {sleep_seconds:.0f} seconds...")
            time.sleep(sleep_seconds)

        try:
            main_task(model)
        except Exception as e:
            print(f"\n!!!!!! A critical error occurred in the main task !!!!!!!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print("Retrying in 5 minutes...\n")
            time.sleep(300)


if __name__ == '__main__':
    model_path = Path(Config["MODEL_PATH"])
    model_path.mkdir(parents=True, exist_ok=True)

    loaded_model = load_model()
    main_task(loaded_model)
    # run_scheduler(loaded_model)   # Enable for auto-hourly updates after deployment.
