# Grant's Kronos Demo

Probabilistic BTC/USDT forecast based on the [Kronos](https://github.com/shiyu-coder/Kronos) foundation model. Adapted from [shiyu-coder/Kronos-demo](https://github.com/shiyu-coder/Kronos-demo).

## What it does

- Fetches the last ~85 hours of BTC/USDT 10-min K-line data from Binance (resampled from 5-min — Binance has no native 10m interval)
- Runs Kronos-Base with 30 Monte Carlo samples to produce a distribution of 24-hour forecasts
- Publishes a static dashboard with: mean forecast line, min–max uncertainty band, upside probability, and volatility amplification probability

## Running locally

```bash
pip install -r requirements.txt
python update_predictions.py
```

Outputs: `prediction_chart_btcusdt.png`, and updated metrics/timestamp in `index.html`. Open `index.html` in a browser to view.

Kronos-Base is a 102M-param model — on CPU, one full run takes **~10–30 min**. On GPU (CUDA/MPS), ~1–3 min.

## Running on Google Colab (free T4 GPU)

Open `colab_notebook.ipynb` in Colab. Runtime → Change runtime type → T4 GPU. Run all cells.

## Deploying to GitHub Pages

1. Push this folder to a new GitHub repo
2. Repo → Settings → Pages → Source: `main` branch, `/ (root)`
3. Wait ~30s; your page is live at `https://<username>.github.io/<repo-name>/`

### Auto-refresh via GitHub Actions (recommended)

This repo ships with `.github/workflows/refresh.yml` — a workflow that runs hourly on GitHub's free runners (CPU, no GPU), executes `update_predictions.py`, and auto-commits the refreshed chart + HTML so GitHub Pages always serves a fresh forecast.

**One-time setup after pushing the repo to GitHub:**

1. Go to **Settings → Secrets and variables → Actions → New repository secret**
2. Name: `HF_TOKEN`, Value: your `hf_...` token from https://huggingface.co/settings/tokens
3. Go to **Actions** tab → enable workflows if prompted
4. Trigger the first run manually: Actions → "Refresh Forecast" → Run workflow

After that, it runs every hour at :00 UTC. Each run takes ~20–30 min on the free CPU runner.

### Other refresh options

- Re-enable `run_scheduler(loaded_model)` in the `__main__` block of `update_predictions.py` and run it on a persistent host
- Use the Colab notebook's Section 6 to push fresh results from each manual Colab run

## Spec notes (for reviewers)

This demo meets the original spec with two documented deviations:

| Spec | Implemented | Deviation |
|---|---|---|
| Kronos-Base | ✅ Kronos-Base | — |
| BTC/USDT | ✅ BTCUSDT | — |
| 10-min candles from Binance | ✅ resampled from 5-min | Binance has no native 10m interval (returns `{"code":-1120,"msg":"Invalid interval"}`) |
| 720h of context | ~85h of context | Kronos-Base `max_context=512`. 720h × 6 candles/hr = 4,320 candles ≫ 512. Capped at 512 to avoid silent truncation |
| N=30 Monte Carlo paths | ✅ | — |
| Mean forecast + uncertainty band + probability metrics | ✅ | — |

## Files

| File | Purpose |
|---|---|
| `update_predictions.py` | Main script: fetch → predict → metrics → chart → HTML |
| `index.html` | Static dashboard page (served by GitHub Pages) |
| `style.css` | Page styling |
| `model/` | Kronos model code (`kronos.py`, `module.py`, `__init__.py`) |
| `img/` | Logo |
| `requirements.txt` | Python dependencies |
| `colab_notebook.ipynb` | Pre-configured Colab runner |
