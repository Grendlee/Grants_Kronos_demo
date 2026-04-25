# Grant's Kronos Demo

Probabilistic BTC/USD and ETH/USD forecast based on the [Kronos](https://github.com/shiyu-coder/Kronos) foundation model. Adapted from [shiyu-coder/Kronos-demo](https://github.com/shiyu-coder/Kronos-demo).

## What it does

- Fetches the last 30 days (720h) of BTC/USD and ETH/USD 10-min K-line data from Binance.US (resampled from 5-min — Binance has no native 10m interval)
- Runs Kronos-Base with 30 Monte Carlo samples to produce a distribution of 24-hour forecasts
- Publishes a static dashboard with: mean forecast line, min–max uncertainty band, upside probability, and volatility amplification probability

## Architecture

```
Kaggle Kernel (free T4 GPU, scheduled every 12h)
   ↓ git clone + run update_predictions.py
   ↓ git push refreshed PNGs + index.html
GitHub repo (main branch)
   ↓ GitHub Pages auto-deploy
Public dashboard URL
```

## Running locally

```bash
pip install -r requirements.txt
python update_predictions.py
```

Outputs: `prediction_chart_btcusd.png`, `prediction_chart_ethusd.png`, and updated metrics/timestamp in `index.html`. Open `index.html` in a browser.

Kronos-Base is a 102M-param model. Per-symbol runtime by device:

| Device | Time per symbol |
|---|---|
| CUDA (T4) | ~3 min |
| Apple MPS | ~5–7 min |
| CPU | ~60+ min |

## Production refresh — Kaggle Kernel (free T4 GPU)

This is the deployed setup. Inference runs on Kaggle, results push back to this repo, GitHub Pages serves them.

> **Note on GPU choice:** Kaggle still offers P100 in the menu, but as of 2026 their PyTorch image dropped support for the P100's older compute capability (`sm_60`). Use **T4 x2** — `sm_75`, fully supported, same throughput on this workload.

**One-time setup:**

1. **Create a GitHub Personal Access Token**
   - GitHub → Settings → Developer settings → Personal access tokens → Fine-grained
   - Repository access: only this repo
   - Permissions: `Contents: Read and write`
   - Copy the `github_pat_...` value (one-shot — GitHub won't show it again)

2. **Create a HuggingFace read token** *(optional but recommended)*
   - https://huggingface.co/settings/tokens → New token → Read role
   - Avoids "unauthenticated requests" rate-limit warnings on model download

3. **Create the Kaggle notebook**
   - Sign in at https://www.kaggle.com (verify your phone if you haven't — required for free GPU and Secrets)
   - **+ New Notebook** → **File → Import Notebook** → **GitHub** tab → paste this repo's `kaggle_runner.ipynb` raw URL

4. **Configure the notebook**
   - Right sidebar → Session options:
     - Accelerator: **GPU T4 x2**
     - Internet: **On**
     - Persistence: Off
   - Top menu → Add-ons → Secrets:
     - `GITHUB_TOKEN` = the PAT from step 1 (required)
     - `HF_TOKEN` = the HF token from step 2 (optional)
     - Toggle "Attached" on for both

5. **First run** — click `Save & Run All` to verify it pushes a refreshed forecast to the repo.

6. **Schedule recurring runs**
   - From the notebook viewer (after a successful run), `⋯` menu → `Schedule a notebook run`
   - Cadence: every 12 hours

Each scheduled run takes ~5–8 min on the T4. Free quota is 30 GPU-hours/week — plenty of headroom.

## Deploying GitHub Pages

1. Push the repo to GitHub
2. Repo → Settings → Pages → Source: `main` branch, `/ (root)`
3. Live at `https://<username>.github.io/<repo-name>/` after ~30s

## Manual / local refresh options

- **Run on your own machine:** `python update_predictions.py`. Then `git add` + commit + push the regenerated PNGs and `index.html`.
- **Kaggle one-shot:** open the notebook on Kaggle and click `Save & Run All` — same as the scheduled path, just on demand.
- **GitHub Actions fallback (CPU, slow):** `.github/workflows/refresh.yml` is preserved as a manual fallback (`workflow_dispatch` only — the 12-hour cron was removed because each CPU run exceeded the 60-min job timeout).

## Spec notes (for reviewers)

The original spec asked for "the last 720 hours (~30 days) of BTC/ETH 10-min K-line data … as context for each new prediction." This is internally inconsistent with Kronos-Base's architecture. The spec is met as follows:

| Spec | Implemented | Deviation |
|---|---|---|
| Kronos-Base | ✅ Kronos-Base | — |
| BTC + ETH | ✅ BTCUSD + ETHUSD on Binance.US | Binance.com geo-blocks US IPs, including Kaggle's data centers — switched to Binance.US |
| 10-min candles | ✅ resampled from 5-min | Binance has no native 10m interval (returns `{"code":-1120,"msg":"Invalid interval"}`) |
| **720h of context** | **30-day history displayed; 85h fed to model** | Kronos-Base's `max_context = 512` candles. 720h × 6 candles/hr = 4,320 candles, 8× over the limit. Architecturally impossible to feed in a single forward pass. The chart shows the full 30 days; the model's input window is the last 512 candles |
| N=30 Monte Carlo paths | ✅ | — |
| Free GPU compute | ✅ Kaggle T4 x2 | — |
| GitHub Pages hosting | ✅ | — |
| Mean forecast + uncertainty band + probability metrics | ✅ | — |

## Files

| File | Purpose |
|---|---|
| `update_predictions.py` | Main script: fetch → predict → metrics → chart → HTML |
| `kaggle_runner.ipynb` | Notebook that runs on Kaggle: clones repo, runs the script, pushes back |
| `index.html` | Static dashboard page (served by GitHub Pages) |
| `style.css` | Page styling |
| `model/` | Kronos model code (`kronos.py`, `module.py`, `__init__.py`) |
| `img/` | Logo |
| `requirements.txt` | Python dependencies |
| `.github/workflows/refresh.yml` | Manual GitHub Actions fallback (CPU, no schedule) |
| `colab_notebook.ipynb` | Older Colab runner — kept as alternative manual option |
