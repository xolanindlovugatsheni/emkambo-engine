# Getting started

Ten minutes from clone to your first haircut table, on synthetic data.

## 1. Install

You need Python 3.10 or later and access to this private repository.

```bash
git clone https://github.com/xolanindlovugatsheni/emkambo-engine
cd emkambo-engine
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[test]"
```

## 2. Check the installation

```bash
pytest -q -m "not slow"      # about 30 seconds
```

Every test should pass. If one fails, stop and report it before running anything else: the tests are how we know the numbers can be trusted.

## 3. Run the demo

The `demo/` folder holds six years of synthetic maize trades, generated from known parameters.

```bash
cd demo
emkambo run --config config.yaml
```

You will see a table like this (your figures should match exactly, because the seed is fixed):

```
 tenor_days  haircut  advance_rate  es_rel_err  es_band_lo  es_band_hi  tally_checks_pass
         30   ...        ...           ...         ...         ...            True
```

How to read it:

| Column | Meaning |
| --- | --- |
| `haircut` | Share of today's value to hold back |
| `advance_rate` | 1 − haircut: the safe loan per unit of collateral value |
| `es_rel_err` | Monte Carlo relative error of the estimate; should be under 0.05 |
| `es_band_lo`, `es_band_hi` | Range of the haircut across plausible model parameters |
| `tally_checks_pass` | Whether the estimate has converged |

The run also writes a folder under `demo/runs/` with `index.csv`, `haircuts.csv` and `run.json`. Never edit these by hand: `run.json` is the audit record.

The demo haircuts are high because the synthetic market is deliberately volatile. They test the machinery, not the Zimbabwean market.

## 4. Try the backtest

```bash
emkambo backtest --config config.yaml
```

This refits the model on data up to each loan start, predicts the loss limit, and checks how often real losses broke it. A Kupiec p-value above 0.05 means the exceedance rate is consistent with the stated confidence level.

## 5. Make your own synthetic market

```bash
emkambo synth --out mydata/trades.csv --truth-out mydata/truth.csv --years 8 --seed 3
```

The truth file records the hidden regime and price deviation each day, so you can check what the calibration recovers.

## 6. Run on real data

1. Copy `config.example.yaml` to `config.yaml` in a working folder **outside the repository**, or in a `data/` folder (both are ignored by git).
2. Put trade records in the format `date, commodity, quantity_t, price, currency`. ZiG trades need an FX file: `date, zig_per_usd`.
3. Set the paths, commodity and costs; see [Configuration](Configuration.md).
4. Run `emkambo run --config config.yaml`.

Real ZMX or bank data must never be committed. See [Working together](Working-Together.md).

## Docker

```bash
docker build -t emkambo .
docker run --rm -v "$PWD/work:/work" emkambo run --config config.yaml
```

← [Home](Home.md)
