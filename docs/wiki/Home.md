# Welcome to Emkambo

**Emkambo prices the risk in grain that lenders already hold as collateral.**

Zimbabwean lenders advance money against warehouse receipts for stored maize and soya. If the price falls before the loan is repaid, the collateral no longer covers it. Lenders guard against this with a haircut: they lend less than the receipt is worth. Today those haircuts are set by convention, because there is no futures market to price the risk. The Zimbabwe Mercantile Exchange (ZMX) runs spot auctions only.

This engine computes the haircut from the price history itself, by commodity and loan tenor. Each figure comes with its Monte Carlo error, its calibration-uncertainty band and a backtest record.

Emkambo is a project of Mveli Labs (Pvt) Ltd, Bulawayo. The code is proprietary and confidential; see [LICENSE](../../LICENSE).

## Start here

| If you want to… | Read |
| --- | --- |
| Install the engine and run it in ten minutes | [Getting started](Getting-Started.md) |
| Understand what the model does and why | [Model and methodology](Model-and-Methodology.md) |
| Change tenors, confidence levels or the GMB floor | [Configuration](Configuration.md) |
| Know how we work together and handle data | [Working together](Working-Together.md) |

## What the engine produces

For each commodity and loan tenor:

- **Haircut and advance rate**: the expected shortfall of the collateral loss at 99%, over loan tenor plus liquidation lag, net of storage cost.
- **Uncertainty band**: the 5–95% range of the haircut, given how well the data pins down the model.
- **Tally checks**: relative error under 5% and a variance-of-variance check, as in a particle-transport code.
- **For maize, three haircuts**: if the GMB floor price holds, if it fails, and blended at the assumed failure probability, plus the gap between them.
- **A run manifest**: input-file hashes, settings and engine version, so every number can be reproduced.

## Where things stand

| Version | What it added |
| --- | --- |
| v0.1.0 | Regime-switching price model, calibration on irregular trades, event-based Monte Carlo with variance reduction, haircuts, backtesting |
| v0.2.0 | GMB floor-price model for maize |

All results so far come from **synthetic data**. They prove the machinery works; they say nothing about the real market. The first real milestone is a run on ZMX transaction data.

## Repository map

```
emkambo/            the engine (one module per job; see Model and methodology)
tests/              acceptance tests, run on every push to main
demo/               synthetic trade data and a config to try the engine
config.example.yaml every setting, with comments
docs/wiki/          these pages
```

## Who to ask

| Area | Owner |
| --- | --- |
| Engine code, simulation, data pipeline, partner relationships | Xolani Ndlovu, founder and architect |
| Risk measures, hedge design, regulatory scan, lender-facing methodology | Financial engineering lead |

Contact: Xolani Ndlovu · +263 77 141 8051 · xolanimart@gmail.com
