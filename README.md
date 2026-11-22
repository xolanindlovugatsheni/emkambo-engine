# Emkambo stochastic engine

Collateral haircut engine for warehouse-receipt lending. Mveli Labs (Pvt) Ltd, Project Emkambo.

For a commodity and a loan tenor, the engine returns the **haircut**: the expected shortfall, at level α, of the collateral loss over tenor plus liquidation lag. Every haircut ships with its Monte Carlo relative error, the calibration-uncertainty band, and a reproducible run manifest.

## Model, in one screen

- **Index**: daily VWAP in USD from trade-level exchange records. No-trade days are left out; gap lengths enter the model.
- **Seasonality**: `ln P = s(t) + y`, with `s` a constant plus two annual Fourier harmonics.
- **Deviation `y`**: Ornstein–Uhlenbeck with shared reversion `κ` and regime volatility `σ₁ < σ₂`; regimes follow a two-state continuous-time Markov chain (rates `q₁₂`, `q₂₁`). Time unit is the day.
- **Calibration**: Hamilton-filter MLE on irregular gaps, `P(Δ) = exp(QΔ)`; covariance from the numerical Hessian. Approximation: regime held fixed within each gap.
- **Simulation**: event-based, as in particle transport. Sample the time to the next regime switch, stream `y` exactly to it, switch, repeat. No time-step error.
- **Variance reduction** (exact weights): regime biasing into stress, plus shock tilting toward losses.
- **Haircut**: `L = 1 − (P_H − cH)/P₀`, haircut `= clip(ES_α(L), 0, 1)`. With `param_uncertainty: true`, parameters are redrawn per batch from the calibration covariance, and the haircut is the pooled (predictive) ES.
- **GMB floor (maize, v0.2)**: the floor producer price is treated as a put that can fail. Its effective value is the headline price less ZiG conversion loss, ZiG depreciation during the payment delay, funding cost and delivery cost. With probability `p_fail` the GMB does not buy or pay. Three haircuts are reported: floor holds, floor fails, and blended at `p_fail` (the headline figure). `floor_dependence` = fails − holds is how much of the lender's protection rests on GMB policy. Enable it under `haircut.floor` in the config. Every floor input except the price is a placeholder assumption until confirmed with lenders.

## Install and run

```bash
pip install .                       # Python 3.10+
emkambo synth --out data/trades.csv --truth-out data/truth.csv   # synthetic truth set
cp config.example.yaml config.yaml  # edit paths, tenors, storage cost
emkambo run --config config.yaml    # index -> calibrate -> haircut table
emkambo backtest --config config.yaml
```

Each run writes a folder under `runs/` containing `index.csv`, `haircuts.csv` (or `backtest_records.csv`) and `run.json`. The manifest records the config, the input-file SHA-256 hashes, the engine and numpy versions, the calibration, and every result.

Docker:

```bash
docker build -t emkambo .
docker run --rm -v "$PWD/work:/work" emkambo run --config config.yaml
```

## Input format

`trades.csv`: `date, commodity, quantity_t, price, currency` (USD, ZiG or ZWG). ZiG trades need `fx.csv`: `date, zig_per_usd`, applied as-of the trade date. Without it they are dropped and counted in the report.

## Tests

```bash
pip install ".[test]"
pytest -q                 # all tests, about 1 minute
pytest -q -m "not slow"   # skip the rolling backtest
```

| Test | Checks |
| --- | --- |
| `test_ou_transition_moments` | Exact streaming reproduces closed-form OU mean and variance |
| `test_weights_unbiased` | Mean importance weight equals 1 for each variance-reduction mode |
| `test_variance_reduction_unbiased_and_pays` | Biased ES equals analogue ES, and FOM gain is at least 3× |
| `test_parameter_recovery` | Calibration recovers true parameters within 3 standard errors |
| `test_seed_reproducibility` | Identical seed gives identical haircuts |
| `test_tally_quality` | ES relative error under 5% and VOV check pass |
| `test_kupiec_on_synthetic` | VaR exceedance rate is consistent with 1 − α (12-year synthetic) |
| `test_index.py` | Index edge cases: single trade, ZiG without FX, duplicates, outliers |
| `test_floor.py` | Floor value matches closed form; floor-fails equals no-floor exactly; haircut rises with `p_fail`; a certain USD floor caps the loss |

## Module map

| Module | Role |
| --- | --- |
| `index.py` | Trade records to daily index |
| `seasonal.py` | Fourier seasonal curve |
| `calibrate.py` | Hamilton filter and MLE |
| `engine.py` | Event-based transport-style Monte Carlo |
| `tallies.py` | Weighted VaR/ES, batch error, VOV, FOM |
| `haircut.py` | Haircut with error and uncertainty band |
| `backtest.py` | Rolling backtest, Kupiec and Christoffersen tests |
| `basis.py` | Local vs JSE basis OU fit, minimum-variance hedge ratio |
| `floor.py` | GMB floor price as a failable put |
| `synthetic.py` | Truth-set generator |
| `pipeline.py`, `cli.py` | Config-driven runs and manifests |

## Known limitations

- The deviation reverts to a fixed seasonal level. If USD prices trend over the sample, `κ` will be overstated. The next upgrade anchors the level to JSE import parity via `basis.py`.
- No jump component for policy shocks such as export bans or price controls.
- Floor failure and ZiG depreciation are drawn independently of the grain price. In a real crisis they tend to arrive together, so the floor-holds haircut is optimistic in stress.
- The backtest tests the price model only; it ignores the floor.
- Liquidation lag and storage cost are inputs; replace them with bank recovery data and ZMX tariffs.
- Synthetic-data haircuts mean nothing commercially. They exist only to prove the machinery.
