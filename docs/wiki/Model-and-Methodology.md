# Model and methodology

The engine answers one question: **how much of a receipt's value can a lender safely advance, for a given tenor?** It answers it in five steps.

## 1. Price index (`index.py`)

Trade records become a daily volume-weighted average price in USD. Duplicates and outliers (keying errors, judged against a rolling median) are removed. ZiG trades are converted with an FX table, as of the trade date.

**Days without trades are left out, never filled.** Filling them with the last price would hide volatility. The gap between trading days is carried into the model instead.

## 2. Seasonality (`seasonal.py`)

Log price splits into a seasonal curve and a deviation:

```
ln P(t) = s(t) + y(t)
s(t)    = a0 + Σ [ a_k cos(2πkτ) + b_k sin(2πkτ) ],  k = 1, 2,  τ = fraction of the year
```

The harvest glut and the lean season live in `s(t)`. By default a seasonal fall over the loan's life counts against the lender, and a seasonal rise does not count for them.

## 3. Price deviation and regimes (`params.py`, `calibrate.py`)

The deviation `y` reverts to zero (Ornstein–Uhlenbeck), with volatility that switches between a **calm** and a **stressed** regime. Regimes follow a two-state Markov chain in continuous time.

Five parameters, time measured in days:

| Parameter | Meaning |
| --- | --- |
| `kappa` | Speed of reversion; half-life = ln 2 / kappa |
| `sigma1`, `sigma2` | Calm and stressed volatility (`sigma2 > sigma1` always) |
| `q12`, `q21` | Rates of switching into and out of stress |

Over a gap of Δ days in regime *i*, the deviation moves exactly as:

```
y(t+Δ) = y(t) e^(−κΔ) + σ_i √((1 − e^(−2κΔ)) / 2κ) · ε,   ε ~ N(0, 1)
```

Calibration is maximum likelihood with a Hamilton filter, which handles the irregular gaps directly. Standard errors come from the Hessian. One approximation is stated openly: the regime is held fixed within each gap.

## 4. Simulation (`engine.py`, `tallies.py`)

The simulation is organised like a **particle-transport Monte Carlo code**:

| Transport | Emkambo |
| --- | --- |
| Particle history | One price path from loan start to liquidation |
| Free flight to the next collision | Time to the next regime switch |
| Exact streaming between collisions | Exact OU transition within a regime |
| Source biasing and exponential transform | Biasing into stress and tilting shocks toward losses, with exact weights |
| Tally with relative error and figure of merit | Expected shortfall with batch error, variance of variance and FOM |

Because each path is propagated exactly between switches, there is no time-step error. The two variance-reduction techniques together cut computing time about 7.6 times for the same precision at 99%, with no measurable bias. The settings (bias 4, tilt 0.75) were validated in testing; too much tilt destroys the gain.

## 5. Haircut (`haircut.py`)

For horizon H = tenor + liquidation lag, and storage cost c per tonne per day:

```
L = 1 − (P_H − cH) / P_0          loss per unit of value
haircut = ES_99%(L)                expected shortfall: the average of the worst 1% of losses
```

With parameter uncertainty switched on, the model parameters are redrawn in each batch from their calibration covariance, so the haircut includes the risk of having estimated the model wrongly.

## The GMB floor for maize (`floor.py`)

The Grain Marketing Board buys maize at a floor producer price (US$364.75/t in 2025/26). A lender who can deliver pledged maize to the GMB has a safety net, but a weaker one than the headline price suggests:

```
V = F · [ (1 − z) + z (1 − g) D ] · e^(−r t) − delivery cost
```

where `z` is the share paid in ZiG, `g` the interbank-to-market conversion loss, `D` the ZiG depreciation over the payment delay `t`, and `r` the lender's funding rate. With probability `p_fail` the floor is not honoured at all.

The engine reports three haircuts: **floor holds**, **floor fails**, and **blended** at `p_fail`. The gap between holds and fails is how much of the lender's protection rests on GMB policy.

**At 99% confidence, a floor that can fail protects less than it seems.** If it fails 20% of the time, the worst 1% of outcomes are all failures, so the blended haircut is close to the no-floor haircut at 95% confidence. Only a floor believed to fail less than 1% of the time reduces the 99% haircut substantially.

## 6. Backtest (`backtest.py`)

The model is refitted on data up to each loan start and its loss limit is checked against what happened, on loan horizons that do not overlap. Two tests:

- **Kupiec**: do losses break the limit at the stated rate?
- **Christoffersen**: do the breaks cluster together?

Five years of non-overlapping loans give too few test points to check a 99% figure directly, so backtests run at 95%.

## Known limitations

- The deviation reverts to a fixed seasonal level. If USD prices trend over the sample, reversion is overstated and haircuts understated. Planned fix: anchor the level to JSE import parity (`basis.py`).
- No jump component for policy shocks such as export bans or producer-price announcements.
- Floor failure and ZiG depreciation are drawn independently of grain prices. In a crisis they tend to come together, so the floor-holds figure is optimistic in stress.
- No forced-sale discount yet; it needs lender recovery data.

## Further reading

- Execution strategy and acceptance tests: the Emkambo Stochastic Engine execution strategy document.
- Literature review: *Emkambo Literature Note: Collateral Haircuts for Physical Commodities*.

← [Home](Home.md)
