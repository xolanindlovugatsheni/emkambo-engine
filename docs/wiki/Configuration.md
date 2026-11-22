# Configuration

Every run is driven by one YAML file. Start from `config.example.yaml` and copy it to a working folder. Paths inside it are relative to the file.

## `data`

| Key | Meaning |
| --- | --- |
| `commodity` | Commodity name as it appears in the trade file, e.g. `maize` |
| `trades` | Trade-level CSV: `date, commodity, quantity_t, price, currency` |
| `fx` | Optional CSV `date, zig_per_usd`; without it ZiG trades are dropped and counted |

## `model`

| Key | Default | Meaning |
| --- | --- | --- |
| `harmonics` | 2 | Annual Fourier terms in the seasonal curve |

## `haircut`

| Key | Default | Meaning |
| --- | --- | --- |
| `alpha` | 0.99 | Expected-shortfall confidence level |
| `tenors_days` | [30, 60, 90, 180] | Loan tenors in the output table |
| `liquidation_lag_days` | 21 | Days from default to sale; **replace with lender data** |
| `storage_usd_per_t_day` | 0.10 | Warehouse charge; **replace with ZMX tariffs** |
| `seasonal_mode` | `adverse` | `full`, `adverse` (count only falls) or `none` |

### `haircut.floor` (maize only)

Set `enabled: true` to model the GMB floor. Keep it off for soya.

| Key | Example | Status |
| --- | --- | --- |
| `price_usd_per_t` | 364.75 | 2025/26 floor producer price |
| `zig_share` | 0.5 | Assumption: confirm with GMB and lenders |
| `zig_market_discount` | 0.10 | Assumption |
| `zig_depreciation_annual` | 0.20 | Assumption |
| `zig_vol_annual` | 0.25 | Assumption |
| `payment_delay_days` | 45 | Assumption |
| `funding_rate_annual` | 0.15 | Assumption |
| `delivery_cost_usd_per_t` | 10.0 | Assumption |
| `p_fail` | 0.20 | Assumption: run a range, never a single guess |

Every floor input except the price is a placeholder. Report results across several `p_fail` values until lenders tell us how they treat the floor.

## `monte_carlo`

| Key | Default | Meaning |
| --- | --- | --- |
| `batches` | 40 | Independent batches for error estimates |
| `paths_per_batch` | 25000 | Price paths per batch |
| `bias` | 4.0 | Regime biasing into stress (1 = off) |
| `tilt` | 0.75 | Shock tilting toward losses (0 = off) |
| `param_uncertainty` | true | Carry calibration uncertainty into the haircut |
| `seed` | 20261018 | Fixes every random draw; same seed, same answer |

Change `bias` or `tilt` only with a test showing the figure of merit improves and the answer does not move.

## `backtest`

| Key | Default | Meaning |
| --- | --- | --- |
| `alpha` | 0.95 | Level tested; lower than the haircut level so short histories have enough breaks to count |
| `tenor_days` | 60 | Loan tenor tested |
| `min_train_days` | 730 | History required before the first test loan |
| `recal_every` | 4 | Refit the model every this many test loans |
| `paths` | 40000 | Paths per test loan |

← [Home](Home.md)
