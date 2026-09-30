# DESIGN.md: decisions, formulas and acceptance values

Values marked (verified) were reproduced independently in Python (NumPy 2.4, SciPy 1.17) before the build. Tests must assert them at the stated tolerances.

## 1. Build order

| Stage | Deliverable | Tag |
|---|---|---|
| 0 | Scaffold: layout, pinned requirements, README skeleton, smoke test | stage-0 |
| 1a | History: SPY, ^VIX, ^IRX frozen with manifest | stage-1a |
| 1b | Chain collector script, run on 3 to 5 market days | stage-1b |
| 2a | Black-76 pricing and Greeks | stage-2a |
| 5a | Hedging engine and synthetic theory tests | stage-5a |
| 5b | Real-data hedging study: Figures 6 to 8, Table 4 | stage-5b |
| 5c | OptionMetrics sensitivity: σ_i = ATM implied vol, VIX gap, Figure 7b, Table 4d | stage-5c |
| 2b | Chain cleaning, parity forwards, filters: Table 1 | stage-2b |
| 2c | Implied vol solver, no-IV filter, Newton demo: Figure 1 | stage-2c |
| 3 | SVI calibration: Figure 2, Table 2 | stage-3 |
| 4 | Arbitrage checks, constrained refit, variance-swap bridge: Figures 3 to 5, Table 3 | stage-4 |
| 6 | Write-up and fresh-install reproducibility check | v1.0 |

The hedging track goes first because it needs only pricing and historical data.

## 2. Pricing (black_scholes.py)
- Core is Black-76 in forward form: call = D·[F·N(d1) - K·N(d2)], put = D·[K·N(-d2) - F·N(-d1)], with d1 = [ln(F/K) + σ²T/2]/(σ√T) and d2 = d1 - σ√T.
- Spot wrapper: F = S·exp((r - q)T), D = exp(-rT).
- Greeks: spot delta, gamma, vega, theta (per year), vectorised for calls and puts.
- Identity to test: Γ·S²·σ·T = vega (exact, including q).
- Acceptance (verified): S = 100, σ = 0.18, T = 21/252, r = q = 0, K = F gives C0 = 2.072732 and vega = 11.512585 (so vega·σ = 2.072265, which is why ATM premium ≈ vega·σ).

## 3. Stage 5 study (hedging.py)

Conventions
- Prices: yfinance closes with auto_adjust=True (a total-return series), so q = 0.
- σ_i = ^VIX close at window start / 100, held fixed through the window.
- ^IRX quotes the 13-week bill discount yield: d_t = ^IRX close / 100, forward-filled, converted at load time (data.irx_to_rate) to the continuously compounded r_t = -ln(1 - d_t·91/360)/(91/365); d = 0.0535 gives 0.0546. The frozen file keeps d, so no re-download is needed. Cash accrues at r_t over calendar days.
- Windows: one per trading day t0 from the first trading day of October 2021 to the last start date with a full window. t_end = last trading day on or before t0 + 30 calendar days. T0 = (t_end - t0 in calendar days)/365.
- Option: European call, K = F0 = S0·exp(r0·T0), bought at its Black-Scholes value with σ_i, expiring at the t_end close.
- Hedge: short Δ (computed at σ_i and remaining calendar T) at each rebalance close; unwind at t_end.
- Rebalance every h trading days, h in {1, 2, 3, 5, 7, 10, 21}, counted back from expiry: the hedge is set at t0 and at the closes h, 2h, ... trading days before t_end. Any stub shorter than h therefore opens the window, where gamma is low, and every window meets expiry with full intervals. N = ceil(n/h) hedge intervals for a window of n trading-day steps.
- Costs: c in {0, 1, 5, 25, 100} bp of notional traded, on every stock trade including the opening and closing trades. SPY's one-cent spread is about 0.2 bp, so this grid is a stress test.
- Money units: P&L and predictors are in expiry money, as fractions of C0·G0, the premium carried to t_end, where G_j = exp(Σ_{i≥j} r_i·Δt_i) is the growth of cash from close j to t_end at the financing rates. Each predictor term is weighted by G at the rebalance close where its interval starts, which makes Σ G_k·Γ_k·S_k²·Δt_k an exact path version of G0·vega0/σ_i under Q (E[e^{-rt}·Γ_t·S_t²] = Γ0·S0²). Check: P&L and P_path/P_gap are unchanged to 1e-10 when a flat r = 0.05 is added in forward terms, which fails if P&L is divided by C0 alone.
- Engine inputs: a window of n steps has S and t of shape (n + 1,) and step rates r of shape (n,) (r_j prices and hedges at close j and accrues cash from j to j + 1), or a scalar r. P windows of one length stack as (P, n + 1) and (P, n), each with its own calendar and rates, with σ_i, K and q scalar or (P,). Any other shape raises. Real windows differ in n, so simulate_windows groups them by n and calls the engine once per group.
- Per-interval output: hedge interval k, from grid close a to grid close b, earns G_b·V_b - G_a·V_a - Δ_a·(G_b·S_b - G_a·S_a) + G_a·cost_a in expiry money, where V is the Black-Scholes value at σ_i, r_a and the remaining calendar T (V at t_end is the payoff) and cost_a ≤ 0 is the cost of the trade at a; the last interval also carries the unwind cost at t_end. Cash adds nothing in expiry money, and the terms telescope, so the intervals sum to the window P&L exactly. simulate_windows returns them as (W, N_max) arrays aligned at expiry, NaN before a shorter window's first interval.
- Puts: the engine also prices and hedges puts. Check: at q = 0 (the Stage 5b configuration) and a flat r = 0.04, a hedged call and a hedged put with the same K give identical P&L path by path to 1e-10, because call minus put is a forward whose delta is exactly one share. Parity is not tested at q > 0: the total-return path pays no dividend while the model assumes one on the e^{-qτ} shares, so the two differ by that missing dividend, the same carry effect the section 4 carry check measures.
- Realised variance: RV = Σ [ln(S_{j+1}/S_j)]² over daily closes in the window, no demeaning, no N - 1. σ_r² = RV/T0. Compare total variances: σ_i²·T0 against RV.

Predictors (the R² ladder), computed on the same rebalancing grid as the actual P&L
- P_gap = vega0·(σ_r² - σ_i²)/(2σ_i)
- P_price = BS(σ_r) - BS(σ_i): the option at t0 (same S0, K, T0, r0 and q) priced at the realised vol σ_r, minus the premium, in expiry money G0·(BS(σ_r) - C0), so BS(σ_r)/C0 - 1 as a fraction of C0·G0. Priced at the true vol of a GBM path with risk-neutral drift, it is the exact expectation of the hedged P&L. Itô's lemma on V(σ_i) under σ_r dynamics makes the continuously hedged P&L ∫ e^{-rt}·½·Γ_i·S²·(σ_r² - σ_i²) dt, whose expectation is BS(σ_r) - BS(σ_i) (Feynman-Kac for V(σ_r) - V(σ_i)). Discrete hedging leaves the expectation unchanged, because the discounted stock gains have zero mean under Q whatever delta is held, so E[P&L] = E[e^{-rT}·payoff] - C0 exactly. This holds at a flat rate with q = 0 and c = 0. The ladder evaluates P_price at each window's realised σ_r, as it does P_gap.
- P_path = ½·(σ_r² - σ_i²)·Σ_j Γ_j·S_j²·Δt_j
- P_step = Σ_j ½·Γ_j·S_j²·(R_j² - σ_i²·Δt_j), with R_j the simple return over hedge interval j
- For each predictor report R²_45 = 1 - Σ(y - p)²/Σ(y - ȳ)² (about the 45 degree line), the plain OLS R² (the squared correlation of y and p) and OLS y = α + β·p. R²_45 charges any intercept or slope away from 1, so it mixes calibration with explanatory power; the OLS R² measures linear explanatory power alone.
- Theory: α = 0 and β = 1 hold exactly in expectation for P_price under GBM, and to first order for P_path and P_step. They do not hold for P_gap, even on GBM paths. P_gap = vega0·(σ_r - σ_i)·(σ_r + σ_i)/(2σ_i) exceeds its linear version vega0·(σ_r - σ_i), the first-order term of P_price in σ, by the factor (σ_r + σ_i)/(2σ_i). An ATM option's vega is nearly flat in σ, so the expected P&L is close to linear in σ_r while P_gap is quadratic: P_gap overstates gains when σ_r > σ_i (by the factor 1.25 at σ_r = 1.5·σ_i) and understates losses when σ_r < σ_i. The mechanism is gamma decay: a high-vol path leaves the strike and its gamma falls, which P_gap, built on the opening gamma, ignores. The P&L is therefore a concave function of P_gap, so even under GBM the OLS slope of P&L on P_gap is not 1, and its value depends on the mix of windows. It is below 1 when windows with σ_r far above σ_i carry most of P_gap's variance, where the overstatement is largest, as in the full real-calendar benchmark that the headline cites. It can exceed 1 when those windows are absent, because P_gap then mainly understates losses, as in the April-excluded benchmark. P_price has neither bias.
- Confidence intervals for R²_45 and β come from a moving block bootstrap over windows in start-date order, block length 42 windows (about two window lengths, so each block spans the overlap), alongside the non-overlapping subsamples. This replaces Newey-West errors. 2,000 resamples, seed 20260930: each resample joins blocks of 42 consecutive windows whose first windows are drawn uniformly, with replacement, from all n - 41 positions, truncated to n windows; the intervals are the 2.5% and 97.5% percentiles. The same resampled windows serve every predictor and every h; the April-excluded sample is resampled the same way on its own sequence.
- Block-length sensitivity: the headline intervals (h = 1, R²_45 and β on the full and April-excluded samples) and the Stage 5c precision of the mean are also computed with blocks of 84 and 126 windows, each block length on its own 2,000 resamples with seed 20260930 (block 42 reproduces the resamples above). Vol regimes (2022, 2023 to 2024, April 2025) last far longer than 42 trading days, so a 42-window block can cut through one and understate the uncertainty. Longer blocks keep more of a regime inside one block at the cost of fewer blocks per resample: 15 and 10 at 1,232 windows, against 30 at block 42.
- The ladder is reported at c = 0 for every h: the predictors are frictionless, and costs only add a mean shift that R²_45 would charge to all three alike.
- The headline R² is P_gap's, because the research question asks about the vol gap. The ladder attributes the shortfall: functional form, that is the expected gamma decay that P_gap ignores (P_price vs P_gap, the same vol gap priced exactly), the path's own moneyness drift against that expectation (P_path vs P_price), timing of moves against gamma (P_step vs P_path), discreteness and jumps (actual vs P_step).
- Benchmark: even on pure GBM windows (the synthetic set below, daily hedging), P_gap's R²_45 is only about 0.8 (0.77 below), because the vol gap alone ignores where the path spends its gamma. Real data is judged against this benchmark, not against 1.
- Sanity ranges on synthetic GBM windows (verified; 3,000 windows, σ_i ~ lognormal(ln 0.17, 0.35), σ_true = σ_i·exp(N(-0.2, 0.3))): daily R²_45 = 0.77 / 0.93 / 0.98 for P_gap / P_path / P_step; every 5 days 0.41 / 0.45 / 0.89 (median of 40 seed pairs). Expiry anchoring puts a full 5-day interval where gamma peaks, so the quadratic approximation is worse; the start-anchored grid's 0.40 / 0.46 / 0.92 was flattered by its 1-day stub before expiry. Real data should sit lower on P_step because of jumps.
- These ranges are illustrative and seed-dependent, so no test asserts them (section 4 tests only P_step R²_45 ≥ 0.95). Across 200 seed pairs the daily P_step median is 0.976, and a single large-move window can pull it down.
- Real-calendar GBM benchmark (the one the headline cites): every real window gets 100 GBM paths from 100 on its own closes, with σ_true equal to its realised vol √(RV/T0), σ_i equal to its VIX, drift and financing at its real step rates (the risk-neutral drift of a total-return series), K = F0 and c = 0. The ladder is computed over all paths for every h, with P_gap from each path's own realised variance. Windows of n steps draw their normals from np.random.default_rng([20261001, n]). The ladder is also reported over the paths of the windows that the April 2025 exclusion keeps, so the April-excluded real ladder has a like-for-like benchmark. The section 3 synthetic set above is printed separately as the textbook reference only: median over 40 seed pairs (20261100 + k, 20261200 + k), k = 0 to 39, for h = 1 and 5.
- Weekday diagnostic (daily hedging, c = 0): P_step residuals, the hedged P&L of each interval (the per-interval output above) minus its P_step term, averaged by the weekday of the interval's closing day. Δt counts calendar days, so a Friday-to-Monday interval charges three days of σ_i² against roughly one trading day of realised variance; the diagnostic shows how much of the residual this weekend theta effect explains. Per weekday: interval count, mean calendar days, mean interval P&L, mean P_step term, mean residual with a moving block bootstrap interval (the full-sample resamples), and the weekday's share of the total residual; plus η², the share of residual variance explained by the weekday means.
- Weekend variance ratio: ρ = mean squared Friday-to-Monday log return / mean squared single-weekday log return, over the daily returns from the first window's t0 to the last window's t_end. A Friday-to-Monday return joins a Friday close to the Monday close three calendar days later, a single-weekday return joins two closes one calendar day apart, and returns across holidays enter neither mean. Interval: moving block bootstrap over the returns in date order, block 21 returns, 2,000 resamples, seed 20260930, with both means recomputed on each resample. Calendar-time pricing, the clock of VIX and of the engine's T, charges a weekend 3 weekday units of σ_i²; a trading-day clock charges 1. Report ρ, its interval and whether 3 lies inside it, and next to it the realised shortfall 1 - mean RV/mean(σ_i²·T0) of the windows.
- How the ratio separates the clock convention from a weekend premium. ρ is a property of SPY returns, so it is the same whatever σ_i. The clock decides only when a window's P&L is booked: with ρ below 3, calendar-time marks book losses on Monday intervals against gains on the other weekdays, while the window total is fixed by C0 and the payoff, because the marks in between telescope. The total follows the realised shortfall instead, which the notebook prints for σ_i = VIX and σ_i = ATM vol on the Stage 5c windows: one ρ next to two different shortfalls shows that the weekend effect moves the timing of the P&L, not its total. A weekend premium in the sense of Jones and Shemesh (2018), option prices that fall over weekends by more than the weekend's variance justifies, is a statement about market option prices inside the window. This study does not observe those (σ_i is fixed at t0 and the marks are model values), so the Monday concentration in the weekday diagnostic is not evidence of such a premium. ρ is not used to size the window total, and no weekend share of the total is derived from it.

Hedging error and Figure 8
- e = P&L - P_gap per window. For each (h, c) report mean, std and RMSE = √(mean² + std²), with the population std (ddof 0), so RMSE² = mean(e²).
- Figure 8: RMSE against N, one line per cost level, plus a panel with mean and std; overlay the synthetic GBM curve (σ_true = σ_i) as a reference. N varies with window length, so the x-axis is the mean N per h across windows, with the range (minimum to maximum) shown.
- The synthetic reference computes P_gap from each synthetic path's own realised variance, the same definition as on real data, so the curves are comparable. It is the real-calendar GBM construction of the benchmark above with σ_true = σ_i, at c = 0, on the same normals, so the two synthetic sets differ only in σ_true. Also draw the known-vol Derman-Kamal curve (P_gap = 0), √(π/4)·mean(vega0·σ_i/C0)/√N, as a labelled dashed line.
- Expected pattern (verified, synthetic, and still true on the expiry-anchored grid). It refers to raw synthetic P&L with σ_true = σ_i; real data reports the same statistics for e = P&L - P_gap. Costs move the mean, not the spread. Daily std of P&L (the P_gap = 0 case) at 0 / 1 / 5 bp was 18.6% / 18.6% / 18.4% of premium. On RMSE, daily beats every-2-days and every-5-days up to 5 bp, ties every-2-days near 25 bp, and loses to every-5-days at 100 bp.

Robustness
- Exclude every window whose closes t0 to t_end include any trading day from 2025-04-03 to 2025-04-09 and rerun the ladder.
- Non-overlapping subsamples at a fixed stride of 22: subsample k holds windows k, k + 22, k + 44, ... for each offset k = 0 to 21, so the 22 subsamples partition all windows (about 56 windows each). No window has more than 22 steps, so each window of a subsample starts at or after the previous one's end close and no two share a daily return; the code checks this. A stride of 21 would share a step wherever a window has 22 steps, and a greedy rule (each next window the first to start at or after the previous end) was rejected because its chains lock onto Friday-to-Friday windows and merge, leaving few distinct subsamples. Report the point estimates for offset 0, plus the median and the minimum to maximum of R²_45, the OLS R² and β across all 22. These subsamples are not bootstrapped. They interleave the same months and the same vol regimes, so their median and range measure sensitivity to window alignment (the trading days on which the windows start), not a sampling distribution, and they are not confidence intervals.
- Report the share of windows with σ_i²·T0 > RV once; it does not depend on h or c.
- Sensitivity: rerun with σ_i = VIX minus the variance-swap gap measured in section 8.

OptionMetrics sensitivity (Stage 5c)
- Data: two licensed WRDS OptionMetrics files in data/raw/. om_spy_std_30d_2021_2025.csv holds the standardised 30-day ATM-forward SPY call and put, 2021-06-01 to 2025-08-29. om_spy_volsurf_2021_2025.csv holds the volatility surface (34 deltas by 11 maturities), of which only days = 30 is read. data.load_optionmetrics returns, per date, sigma_atm = the mean of the call and put impl_volatility (NaN if either is missing) and the 30-day surface IVs at delta -25 (put), 25 (call) and 50 (call). It returns None if either file is absent and never downloads anything.
- Licensing (binding): WRDS data is never committed and never printed row by row. Notebook outputs, figures and README may contain only aggregates and charts, and no single day's value: no minimum, maximum or other value that one day determines. A daily series is summarised by its mean, median, std (ddof 0) and 5th and 95th percentiles, a statement that holds on every date is printed as a computed boolean, and charts show monthly means. tests/test_notebook.py enforces this when the files are present: it fails if any number printed with 5 or more decimals in the committed outputs equals a daily OptionMetrics IV to 1e-6, as printed or divided by 100. The Stage 5c notebook section prints one line and skips when the files are absent, so a clean install still runs top to bottom.
- Rerun: the section 3 windows that start on or before 2025-08-29, the last OptionMetrics date, hedged daily (h = 1) at c = 0 with K = F0 as before. Run once with σ_i = sigma_atm at t0 and once with σ_i = VIX on exactly the same windows, so the two are directly comparable.
- Reported for both: window count, mean σ_i, the share of windows with σ_i²·T0 > RV, and the mean and std (ddof 0) of hedged P&L. Then the R² ladder with moving block bootstrap intervals (block 42, 2,000 resamples, seed 20260930, drawn on this shorter sequence of windows, the same resamples for both σ_i), the April 2025 exclusion (resampled on its own sequence), and the stride-22 subsamples of these windows (offset 0, median, and minimum to maximum, a window-alignment sensitivity as above).
- VIX gap: VIX - sigma_atm on every OptionMetrics date, reported as count, mean, median, std and 5th and 95th percentiles in vol points, plus whether VIX is above sigma_atm on every date (a computed boolean). The 30-day risk reversal RR = IV(25-delta put) - IV(25-delta call) gets the same summary. OLS of the gap on RR, reporting slope, intercept and R².
- The slope and R² get moving block bootstrap intervals on the daily series: block 21 trading days, 2,000 resamples, seed 20260930, the OLS refitted on each resample, 2.5% and 97.5% percentiles. Daily observations need this because they are not independent. Consecutive days' 30-day vols price options whose lives overlap by all but one day, and both the gap and RR are persistent level series. OLS standard errors or an i.i.d. bootstrap would treat about a thousand dependent days as independent draws and give intervals that are too narrow. A block of 21 trading days (about one option month) keeps that dependence inside each block. The notebook prints the lag-1 autocorrelation of both series as evidence.
- Precision of the mean: 95% moving block bootstrap intervals for the mean hedged P&L under each σ_i and for the paired per-window difference, VIX minus ATM, on the same windows. They use the ladder's resamples (block 42, 2,000 resamples, seed 20260930), and are repeated with blocks of 84 and 126 windows as in the block-length sensitivity above. Each run's P&L is a fraction of its own premium, so the difference is in points of premium, not money. Also the median and range of mean P&L across the 22 stride subsamples, as a window-alignment sensitivity.
- Scaling diagnostic: as a fraction of the premium, P_gap is about (σ_r² - σ_i²)/(2σ_i²), because the ATM premium is close to vega0·σ_i, so the same variance gap weighs more when σ_i is lower. The ladder is recomputed with P&L and predictors as fractions of S0·G0 (the spot carried to expiry) instead of C0·G0, that is multiplied by C0/S0 in each window and shown in bp of S0, for the full and April-excluded samples on the same resamples. It is reported with the Spearman rank correlation of P&L with each predictor, under both normalisations, for both σ_i.
- Second gap regressor: the 30-day butterfly BF = (IV(25-delta put) + IV(25-delta call))/2 - sigma_atm alongside RR, summarised as the gap is, with its correlation with RR. OLS of the gap on RR and BF, reporting intercept, both slopes and R², with block bootstrap intervals on the daily series as above (block 21).
- ATM against the surface: sigma_atm against the surface's 50-delta call IV, reported as aggregates only. These are the mean, median, std and 5th and 95th percentiles of the difference, the share of dates where its absolute value exceeds 1 and 2 vol points, and the count of those dates in each year.
- Weekend check: ρ over the returns of the Stage 5c windows, next to the realised shortfall for both σ_i on those windows (the weekend variance ratio above).
- Figure 7b: VIX, the OptionMetrics 30-day ATM vol and the realised vol of the following 30 days, as monthly means over the windows starting in each calendar month. The title's mean gap is VIX - sigma_atm averaged over the same window starts.

## 4. Discrete hedging checks (tests/test_hedging.py)
Setup: GBM with σ_true = σ_i = 0.18, r = q = 0, S0 = 100, K = F0, T = 21/252, 21 daily steps, 200,000 paths, fixed seed. With r = 0, C0·G0 = C0, so only the carry check depends on the money-unit convention.
- Mean P&L/C0 within ±0.5% (verified -0.02%).
- Std/C0 in [0.175, 0.200] (verified 0.187; Derman-Kamal √(π/4)·vega·σ/(√N·C0) = 0.193347, an asymptotic formula that overstates at small N).
- Every 2 days: std/C0 about 0.2566 (Derman-Kamal 0.273). The grid is anchored to expiry, so the one-day stub opens the window instead of sitting on the final day, where ATM gamma peaks; the start-anchored grid gave 0.252.
- Mean interior turnover within 5% of √N/π = 1.458679 shares per option share (verified 1.417).
- 5 bp, daily: mean -6.0% ± 0.5% of C0, std about 18.4%.
- Carry consistency: a total-return path (drift r = 0.04, σ = 0.18) priced and hedged with q = 0.013 gives mean +2.69% ± 0.3% of C0·G0; with q = 0 it gives 0 ± 0.3% (verified -0.08%). This guards against pairing adjusted closes with a dividend yield. +2.69% is the exact expectation C(q = 0)/C(q) - 1 = 0.026907 with K = S0·exp((r - q)T), since the hedge has zero mean under Q whatever delta is used. It was +2.70% (e^{rT} times this) when P&L was divided by C0 without carrying it to expiry; the earlier +2.64% included sampling noise.
- P_step R²_45 ≥ 0.95 on the synthetic window set in section 3 (verified 0.977).
- Price predictor: σ_i = 0.18 on GBM paths with σ_true = 0.27 and 0.12, drift and flat financing r = 0.04, K = F0, daily hedging, c = 0, 200,000 paths. Mean P&L/(C0·G0) equals P_price at the true vol, BS(σ_true)/BS(σ_i) - 1 = 0.499789 and -0.333292 (exact), within ±0.003 (verified 0.4978 and -0.3330; the tolerance is 0.6% and 0.9% of the exact values, about 4 and 8 standard errors). P_gap at the true vol overstates the first and understates the second by the factor (σ_true + σ_i)/(2σ_i), 1.25 and 0.83.

## 5. Chain cleaning (Stage 2b)
- T = (expiry at 16:00 America/New_York - snapshot timestamp)/365 days, using the full datetime.
- Expiry selection: third-Friday monthlies out to 1 year plus quarterly and LEAPS expiries beyond; 8 to 12 slices; drop T < 7 days.
- Forwards from parity, per expiry, before any OTM filtering: strikes with valid call and put quotes and |ln(K/S)| ≤ 0.05 (at least 8 pairs); weighted least squares of (C_mid - P_mid) on K with weights 1/(h_C² + h_P²), h = half-spread. D = -slope, F = intercept/D. Implied r = -ln(D)/T; implied carry q = r - ln(F/S)/T. Tabulate per expiry (expect near-zero dividend carry for expiries before the mid-December ex-date).
- Parity diagnostic: residuals against the combined half-spread; report the share within.
- Filters in order, counting removals: zero bid; (ask - bid)/mid > 0.25; open interest < 10. The no-IV filter is applied in Stage 2c.
- OTM selection with the fitted F: puts with K < F, calls with K ≥ F. k = ln(K/F), w = σ²T.
- Test: a synthetic chain built from known F and D recovers both to 1e-8.

## 6. Implied vol (Stage 2c)
- brentq on [0.001, 5.0] with xtol 1e-14, after checking for a sign change; otherwise return NaN and count the contract under "no IV" in Table 1.
- Round trip (verified): T in {7, 14, 30, 91, 365} days, K/F in {0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5}, σ in {0.05, 0.1, 0.2, 0.4, 0.8}, puts below F and calls at or above: σ recovered within 1e-6 in every case where the price is above 0. The only failures are prices that underflow to exactly 0, so the test skips those.
- Newton demo (verified): F = 100, K = 85, T = 14/365, put, σ = 0.40, price 0.05007364. Naive start σ0 = 0.20: iterate 1 = 38.243449, iterate 2 = -13054.718015 (fails). Manaster-Koehler start σ0 = √(2|ln(F/K)|/T) = 2.911048, the inflection point where vega peaks in σ: converges monotonically to 0.40 in 8 iterations.
- Figure 1: raw smiles, IV (%) against k, one series per expiry.

## 7. SVI (Stage 3)
- Raw SVI: w(k) = a + b·(ρ·(k - m) + √((k - m)² + s²)).
- Constraints: b ≥ 0; |ρ| < 1; s > 0; a + b·s·√(1 - ρ²) ≥ 0 (so w ≥ 0); b·(1 + |ρ|) ≤ 2 (Lee's moment bound).
- Calibration (Zeliade quasi-explicit): for fixed (m, s), with y = (k - m)/s, w = a + d·y + c·√(y² + 1), where c = b·s and d = ρ·b·s. Inner problem: weighted least squares in (a, c, d) subject to c ≥ 0, |d| ≤ c, c + |d| ≤ 2s, and a + √(c² - d²) ≥ 0 (a convex set; solve with SLSQP). Outer problem: minimise over (m, s) with Nelder-Mead from a small grid of starts.
- Weights: vega-weighted residuals in total variance, normalised per slice.
- Cross-check: a direct 5-parameter L-BFGS-B fit from 20 random starts must not beat the quasi-explicit objective by more than 1e-10.
- Fit error in vol points: RMSE over |k| ≤ √w_ATM (near the money) and over all points. Target: near the money under 1 vol point on every slice.
- Test: a noiseless synthetic slice is recovered to 1e-8 in w.

## 8. Arbitrage checks and bridge (Stage 4)
- Butterfly: g(k) = (1 - k·w'/(2w))² - (w'²/4)·(1/w + 1/4) + w''/2 ≥ 0 on a fine grid k in [-2.0, 1.0] (6,001 points). Density p(k) = g(k)/√(2πw)·exp(-d_-²/2), with d_- = -k/√w - √w/2.
- Density test slice (verified; a = 0.0010, b = 0.012, ρ = -0.75, m = 0.015, s = 0.03, T = 30/365): p(k) at k = -0.3, -0.2, -0.1, 0, 0.1, 0.2 equals 0.00483804, 0.0590773, 0.782553, 11.2865, 0.196568, 2.00322e-05, matching a numerical Breeden-Litzenberger second derivative of forward call prices.
- Calendar: on the same grid, w_{i+1}(k) ≥ w_i(k) for consecutive expiries.
- Constrained refit, shortest expiry first: add g(k_j) ≥ 0 and w_i(k_j) ≥ w_{i-1}(k_j) on a 201-point grid; report violations on the fine grid before and after (Table 3).
- Figure 3: linear interpolation of w in T at fixed k (this preserves calendar ordering). README must state that butterfly-freeness is certified on the fitted slices; between slices only Gatheral-Jacquier price interpolation guarantees it.
- Bridge: build a 30-day slice by linear interpolation of w in T at fixed k between the bracketing expiries. Variance-swap total variance by replication, 2·∫ OTM(K)/K² dK in forward terms, cross-checked with Gatheral's ∫ φ(z)·w(k(z)) dz where z = -k/√w - √w/2. Test slice (verified): 14.9647% by both methods against ATM 13.6770%. For the snapshot day report the variance-swap vol, ATM vol, their gap, and the ^VIX close; feed the gap into the Stage 5 sensitivity.

## 9. Figures and tables
Figures
1. Raw smiles: IV (%) against k, one series per expiry.
2. SVI fits: one panel per expiry, w against k, market points and fitted curve.
3. Surface: IV (%) over k and T.
4. Calendar check: w against k, all expiries on shared axes.
5. Butterfly check: density (or g) against k per expiry.
6. Actual P&L against P_gap, one point per window, 45 degree line, R²_45 and slope on the plot (optional second panel for P_step).
7. VIX against realised vol of the following 30 days, by date.
7b. VIX and the OptionMetrics 30-day ATM vol against realised vol of the following 30 days, monthly means by window start month (Stage 5c; skipped without the licensed files).
8. Hedging error RMSE against rebalances per window, one line per cost level, with the mean/std panel.

Tables
1. Data summary: expiries, strike ranges, contracts kept, removals per filter, and implied F, D, r and carry per expiry.
2. SVI parameters per expiry with near-the-money and overall fit errors in vol points.
3. Butterfly and calendar violations per expiry, before and after constraints.
4. Hedging summary by h and c: mean, std, RMSE, the R² ladder with slopes, plus the robustness and sensitivity rows. Table 4d is the Stage 5c sensitivity: σ_i = OptionMetrics ATM vol against σ_i = VIX on the same windows.

## 10. Limitations to state in README
- Overlapping windows are not independent: report the non-overlapping subsample and moving block bootstrap intervals.
- SPY options are American; European pricing on OTM quotes leaves a small early-exercise bias, largest for long-dated puts.
- VIX is a variance-swap level and sits above ATM vol by the section 8 gap; the sensitivity bounds its effect.
- One snapshot: the surface describes a single moment.
- Yahoo quotes are delayed about 15 minutes, and mids from wide markets are noisy.
- The hedge is simplified: σ_i fixed, flat costs, no market impact, one rate for financing.
- The Stage 5 option is written on the total-return series, so ex-dividend drops are absent by construction.
- The option is struck and priced at the opening rate r0 while cash accrues at the daily rates, so rate moves within a window leave a small term in the P&L.
- ^VIX closes at 16:15 ET, 15 minutes after SPY, so σ_i is observed slightly after the SPY close it is paired with.
- Delta is computed from a close and traded at that same close, which assumes no execution lag.
- P_gap is ex post by construction: it uses the window's own realised variance, so it attributes P&L after the fact and is not a forecast.
- The OptionMetrics data is licensed and not distributed, so the Stage 5c sensitivity runs only where the files are present, and it covers only windows starting up to 2025-08-29.
