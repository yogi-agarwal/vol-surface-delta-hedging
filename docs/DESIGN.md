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
- P_path = ½·(σ_r² - σ_i²)·Σ_j Γ_j·S_j²·Δt_j
- P_step = Σ_j ½·Γ_j·S_j²·(R_j² - σ_i²·Δt_j), with R_j the simple return over hedge interval j
- For each predictor report R²_45 = 1 - Σ(y - p)²/Σ(y - ȳ)² (about the 45 degree line) and OLS y = α + β·p. Theory says α = 0, β = 1.
- Confidence intervals for R²_45 and β come from a moving block bootstrap over windows in start-date order, block length 42 windows (about two window lengths, so each block spans the overlap), alongside the non-overlapping subsamples. This replaces Newey-West errors. 2,000 resamples, seed 20260930: each resample joins blocks of 42 consecutive windows whose first windows are drawn uniformly, with replacement, from all n - 41 positions, truncated to n windows; the intervals are the 2.5% and 97.5% percentiles. The same resampled windows serve every predictor and every h; the April-excluded sample is resampled the same way on its own sequence.
- The ladder is reported at c = 0 for every h: the predictors are frictionless, and costs only add a mean shift that R²_45 would charge to all three alike.
- The headline R² is P_gap's, because the research question asks about the vol gap. The ladder attributes the shortfall: moneyness drift (P_path vs P_gap), timing of moves against gamma (P_step vs P_path), discreteness and jumps (actual vs P_step).
- Benchmark: even on pure GBM windows (the synthetic set below, daily hedging), P_gap's R²_45 is only about 0.8 (0.77 below), because the vol gap alone ignores where the path spends its gamma. Real data is judged against this benchmark, not against 1.
- Sanity ranges on synthetic GBM windows (verified; 3,000 windows, σ_i ~ lognormal(ln 0.17, 0.35), σ_true = σ_i·exp(N(-0.2, 0.3))): daily R²_45 = 0.77 / 0.93 / 0.98 for P_gap / P_path / P_step; every 5 days 0.41 / 0.45 / 0.89 (median of 40 seed pairs). Expiry anchoring puts a full 5-day interval where gamma peaks, so the quadratic approximation is worse; the start-anchored grid's 0.40 / 0.46 / 0.92 was flattered by its 1-day stub before expiry. Real data should sit lower on P_step because of jumps.
- These ranges are illustrative and seed-dependent, so no test asserts them (section 4 tests only P_step R²_45 ≥ 0.95). Across 200 seed pairs the daily P_step median is 0.976, and a single large-move window can pull it down.
- Real-calendar GBM benchmark (the one the headline cites): every real window gets 100 GBM paths from 100 on its own closes, with σ_true equal to its realised vol √(RV/T0), σ_i equal to its VIX, drift and financing at its real step rates (the risk-neutral drift of a total-return series), K = F0 and c = 0. The ladder is computed over all paths for every h, with P_gap from each path's own realised variance. Windows of n steps draw their normals from np.random.default_rng([20261001, n]). The section 3 synthetic set above is printed separately as the textbook reference only: median over 40 seed pairs (20261100 + k, 20261200 + k), k = 0 to 39, for h = 1 and 5.
- Weekday diagnostic (daily hedging, c = 0): P_step residuals, the hedged P&L of each interval (the per-interval output above) minus its P_step term, averaged by the weekday of the interval's closing day. Δt counts calendar days, so a Friday-to-Monday interval charges three days of σ_i² against roughly one trading day of realised variance; the diagnostic shows how much of the residual this weekend theta effect explains. Per weekday: interval count, mean calendar days, mean interval P&L, mean P_step term, mean residual with a moving block bootstrap interval (the full-sample resamples), and the weekday's share of the total residual; plus η², the share of residual variance explained by the weekday means.

Hedging error and Figure 8
- e = P&L - P_gap per window. For each (h, c) report mean, std and RMSE = √(mean² + std²), with the population std (ddof 0), so RMSE² = mean(e²).
- Figure 8: RMSE against N, one line per cost level, plus a panel with mean and std; overlay the synthetic GBM curve (σ_true = σ_i) as a reference. N varies with window length, so the x-axis is the mean N per h across windows, with the range (minimum to maximum) shown.
- The synthetic reference computes P_gap from each synthetic path's own realised variance, the same definition as on real data, so the curves are comparable. It is the real-calendar GBM construction of the benchmark above with σ_true = σ_i, at c = 0, on the same normals, so the two synthetic sets differ only in σ_true. Also draw the known-vol Derman-Kamal curve (P_gap = 0), √(π/4)·mean(vega0·σ_i/C0)/√N, as a labelled dashed line.
- Expected pattern (verified, synthetic, and still true on the expiry-anchored grid). It refers to raw synthetic P&L with σ_true = σ_i; real data reports the same statistics for e = P&L - P_gap. Costs move the mean, not the spread. Daily std of P&L (the P_gap = 0 case) at 0 / 1 / 5 bp was 18.6% / 18.6% / 18.4% of premium. On RMSE, daily beats every-2-days and every-5-days up to 5 bp, ties every-2-days near 25 bp, and loses to every-5-days at 100 bp.

Robustness
- Exclude every window whose closes t0 to t_end include any trading day from 2025-04-03 to 2025-04-09 and rerun the ladder.
- Non-overlapping subsamples at a fixed stride of 22: subsample k holds windows k, k + 22, k + 44, ... for each offset k = 0 to 21, so the 22 subsamples partition all windows (about 56 windows each). No window has more than 22 steps, so each window of a subsample starts at or after the previous one's end close and no two share a daily return; the code checks this. A stride of 21 would share a step wherever a window has 22 steps, and a greedy rule (each next window the first to start at or after the previous end) was rejected because its chains lock onto Friday-to-Friday windows and merge, leaving few distinct subsamples. Report the point estimates for offset 0, plus the median and the minimum to maximum of R²_45 and β across all 22. These subsamples are not bootstrapped.
- Report the share of windows with σ_i²·T0 > RV once; it does not depend on h or c.
- Sensitivity: rerun with σ_i = VIX minus the variance-swap gap measured in section 8.

## 4. Discrete hedging checks (tests/test_hedging.py)
Setup: GBM with σ_true = σ_i = 0.18, r = q = 0, S0 = 100, K = F0, T = 21/252, 21 daily steps, 200,000 paths, fixed seed. With r = 0, C0·G0 = C0, so only the carry check depends on the money-unit convention.
- Mean P&L/C0 within ±0.5% (verified -0.02%).
- Std/C0 in [0.175, 0.200] (verified 0.187; Derman-Kamal √(π/4)·vega·σ/(√N·C0) = 0.193347, an asymptotic formula that overstates at small N).
- Every 2 days: std/C0 about 0.2566 (Derman-Kamal 0.273). The grid is anchored to expiry, so the one-day stub opens the window instead of sitting on the final day, where ATM gamma peaks; the start-anchored grid gave 0.252.
- Mean interior turnover within 5% of √N/π = 1.458679 shares per option share (verified 1.417).
- 5 bp, daily: mean -6.0% ± 0.5% of C0, std about 18.4%.
- Carry consistency: a total-return path (drift r = 0.04, σ = 0.18) priced and hedged with q = 0.013 gives mean +2.69% ± 0.3% of C0·G0; with q = 0 it gives 0 ± 0.3% (verified -0.08%). This guards against pairing adjusted closes with a dividend yield. +2.69% is the exact expectation C(q = 0)/C(q) - 1 = 0.026907 with K = S0·exp((r - q)T), since the hedge has zero mean under Q whatever delta is used. It was +2.70% (e^{rT} times this) when P&L was divided by C0 without carrying it to expiry; the earlier +2.64% included sampling noise.
- P_step R²_45 ≥ 0.95 on the synthetic window set in section 3 (verified 0.977).

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
8. Hedging error RMSE against rebalances per window, one line per cost level, with the mean/std panel.

Tables
1. Data summary: expiries, strike ranges, contracts kept, removals per filter, and implied F, D, r and carry per expiry.
2. SVI parameters per expiry with near-the-money and overall fit errors in vol points.
3. Butterfly and calendar violations per expiry, before and after constraints.
4. Hedging summary by h and c: mean, std, RMSE, the R² ladder with slopes, plus the robustness and sensitivity rows.

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
