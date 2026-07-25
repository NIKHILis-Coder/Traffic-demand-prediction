# Interview Prep: Traffic Demand Prediction

This is a study guide for defending this project in an interview. Every number here is traceable to a specific notebook cell — if asked "how do you know that," the answer is always "I printed/plotted it in step X." Nothing here is rounded up or hand-waved.

---

## 1. Problem framing & why it matters

**Task:** predict `demand` — a continuous value in `[0, 1]` — for a specific road segment (`geohash`) at a specific 15-minute time slot, given static attributes of that segment (road type, lane count, whether large vehicles/landmarks are nearby) and conditions at that moment (weather, temperature).

**Why this is a regression problem, not classification:** the target is a continuous proportion, not a category, and the loss that matters (how far off is the prediction) is naturally squared/absolute error, not a decision boundary.

**Why it matters in the real world:** this is the shape of problem behind traffic-aware routing, dynamic signal timing, and ride-hailing/delivery ETA and pricing systems — all of which need a demand estimate *before* the event happens, from features you already know (time, place, weather forecast), not from live sensor feeds. A model like this is a component you'd feed into a larger system, not a standalone product.

**What I'd tell an interviewer if asked "is this a real dataset":** I don't have external confirmation of the data's provenance, and I should say so directly rather than imply it's a live traffic feed. See [§7 Limitations](#7-limitations-honest-version) for why I suspect it's synthetic/competition-style data.

---

## 2. Data

**Source files:** `data/raw/train.csv` (77,299 rows, labeled), `data/raw/test.csv` (41,778 rows, unlabeled — this is a holdout set with no target column, so I never get to see a true test-set score, only the cross-validated estimate from train). `sample_submission.csv` defines the expected output format.

**Raw columns (11) and why each was kept:**

| Column | Type | Why it's in the model |
|---|---|---|
| `geohash` | categorical, 1,249 unique | Dominant signal — see §3. Encoded via LOO target encoding + decoded lat/lon. |
| `day` | int, 2 values (48, 49) | Checked in EDA, found to carry ~zero signal (near-identical mean demand across both days) — kept as a raw feature during exploration but **dropped from the final 26** because `config.FINAL_FEATURES` doesn't include it. If asked why it's not in the final set: two days isn't a time series, it's just two samples of "day," so there's no seasonality or trend to learn from it. |
| `timestamp` | string `H:MM` | Parsed into `hour`, then cyclically encoded (`hour_sin`/`hour_cos`) plus several derived flags — the single most informative raw column after geohash. |
| `RoadType` | categorical (Highway/Street/Residential), 600 missing (0.78%) | Ordinal-encoded by observed mean demand (Highway > Street > Residential). Strong signal (Pearson 0.86 with demand once encoded). |
| `NumberofLanes` | int, 1–5 | Used as-is; demand increases monotonically with lane count. |
| `LargeVehicles` | binary, no missing | Used to build the composite `road_combo` feature (see §4), then the raw binary column is dropped in step08's trim to `FINAL_FEATURES`. |
| `Landmarks` | binary, no missing | Same treatment as `LargeVehicles` — folded into `road_combo`. |
| `Temperature` | float, 2,495 missing (3.23%) | Geohash-group-median imputed (location-aware, not a single global fill). Weak linear correlation with demand but a real non-linear (inverted-U) relationship — see `temp_squared`. |
| `Weather` | categorical (Sunny/Foggy/Rainy/Snowy), 797 missing (1.03%) | Ordinal-encoded by severity, confirmed against observed demand suppression order. |
| `Index` | row id | Dropped immediately, zero predictive value. |
| `demand` | target, float `[0,1]` | Mean 0.094, median 0.048 — **right-skewed**, skewness ≈ 3.73 (computed in `notebooks/eda/univariate_analysis.ipynb`). |

**Row counts, precisely:** 77,299 train / 41,778 test (verified by reading the raw CSVs directly — this replaced a bug where an earlier version of the pipeline printed fabricated row counts; see §7).

---

## 3. EDA — the findings that actually drove decisions

Three EDA notebooks, in order: `raw_data_analysis` → `univariate_analysis` → `bivariate_analysis` → `multivariate_analysis`. Every feature-engineering choice in step04–step08 traces back to one of these.

**The single biggest finding: geohash dominates.** Mean demand across the 1,249 geohash cells ranges from 0.0005 to 0.96 — a **~1,940x** gap between the busiest and quietest cell (`bivariate_analysis`, cell 2). This one fact drives the whole spatial feature strategy: one-hot encoding 1,249 categories is impractical and would dilute the signal, so target encoding is the only realistic choice.

**Dual-peak daily pattern, and it's road-type-specific.** Demand has a clear morning (7–9) and evening (17–20) peak with an overnight trough — but the multivariate EDA showed this spike is a **highway phenomenon**; residential streets stay nearly flat all day (`multivariate_analysis`, cell 1). This directly motivated the `hour_x_roadtype` interaction feature, which turned out to be the most important interaction feature by model importance.

**RoadType > Weather > Temperature in strength, and geohash > all of them.** Bivariate correlation ranking (Pearson, non-geohash features): `RoadType_enc` 0.86, `NumberofLanes` 0.21, `LargeVehicles_enc` 0.19, everything else under 0.04. Weather and Temperature individually have almost no linear correlation with demand (`Weather` −0.002, `Temperature` 0.003) — but that doesn't mean they're useless, it means their effect isn't linear (see below).

**Weather is a shift, not a shape change.** Multivariate EDA showed the hourly demand curve keeps its dual-peak shape under every weather condition — weather suppresses the *overall level*, it doesn't move *when* people travel (`multivariate_analysis`, cell 2). This is why weather doesn't get its own hour-interaction feature — the interaction with road type (`road_x_weather`) mattered more, since severe weather suppresses highway demand more than residential.

**Temperature has a real but non-linear (inverted-U) relationship** — lowest demand at extremes, highest in the mid-range (`bivariate_analysis`, cell 10). This is why `temp_squared` exists despite Temperature's near-zero linear correlation — see §7 for the honest caveat about that near-zero number.

**Outliers are real, not errors, and weren't removed.** High-demand rows (top ~8% by IQR) have genuinely higher mean demand (0.47 vs 0.06) — they're busy locations at busy times, not sensor noise. `demand` is bounded `[0,1]` by construction, so nothing there needed clipping at the EDA stage.

---

## 4. Feature engineering — what and why

11 raw columns → 26 features, across steps 01–08 of `notebooks/pipeline/`.

### Geospatial (step04) — the most consequential step

- **`latitude`/`longitude`**: decoded from the geohash string via `pygeohash.decode()`. Adds continuous spatial proximity that a purely categorical target encoding can't express (two adjacent-but-distinct geohash cells get very different target-encoded values even though they're physically next to each other; lat/lon lets the model learn that they're close).
- **`geohash_target_enc`**: **leave-one-out (LOO)** target encoding. Formula: for each row, `(sum of demand for this geohash − this row's own demand) / (count − 1)`, falling back to the global mean for singleton cells. This is the single most important design choice in the whole feature set — a plain `groupby().mean()` would leak each row's own target into its own feature, and with cells as small as 1 row, that leak would be severe. LOO explicitly excludes the current row.
- **`geohash_prefix_enc`**: same LOO idea applied to a coarser 4-character geohash prefix (6 groups instead of 1,249), as a smoothing fallback for sparse cells. **Honest note:** this one *did* originally use a plain (non-LOO) mean — I found and fixed that inconsistency during a repo cleanup pass. Given only 6 groups of thousands of rows each, the leak was numerically negligible (order 1/10,000 per row), and a full retrain confirmed it: OOF RMSE/R² came back matching the pre-fix numbers to 4 decimal places (see [§7, limitation #2](#7-limitations-honest-version)) — the fix was correct but its practical effect was, as predicted, negligible.
- **Three-level test-set fallback**: exact geohash mean → prefix mean → global mean. Necessary because test can contain locations training never saw (25 of 41,778 test rows fell back to the prefix mean; zero needed the global mean).

### Temporal (step05)

- **`hour_sin`/`hour_cos`**: paired cyclical encoding, `sin/cos(2π·hour/24)`. Needed as a *pair* — sin alone can't distinguish hour 6 from hour 18 (same sin value, different cos), so together they place every hour at a unique point on a circle, making 23:00 and 00:00 adjacent instead of maximally far apart on a naive 0–23 scale.
- **`is_peak_hour`/`is_business_hour`/`is_night`**: boolean flags from EDA-confirmed hour ranges (`PEAK_HOURS = [(7,9),(17,20)]` etc., defined once in `config.py`).
- **`hour_squared`**: alongside sin/cos to let tree splits separate "near midnight" from "far from midnight" directly, since the actual demand curve has a mid-day plateau that isn't a clean sinusoid.

### Road (step06)

- **`road_capacity` = `road_type_ord × NumberofLanes`**: a 3-lane highway and a 3-lane residential street are very different roads; this makes that difference explicit as one feature instead of leaving the model to infer the interaction.
- **`road_combo`**: `LargeVehicles × Landmarks` as a 4-category composite, ordinally encoded by *observed* mean demand (not a guessed order) — computed in step07 after the raw string combo is built in step06.

### Interactions (step07) — "65% of RMSE gain" per the approach doc's staged comparison

8 features, each justified by a specific multivariate EDA finding (see `notebooks/eda/multivariate_analysis.ipynb`'s interaction-justification table): `hour_x_roadtype`, `geo_hour_sin`, `road_x_weather`, `geo_x_roadtype`, `geo_x_weather`, `lanes_x_weather`, `geo_x_peak`, `temp_x_weather`.

**Interview trap to be ready for: "your `geo_x_roadtype` has 0.88 correlation with demand — isn't that leakage?"** No, but it's worth explaining precisely: `geohash_target_enc` is *already* an LOO estimate of that location's mean demand, so anything multiplied by it inherits a chunk of that correlation "for free" — not because of leakage (LOO still excludes the row's own value), but because the base feature it's built from is already highly informative. I have this exact caveat written into the step07 notebook, not just in my head.

### Weather (step08) + final trim

- **`temp_squared`**: captures the inverted-U shape. Correlation with demand comes out near zero (0.0032) — expected for a symmetric non-linear relationship (a straight-line correlation coefficient can't see a U-shape), and not a signal to drop the feature for the tree-based models actually used.
- Trimmed to the final 26-feature list (`config.FINAL_FEATURES`), confirmed 0 nulls across all of them.

---

## 5. Modeling

**Progression (from the approach doc, all numbers verified against the notebooks):**

| Stage | Model | RMSE | R² | What changed |
|---|---|---|---|---|
| 1 | XGBoost, manual params | 0.0395 | 92.27% | Baseline |
| 2 | XGB + LGB stacked, manual params | 0.0363 | 93.49% | Added LightGBM — gains were small because both models made similar errors (low diversity) |
| 3 | XGBoost, Optuna-tuned | 0.0309 | 95.26% | Optuna search + added lat/lon |
| 4 | LightGBM, Optuna-tuned | 0.0274 | 96.27% | Same |
| 5 | CatBoost, Optuna-tuned (new) | 0.0274 | 96.29% | Ordered boosting gives genuinely different error patterns than 3/4 |
| 6 | **Stacked ensemble (final)** | **0.0264** | **96.55%** | Ridge meta-learner on 3 base models' OOF predictions |

*(Row 6 updated 2026-07-25: re-verified end-to-end after fixing the meta-learner OOF leak and the final-fit tree-count bug — see [§7, limitations #8–9](#7-limitations-honest-version). The stacked number moved from 0.0263/96.57% to 0.0264/96.55%, i.e. it barely changed; the fixes were about methodological correctness, not about chasing a better score.)*

**Why gradient boosted trees, not a neural net:** tabular data with mostly low-to-moderate cardinality categoricals and a bounded skewed target — trees handle this natively (no need to normalize/embed features), need less tuning, and are the standard strong baseline for this data shape. I did not try a neural net; if asked "would a NN help here," the honest answer is "probably not enough to justify the complexity, given how much of the signal is already captured by geohash + interactions — but I haven't tested it."

**Why three different tree models, not three seeds of the same model:** the whole point of stacking is combining models that make *different* errors. XGBoost (depth-wise, regularized), LightGBM (leaf-wise, very deep/high-capacity here — `num_leaves=237`, `max_depth=9`), and CatBoost (ordered boosting, built-in categorical handling) have different inductive biases, so their errors are less correlated than three runs of the same algorithm would be.

**Hyperparameters (from `models/best_params.json`, Optuna, 200 trials/model, 3-fold CV per trial):**
- **XGBoost**: `n_estimators=1266, lr=0.0403, max_depth=9, subsample=0.816, colsample_bytree=0.682, min_child_weight=3, gamma=0.0015, reg_alpha=0.116, reg_lambda=4.397`
- **LightGBM**: `n_estimators=2643, lr=0.0304, num_leaves=237, max_depth=9, subsample=0.898, colsample_bytree=0.710, min_child_samples=5, reg_alpha=0.004, reg_lambda=3.447`
- **CatBoost**: `iterations=1913, lr=0.257, depth=8, l2_leaf_reg=4.608, bagging_temperature=0.598, random_strength=0.822`

**But the *deployed* tree counts are slightly lower than the numbers above** — those are the Optuna search budgets, not what the final full-data models actually use. Per the [§7, limitation #9](#7-limitations-honest-version) fix, the final fit reuses the average early-stopped `best_iteration` across the 5 CV folds instead: **XGBoost 1,200 trees** (folds ranged `[1262, 945, 1264, 1264, 1265]` — one fold stopped well short of the others), **LightGBM 2,637 trees** (folds `[2643, 2643, 2641, 2618, 2640]` — barely reduced, it was using nearly its whole budget), **CatBoost 1,909 trees** (folds `[1911, 1913, 1908, 1913, 1900]`). These are logged by the training notebook and saved to `models/metrics.json` alongside the RMSE/R² numbers.

**Why Optuna and not grid/random search:** Optuna uses Bayesian optimization (TPE sampler by default) — it uses the results of earlier trials to pick more promising regions of the search space, which matters here because some params (e.g. `n_estimators` up to 3000, `num_leaves` up to 500) create a huge search space where grid search would be wasteful and pure random search would need many more trials to find the same quality of optimum.

**Why Ridge (not XGBoost-on-top-of-XGBoost, not a simple average) as the meta-learner:** the meta-learner only sees 3 inputs — one prediction per base model. A complex model on 3 features would overfit; Ridge is linear with L2 regularization, and its coefficients are directly interpretable as "how much weight does each base model get." Actual fitted coefficients: **XGB −0.23, LGB 0.63, CAT 0.60** — the negative XGBoost weight is worth being ready to explain (see §7).

**Why 5-fold CV for the final stack but only 3-fold inside the Optuna search:** the 3-fold CV during search is a speed/quality tradeoff — 200 trials × 3 models × 3-fold is already 1,800 model fits; going to 5-fold there would cost 67% more compute for a hyperparameter *search* where you don't need the absolute best CV estimate, just a good relative ranking between trial configs. The final 5-fold is the number that actually gets reported, where the extra fold count buys a lower-variance estimate.

---

## 6. Evaluation

**Metric: RMSE**, primary, because it's on the same scale as the target (interpretable directly as "typical error in demand units") and penalizes large misses more than MAE would — appropriate here because under/over-predicting a genuine high-demand spike is worse than being off on a routine low-demand slot. **R²** reported alongside as a scale-free "how much variance explained" sanity check. **MAPE was not used** — with `demand` values near zero for the median row (0.048), MAPE blows up/becomes unstable on small denominators; this is a real gap, see §7.

**All headline numbers are out-of-fold (OOF)**, from 5-fold CV — not train-set fit, and not a leaked validation set. Concretely: each of the 5 folds' predictions come from a model that never saw that fold during training, so stitching all 5 folds' predictions together and scoring against the true `demand` gives an unbiased estimate of generalization performance *on data shaped like the training set*. Because `test.csv` has no labels (Kaggle-style holdout), the OOF score is the *only* performance estimate available — there's no way to independently verify it against a true held-out test score.

**Honest caveat on the word "all" above:** the 3 base models' OOF predictions were genuinely OOF from the start. The *stacked* number was not, until a bug-fix-and-retrain pass on 2026-07-25 — see [§7 limitation #8](#7-limitations-honest-version) for the bug, the fix, and the verified before/after numbers.

**Residuals** (from the *now-genuinely-OOF* stacked predictions, `step10_part2_training.ipynb`, re-verified 2026-07-25): mean ≈ **−0.0001** (no systematic bias), std ≈ **0.0257**, range **−0.246 to +0.293**. Essentially identical to the pre-fix numbers (std 0.0256, range −0.25 to +0.30) — the near-zero mean is a good sign the model isn't systematically over- or under-predicting, and the tail means there are individual rows the model misses badly (some real high-demand spikes get under-predicted), expected for a right-skewed target where the rare high-demand tail is intrinsically harder to hit exactly.

---

## 7. Limitations (honest version)

Things I would not want to be caught not knowing about my own project:

1. **The 26-feature set fails VIF badly, and I kept it anyway.** 16 of 23 extended features exceed VIF 10 (some, like `latitude`, exceed 10,000). My documented reasoning: most of the high-VIF features are literal products of other retained features (`road_capacity`, every `*_x_*` interaction), so high VIF is structurally expected, not a modeling mistake. It matters for linear models — my Ridge meta-learner never sees these 26 features directly, only the 3 base models' predictions — and doesn't matter for split-based tree models, which are invariant to linear redundancy. I did *not* verify this claim by actually training a VIF-trimmed linear baseline for comparison — that's a real gap, not just a rhetorical answer. If pushed: "I'd want to prove this by actually training a linear model on the full 26 vs. the VIF-trimmed 10 and showing the trimmed one is *not* meaningfully worse, rather than asserting it from theory."

2. **A leakage fix exists in code, and it took a second attempt to actually verify it.** I found that `geohash_prefix_enc` used plain (non-LOO) mean encoding — inconsistent with the primary geohash encoding's LOO discipline — and fixed it. The *first* time I tried retraining against this fix, the current environment's XGBoost/LightGBM/CatBoost versions produced results ~3x better than the documented numbers in a way I couldn't explain (early stopping behaving very differently between versions) — swapping in an unverified, suspiciously-good result would have been worse than leaving a known, tiny, well-understood gap, so I left it undocumented as a retrain rather than pretend it was resolved. **Update, 2026-07-25:** I retrained again, this time in the project's pinned `mlpr` virtual environment (matches `requirements.txt` exactly — pandas 3.0.3, numpy 2.4.6, xgboost 3.2.0, lightgbm 4.6.0, catboost 1.2.10, scikit-learn 1.8.0), and this run reproduced the *original* documented numbers almost exactly (XGB/LGB/CAT OOF RMSE 0.0309/0.0274/0.0274 — identical to 4 decimals) — the earlier ~3x-better anomaly did **not** recur. I still don't know exactly what caused the first anomalous attempt (different package versions at the time, most likely), but this gives me real confidence the pipeline itself is correct and the headline numbers are reproducible, not a fluke of one lucky run. If asked "did retraining after your fix change the results": **"Yes, I verified it end-to-end — same numbers to 4 decimal places, confirming the `geohash_prefix_enc` fix had no meaningful effect (as expected, since I'd already reasoned the leak was numerically tiny) and that the pipeline reproduces cleanly in a pinned environment."**

3. **Mild optimism bias from using the same fold for early stopping and scoring.** Each base model's `eval_set` during CV is the same held-out fold used for that fold's OOF prediction — meaning early stopping is tuned to minimize error on the exact data later used to score it. This is common practice (not a bug), but it does mean the reported OOF RMSE is very slightly optimistic vs. a stricter nested-CV setup with a third split reserved purely for stopping decisions.

4. **`day` has only 2 distinct values (48, 49).** This isn't a real time series — there's no way to validate the model generalizes across different days, seasons, or long-term trends. Anything said about "how the model handles day-of-week effects" would be speculation; it simply hasn't seen more than 2 days.

5. **The data may well be synthetic/competition data, not real traffic telemetry.** Signs: the near-perfect regularity of geohash sampling (~96 rows per cell, one per 15-minute slot), the suspiciously narrow geographic bounding box, and how cleanly the strong features (geohash, road type) separate the target. I have not independently verified the data's provenance and would say so plainly if asked, rather than imply operational realism it may not have.

6. **No MAPE, and demand near zero makes it a bad fit anyway.** If a stakeholder specifically needs "percentage error" framing (common in ops contexts), RMSE/R² don't directly answer that, and MAPE is numerically unstable given the target's right skew toward near-zero values.

7. **No true holdout test score.** Because `test.csv` is unlabeled, I can't report a number I'm fully confident wasn't at all influenced by iterating on the pipeline against OOF CV — a classic risk in any Kaggle-style setup where you tune against the only evaluation signal you have.

8. **The stacked "OOF" number wasn't actually out-of-fold — found, fixed, and re-verified.** In `step10_model_training.ipynb` and `step10_part2_training.ipynb` (cell `fe3d2d8a`), the Ridge meta-learner was `.fit(oof_stack, y)` on the *full* set of base-model OOF predictions and then `.predict()`'d on that same `oof_stack` — that's a train-fit score, not an OOF score, even though it's one step removed from the base models (the base models' own predictions were still genuinely OOF). The notebook also computed a second, honest number alongside it via `cross_val_score`, and by coincidence both rounded to the same 0.0263 RMSE — but that's luck, not something the code guaranteed, and a stronger or less-regularized meta-learner could easily have shown a real gap between the two. **Fix applied:** replaced the fit-then-predict-on-same-data step with `cross_val_predict` (5-fold), so `oof_stacked_pred` — and everything downstream of it (the residual stats in [§6](#6-evaluation), the scatter plot, the reported RMSE/R²) — is now genuinely out-of-fold. The Ridge coefficients themselves are still fit on the *full* OOF stack afterward, which is correct: the final deployed meta-learner should use every available row, it's only the *evaluation number* that has to stay OOF. **Re-run and verified 2026-07-25:** the corrected stacked OOF RMSE is **0.0264** (R² 96.55%), versus the old leaky 0.0263 (96.57%) — a real but tiny difference, confirming the bug existed but had negligible practical impact here. The Ridge coefficients were unchanged (XGB −0.2323, LGB 0.6344, CAT 0.5981), as expected since coefficient fitting was never the leaky part.

9. **Final full-data models had no early-stopping signal at all — found, fixed, and re-verified.** In the same two notebooks (cell `bb77a2f7`), the final XGBoost/LightGBM/CatBoost models were re-fit on 100% of `X`/`y` using the raw Optuna-tuned `n_estimators`/`iterations` (e.g. LightGBM's 2,643) with no `eval_set` and no early stopping. That parameter value is a *search-budget ceiling* the Optuna trial was allowed to spend, actually used only up to whatever early stopping (against a validation fold) cut it off at during CV — it was never validated as the right tree count for a model with no validation set. **Fix applied:** the 5-fold OOF loop already trains and keeps one model per fold (`xgb_models`, `lgb_models`, `cat_models`), each of which knows its own early-stopped `best_iteration`. The final-fit cell now averages those five per-fold best iterations per model family and uses that as a fixed tree count for the full-data fit — no validation split needed for this step, and the final model's capacity now matches what CV actually showed generalizes, instead of an unvalidated upper bound. **Re-run and verified 2026-07-25:** the actual per-fold best iterations were XGB `[1262, 945, 1264, 1264, 1265]` (mean **1200**, vs. the 1266 search budget — fold 2 stopped dramatically early), LGB `[2643, 2643, 2641, 2618, 2640]` (mean **2637**, barely below its 2643 budget), CAT `[1911, 1913, 1908, 1913, 1900]` (mean **1909**, vs. 1913). So the fix mattered a fair amount for XGBoost and barely at all for LightGBM/CatBoost in practice — but it's evidence-based now instead of assumed. Corroborating evidence this was a real risk, not just theoretical: the *old* cell's committed output (before I cleared it) showed LightGBM logging `[LightGBM] [Warning] No further splits with positive gain, best gain: -inf` on repeat for most of its 2,643-tree budget — LightGBM itself flagging it had run out of useful splits well before the end. These tree counts, plus the corrected RMSE/R² above, are now saved to `models/metrics.json` by the training notebook rather than existing only as one-off print output.

10. **The repo was committing ~150MB of fully-regenerable intermediate CSVs directly into git, with no LFS.** Every pipeline step (`step01` through `step08`) saved a full copy of train *and* test to `data/processed/` and committed it — 17 CSVs, several 15–20MB each, none of which add information beyond "the raw data plus the code in this notebook," since every one is deterministically reproducible by rerunning `notebooks/pipeline/` from `data/raw/`. Combined with three model binaries (`best_lgbm_model.pkl` alone is 49MB) and no Git LFS, `.git` had grown to ~450MB. **Fix applied:** `data/processed/` is now git-ignored and untracked (`git rm --cached`, files kept on disk) — this stops further growth on every retrain, though it doesn't shrink the *history* already baked into `.git`, since that would need a destructive history rewrite I didn't want to do without asking first. The model `.pkl` files are left tracked as-is: unlike the processed CSVs they're not trivially regenerable (they're the actual trained artifacts `step11_prediction.ipynb` depends on) and 49MB is still under GitHub's hard limits, but Git LFS would be the right long-term home for them if the repo keeps growing.

11. **The prediction notebook's final summary hardcoded metrics that had already drifted stale — found and fixed.** `step11_prediction.ipynb`'s last cell printed a "FINAL MODEL SUMMARY" with the headline RMSE/R² numbers written as literal strings, not computed — and it showed exactly how fast that goes wrong: it was still printing the pre-fix "Stacked OOF RMSE: 0.0263" even after the limitation #8 fix changed the real number to 0.0264, because nothing forced the two to stay in sync. **Fix applied:** the training notebook now saves every headline metric (`xgb_rmse`, `lgb_rmse`, `cat_rmse`, `stacked_rmse`, the R² equivalents, and the final tree counts from #9) to `models/metrics.json`; the prediction notebook loads that file and formats its summary from it instead of hardcoding. Re-run and confirmed: the summary now correctly prints `0.0264` / `96.55%`. Small bug, but a good example of why "print a number once" and "guarantee two files agree" are different things.

**What I'd do with more time/data:**
- Geohash × hour target encoding (per-cell, per-hour mean) instead of per-cell only, to capture location-specific hourly patterns directly rather than through the `geo_hour_sin` product proxy.
- An actual VIF-trimmed linear baseline to test claim #1 above empirically instead of asserting it.
- If this were real operational data: validate on a genuinely held-out future time period (a true walk-forward split), not just k-fold CV on a 2-day window.
- Move `models/*.pkl` to Git LFS if the project keeps producing new model versions over time.

---

## 8. Likely interview questions

**Q1: Walk me through your pipeline end to end.**
11 numbered notebooks, each reads the previous step's CSV and writes its own: clean → impute → encode → geohash target-encode → temporal features → road features → interaction features → weather features + trim → VIF check → train (tuning, then final fit) → predict. All paths and constants come from one `config.py`. Every encoder is fit on train only.

**Q2: How did you prevent data leakage?**
Every imputer/encoder is `.fit()` on train and only `.transform()`'d on test. The one target-derived feature (`geohash_target_enc`) uses leave-one-out so a row never sees its own label in its own feature. Model evaluation is 5-fold out-of-fold, so the reported score is never computed on data the scoring model was trained on.

**Q3: Why target encoding for geohash instead of one-hot or embeddings?**
1,249 categories — one-hot would make the matrix huge and sparse with no ordering information; a learned embedding needs a lot more data/tuning than a 77K-row dataset can justify. LOO target encoding gives a single dense, informative column directly tied to what we actually care about (demand), at the cost of needing the leakage-prevention discipline above.

**Q4: Why is `geo_x_roadtype` correlated 0.88 with demand — is that leakage?**
No — `geohash_target_enc` itself is LOO (excludes the row's own value), so the correlation is inherited legitimately, not leaked. But it does mean this "interaction feature" is partly re-expressing signal already in `geohash_target_enc`, not purely new information — worth being upfront about rather than presenting it as a fully independent predictor.

**Q5: Why didn't you drop the 16 features that fail VIF?**
Because VIF measures linear collinearity, which threatens linear model coefficients, not split-based tree models. My meta-learner (the one linear model in the pipeline) never sees these 26 features directly — only 3 base-model predictions. See §7 limitation #1 for the honest caveat that I didn't empirically prove this, only reasoned it.

**Q6: Why stacking instead of a simple weighted average of the 3 models?**
A meta-learner can learn *data-dependent* weighting implicitly (Ridge here is a single global linear combination, so not truly data-dependent, but it does learn the optimal fixed weights from data rather than me guessing them), and its coefficients are directly interpretable — I can point to "LightGBM gets 0.63, CatBoost 0.60, XGBoost −0.23" and reason about it.

**Q7: Your Ridge XGBoost weight is negative. What does that mean, and is that a problem?**
It doesn't mean "subtract XGBoost's prediction" in isolation — Ridge coefficients on correlated inputs (all 3 base models are predicting the same target, so their predictions are highly correlated with each other) can go negative to correct for redundancy between predictors, similar to how coefficients behave in any regression with correlated regressors. It's a legitimate, if slightly unintuitive, result of the 3 base predictions being correlated — not a bug. I'd flag that I did not dig further into whether a non-negative-constrained blend would perform meaningfully differently.

**Q8: Why RMSE over MAE or MAPE?**
RMSE is same-scale and penalizes large misses more, which matches the cost structure I assumed (a badly-missed demand spike is worse than several small misses). MAPE is unstable here because the median `demand` is close to zero. This was a judgment call, not something validated against a real business cost function — see §7.

**Q9: How do you know your model isn't overfitting?**
All reported numbers are 5-fold OOF, not train-fit. Residual mean is ~0 (no systematic bias) with a tight std (0.0256) relative to the target's own std (0.142). That said, I don't have a true external holdout to fully rule out overfitting to the CV process itself (see §7, point 7).

**Q10: What's the single most important feature, and how do you know?**
`geohash_target_enc` (or its correlated relatives) — the bivariate EDA showed a ~1,940x demand range across geohash cells before any modeling happened, and feature importance plots (`models/xgb_feature_importance.png` etc.) confirm geohash-derived and road-type features dominate the trained models' splits.

**Q11: If you had to cut this down to 3 features, which would you keep?**
`geohash_target_enc`, `hour_x_roadtype`, `road_type_ord` — this maps directly to the three EDA findings that had the largest, cleanest effect sizes (location, the location-dependent rush-hour spike, and road hierarchy).

**Q12: What would break this model in production?**
A genuinely new geohash cell with a genuinely new prefix (falls back all the way to the global mean — a real but rare case, 0/41,778 in this test set). A shift in the underlying demand pattern (e.g., a new road opens) wouldn't be caught until retraining — there's no online/incremental learning here. And since I only have 2 days of data, I have literally no evidence about how stable these patterns are month-to-month.

**Q13: Why 3 different gradient boosting libraries instead of, say, bagging one model with different seeds?**
Different libraries make structurally different errors (leaf-wise vs depth-wise growth, ordered vs standard boosting, different regularization defaults) — that diversity is what a stacking ensemble needs to gain over any single model. Same-model-different-seed bagging mostly reduces variance from training randomness, not the different-blind-spots diversity stacking is designed to exploit.

**Q14: What's the weakest part of this project if you're honest?**
Probably that I found three separate correctness bugs (a leaky "OOF" metric, an unvalidated final tree count, a stale hardcoded metric) in a project I'd already written up as finished and audited — meaning the first pass wasn't as rigorous as I thought. The good news is the process that caught them (asking "can I point to the exact line that proves this number never saw its own label") is repeatable and I now trust the pipeline more, not less, because I've actually re-run it end-to-end and watched the numbers hold up. The remaining honest gap: the VIF-features-kept decision (§7 #1) is still reasoned from first principles, not empirically tested against a trimmed linear baseline.

**Q15: How would you extend this to a real production system?**
Add a true walk-forward temporal validation (not just k-fold, since 2 days of data can't tell you about drift), add monitoring for feature distribution shift (especially new geohash cells), and probably swap the two-stage manual Optuna-then-retrain notebook flow for a single tracked pipeline (e.g., with MLflow) so hyperparameters and metrics are versioned together instead of living in a JSON file and a markdown doc respectively.

**Q16: You said your stacked "OOF" score wasn't actually out-of-fold. Walk me through that.**
The Ridge meta-learner takes the 3 base models' OOF predictions as its own inputs. I originally evaluated it by fitting Ridge on all of those inputs and then predicting on the same data I'd just fit on — that's a train-fit score for the meta-learner, even though its *inputs* were legitimately OOF. It's a subtle bug because it's one level removed from the obvious mistake (fitting on the labels directly); the giveaway is that "OOF" and "fit-then-predict-on-the-same-rows" are different operations even when the thing you're fitting only has 3 features. I caught it by asking whether every number in the eval section could point to a specific line proving it never saw its own label — this one couldn't, cleanly. The fix is `cross_val_predict` instead of `fit` + `predict` on the same array; see [§7, limitation #8](#7-limitations-honest-version).

**Q17: Why train the final models with a different tree count than the Optuna search suggested?**
Because the Optuna-suggested `n_estimators` is a search-space ceiling, not a validated answer — each CV trial actually stopped early against a validation fold, usually well short of that ceiling. The final model is fit on 100% of the data with no fold to validate against, so blindly using the ceiling has no early-stopping signal at all and risks a model with meaningfully more capacity than anything cross-validation actually confirmed generalizes. I fixed this by averaging the early-stopped `best_iteration` from each of the 5 CV folds and using that as a fixed tree count for the full-data fit — same idea as "refit on all data using the CV-determined stopping point," just without needing to carve out a validation slice for the final fit itself. See [§7, limitation #9](#7-limitations-honest-version).

**Q18: Your repo is committing a lot of data — is that normal?**
It wasn't good practice, and I fixed part of it: every pipeline step was saving *and committing* a full copy of the dataset, purely for debuggability, which bloated the repo to ~450MB in `.git` alone for data that's 100% reproducible from `data/raw/` plus the notebook code. I've since untracked `data/processed/` going forward. I did *not* rewrite git history to shrink what's already committed, since that's a destructive, force-push-requiring operation I'd want explicit sign-off on rather than doing unilaterally. The trained model binaries are still committed as-is (they're not reproducible without a full retrain, and they're the actual thing `step11_prediction.ipynb` depends on) — Git LFS would be the right longer-term home for those.

**Q19: How do you actually know your fixes worked, versus just believing your own patched code?**
I re-ran the full training and prediction notebooks end-to-end in the project's pinned virtual environment (matches `requirements.txt` exactly) and compared before/after numbers rather than trusting the diff alone. Concretely: the 3 base models' OOF RMSE came back identical to 4 decimals (0.0309/0.0274/0.0274), confirming the environment/pipeline itself reproduces cleanly; the stacked RMSE moved slightly (0.0263 → 0.0264) exactly as expected once the meta-learner evaluation became genuinely OOF; and the final tree counts came back lower than the old search-budget ceilings (e.g. XGBoost 1,200 vs. 1,266), with one CV fold showing it stopped at only 945 trees — a concrete number I couldn't have gotten from reasoning alone. I also checked `models/*.pkl` file timestamps to confirm the artifacts were actually regenerated, not just the notebook cells re-executed.

**Q20: One of your fixed notebook cells' old output showed a stale file path from a different folder structure. What does that tell you?**
That the notebook outputs on disk were stale relative to the current repo layout — evidence the pipeline hadn't been re-run since an earlier restructuring commit, which is exactly the kind of thing that erodes trust in a "here are my results" claim if you don't catch it. It's part of why I insisted on an actual end-to-end re-run with fresh timestamps rather than just trusting that patched code equals patched results.
