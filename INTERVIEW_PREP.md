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
- **`geohash_prefix_enc`**: same LOO idea applied to a coarser 4-character geohash prefix (6 groups instead of 1,249), as a smoothing fallback for sparse cells. **Honest note:** this one *did* originally use a plain (non-LOO) mean — I found and fixed that inconsistency during a repo cleanup pass. Given only 6 groups of thousands of rows each, the leak was numerically negligible (order 1/10,000 per row), but I fixed the code for consistency with the primary encoding's discipline. I did **not** retrain the saved models against this fix — see §7 for why.
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
| 6 | **Stacked ensemble (final)** | **0.0263** | **96.57%** | Ridge meta-learner on 3 base models' OOF predictions |

**Why gradient boosted trees, not a neural net:** tabular data with mostly low-to-moderate cardinality categoricals and a bounded skewed target — trees handle this natively (no need to normalize/embed features), need less tuning, and are the standard strong baseline for this data shape. I did not try a neural net; if asked "would a NN help here," the honest answer is "probably not enough to justify the complexity, given how much of the signal is already captured by geohash + interactions — but I haven't tested it."

**Why three different tree models, not three seeds of the same model:** the whole point of stacking is combining models that make *different* errors. XGBoost (depth-wise, regularized), LightGBM (leaf-wise, very deep/high-capacity here — `num_leaves=237`, `max_depth=9`), and CatBoost (ordered boosting, built-in categorical handling) have different inductive biases, so their errors are less correlated than three runs of the same algorithm would be.

**Hyperparameters (from `models/best_params.json`, Optuna, 200 trials/model, 3-fold CV per trial):**
- **XGBoost**: `n_estimators=1266, lr=0.0403, max_depth=9, subsample=0.816, colsample_bytree=0.682, min_child_weight=3, gamma=0.0015, reg_alpha=0.116, reg_lambda=4.397`
- **LightGBM**: `n_estimators=2643, lr=0.0304, num_leaves=237, max_depth=9, subsample=0.898, colsample_bytree=0.710, min_child_samples=5, reg_alpha=0.004, reg_lambda=3.447`
- **CatBoost**: `iterations=1913, lr=0.257, depth=8, l2_leaf_reg=4.608, bagging_temperature=0.598, random_strength=0.822`

**Why Optuna and not grid/random search:** Optuna uses Bayesian optimization (TPE sampler by default) — it uses the results of earlier trials to pick more promising regions of the search space, which matters here because some params (e.g. `n_estimators` up to 3000, `num_leaves` up to 500) create a huge search space where grid search would be wasteful and pure random search would need many more trials to find the same quality of optimum.

**Why Ridge (not XGBoost-on-top-of-XGBoost, not a simple average) as the meta-learner:** the meta-learner only sees 3 inputs — one prediction per base model. A complex model on 3 features would overfit; Ridge is linear with L2 regularization, and its coefficients are directly interpretable as "how much weight does each base model get." Actual fitted coefficients: **XGB −0.23, LGB 0.63, CAT 0.60** — the negative XGBoost weight is worth being ready to explain (see §7).

**Why 5-fold CV for the final stack but only 3-fold inside the Optuna search:** the 3-fold CV during search is a speed/quality tradeoff — 200 trials × 3 models × 3-fold is already 1,800 model fits; going to 5-fold there would cost 67% more compute for a hyperparameter *search* where you don't need the absolute best CV estimate, just a good relative ranking between trial configs. The final 5-fold is the number that actually gets reported, where the extra fold count buys a lower-variance estimate.

---

## 6. Evaluation

**Metric: RMSE**, primary, because it's on the same scale as the target (interpretable directly as "typical error in demand units") and penalizes large misses more than MAE would — appropriate here because under/over-predicting a genuine high-demand spike is worse than being off on a routine low-demand slot. **R²** reported alongside as a scale-free "how much variance explained" sanity check. **MAPE was not used** — with `demand` values near zero for the median row (0.048), MAPE blows up/becomes unstable on small denominators; this is a real gap, see §7.

**All headline numbers are out-of-fold (OOF)**, from 5-fold CV — not train-set fit, and not a leaked validation set. Concretely: each of the 5 folds' predictions come from a model that never saw that fold during training, so stitching all 5 folds' predictions together and scoring against the true `demand` gives an unbiased estimate of generalization performance *on data shaped like the training set*. Because `test.csv` has no labels (Kaggle-style holdout), the OOF score is the *only* performance estimate available — there's no way to independently verify it against a true held-out test score.

**Residuals** (from the OOF stacked predictions, `step10_part2_training.ipynb`): mean ≈ **−0.0001** (no systematic bias), std ≈ **0.0256**, range **−0.25 to +0.30**. The near-zero mean is a good sign — the model isn't systematically over- or under-predicting. The 0.25–0.30 tail means there are individual rows the model misses badly (some real high-demand spikes get under-predicted) — expected for a right-skewed target where the rare high-demand tail is intrinsically harder to hit exactly.

---

## 7. Limitations (honest version)

Things I would not want to be caught not knowing about my own project:

1. **The 26-feature set fails VIF badly, and I kept it anyway.** 16 of 23 extended features exceed VIF 10 (some, like `latitude`, exceed 10,000). My documented reasoning: most of the high-VIF features are literal products of other retained features (`road_capacity`, every `*_x_*` interaction), so high VIF is structurally expected, not a modeling mistake. It matters for linear models — my Ridge meta-learner never sees these 26 features directly, only the 3 base models' predictions — and doesn't matter for split-based tree models, which are invariant to linear redundancy. I did *not* verify this claim by actually training a VIF-trimmed linear baseline for comparison — that's a real gap, not just a rhetorical answer. If pushed: "I'd want to prove this by actually training a linear model on the full 26 vs. the VIF-trimmed 10 and showing the trimmed one is *not* meaningfully worse, rather than asserting it from theory."

2. **A leakage fix exists in code but isn't reflected in the saved models.** I found that `geohash_prefix_enc` used plain (non-LOO) mean encoding — inconsistent with the primary geohash encoding's LOO discipline — and fixed it. I chose *not* to retrain the full pipeline against this fix: while attempting a full retrain, I discovered the current environment's newer XGBoost/LightGBM/CatBoost versions produce results ~3x better than the documented numbers in a way I could not fully explain (early stopping behaving very differently between versions) — swapping in an unverified, suspiciously-good result would have been worse than leaving a known, tiny, well-understood gap. This is the single most honest thing to say if asked "did retraining after your fix change the results": **"I don't fully trust my current retraining environment yet, so no — the fix is real and correct, but I'd want to reproduce the original result in a pinned environment before trusting a retrain."**

3. **Mild optimism bias from using the same fold for early stopping and scoring.** Each base model's `eval_set` during CV is the same held-out fold used for that fold's OOF prediction — meaning early stopping is tuned to minimize error on the exact data later used to score it. This is common practice (not a bug), but it does mean the reported OOF RMSE is very slightly optimistic vs. a stricter nested-CV setup with a third split reserved purely for stopping decisions.

4. **`day` has only 2 distinct values (48, 49).** This isn't a real time series — there's no way to validate the model generalizes across different days, seasons, or long-term trends. Anything said about "how the model handles day-of-week effects" would be speculation; it simply hasn't seen more than 2 days.

5. **The data may well be synthetic/competition data, not real traffic telemetry.** Signs: the near-perfect regularity of geohash sampling (~96 rows per cell, one per 15-minute slot), the suspiciously narrow geographic bounding box, and how cleanly the strong features (geohash, road type) separate the target. I have not independently verified the data's provenance and would say so plainly if asked, rather than imply operational realism it may not have.

6. **No MAPE, and demand near zero makes it a bad fit anyway.** If a stakeholder specifically needs "percentage error" framing (common in ops contexts), RMSE/R² don't directly answer that, and MAPE is numerically unstable given the target's right skew toward near-zero values.

7. **No true holdout test score.** Because `test.csv` is unlabeled, I can't report a number I'm fully confident wasn't at all influenced by iterating on the pipeline against OOF CV — a classic risk in any Kaggle-style setup where you tune against the only evaluation signal you have.

**What I'd do with more time/data:**
- Geohash × hour target encoding (per-cell, per-hour mean) instead of per-cell only, to capture location-specific hourly patterns directly rather than through the `geo_hour_sin` product proxy.
- An actual VIF-trimmed linear baseline to test claim #1 above empirically instead of asserting it.
- Re-run the full training pipeline in a version-pinned environment (`requirements.txt` now exists for exactly this reason) to get a trustworthy before/after comparison for the `geohash_prefix_enc` fix.
- If this were real operational data: validate on a genuinely held-out future time period (a true walk-forward split), not just k-fold CV on a 2-day window.

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
Two things: (1) I found and fixed a small leakage inconsistency post-hoc but chose not to re-verify it end-to-end because retraining in my current environment gave me numbers I don't trust — I'd rather say "I didn't verify this" than present an unverified number as fact. (2) The VIF-features-kept decision is reasoned from first principles but not empirically tested against a trimmed baseline. Both are documented, not hidden.

**Q15: How would you extend this to a real production system?**
Add a true walk-forward temporal validation (not just k-fold, since 2 days of data can't tell you about drift), add monitoring for feature distribution shift (especially new geohash cells), and probably swap the two-stage manual Optuna-then-retrain notebook flow for a single tracked pipeline (e.g., with MLflow) so hyperparameters and metrics are versioned together instead of living in a JSON file and a markdown doc respectively.
