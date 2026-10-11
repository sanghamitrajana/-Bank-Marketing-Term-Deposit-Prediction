# Dial Smarter, Not Harder
### Predicting Bank Term-Deposit Subscriptions with Leakage-Free ML, Statistical Validation and a Business-Ready Call Strategy

**Stack:** Python 3.12 · pandas · NumPy · SciPy · statsmodels · scikit-learn · XGBoost · TensorFlow/Keras · SHAP · Matplotlib/Seaborn · **SQL (MySQL)** · Jupyter
**Domain:** Retail banking, direct marketing (term deposits)
**Data:** UCI Bank Marketing, `bank-full.csv`, 45,211 client contacts

---

## 1. The Answer in 30 Seconds

A bank's call centre was dialling tens of thousands of clients, and only **11.7%** said yes. This project builds a model that ranks clients by their likelihood to subscribe *before* the call is made, and proves, with statistical tests rather than a leaderboard, which findings can be trusted.

| What the stakeholder asks | What the analysis says |
|---|---|
| Who should we call first? | Calling only the **top 20%** of clients ranked by the model reaches **59.2%** of all eventual subscribers, at **34.6%** precision versus an 11.7% baseline (**about 3x call efficiency**). |
| What is the strongest signal? | A previous campaign **success** gives **8.5x higher odds** of subscribing again. Logistic Regression and XGBoost found this independently. |
| When should we call? | March, September, October and December convert at roughly **44-52%**. May to August, where most of the effort goes, converts at roughly **7-11%**. |
| How many times should we call? | Conversion falls after about **3 contacts**. Repeated calling is associated with lower success. |
| Can we trust the model? | Only because one tempting column (`duration`) was **deliberately removed**. It is known only after the call ends and inflated every model's ROC-AUC, by **0.13-0.15** for six of the seven models (0.07 for Naive Bayes). |
| Which algorithm should we deploy? | SVM and XGBoost are **statistically indistinguishable** (McNemar p = 0.386). XGBoost is chosen on practical grounds: speed, native imbalance handling and SHAP support. |

---

## 2. Business Problem and Objective

**Problem.** Outbound telemarketing is expensive. Every wasted call costs agent time and client goodwill, and the base rate of success is roughly 1 in 9.

**Objective.** Produce a ranked, pre-call lead list and the evidence needed to defend it, not just an accuracy score.

**Decision this supports.** Which clients to call, in which months, and how many times.

---

## 3. Dataset and Data Dictionary

45,211 rows, 17 columns, 0 duplicate rows. Target `y` is binary: 88.3% no (39,922) and 11.7% yes (5,289).

| Column | Type | Meaning | Notes |
|---|---|---|---|
| age | numeric | Client age | Non-linear relationship with `y` (see Section 7) |
| job | categorical | Occupation | 288 unknowns |
| marital | categorical | Marital status | |
| education | categorical (ordinal) | primary < secondary < tertiary | 1,857 unknowns |
| default | binary | Has credit in default | |
| balance | numeric | Average yearly balance (EUR) | Heavily right-skewed |
| housing | binary | Has housing loan | |
| loan | binary | Has personal loan | |
| contact | categorical | Contact channel | 13,020 unknowns |
| day_of_week | numeric | Last contact day, values 1-31 | Despite the name, this behaves as day of month |
| month | categorical | Last contact month | Strongest categorical predictor |
| duration | numeric | Call length in seconds | **Leakage. Excluded from the deployable model** |
| campaign | numeric | Contacts in this campaign | Diminishing returns after about 3 |
| pdays | numeric | Days since previous campaign contact | **-1 = never contacted** (36,954 rows, 81.7%) |
| previous | numeric | Contacts before this campaign | |
| poutcome | categorical | Outcome of previous campaign | 36,959 unknowns. Top predictor |
| y | binary | Subscribed to term deposit | Target |

**Engineered columns:** `was_previously_contacted`, `age_group`, `campaign_intensity`, `contact_month_quarter`, `has_any_loan`, `demographic_profile` (details in Section 8).

---

## 4. Data Quality Checklist

| Check | Finding | Action |
|---|---|---|
| Duplicate rows | 0 | None needed |
| True nulls | The loader returned `NaN` where the source says "unknown" (job 288, education 1,857, contact 13,020, poutcome 36,959) | Restored to an explicit `"unknown"` category. Not dropped and not mode-imputed |
| Sentinel values | `pdays = -1` means "never contacted", not a day count (81.7% of rows) | Handled explicitly. See the root-cause log |
| Class balance | 88.3% / 11.7% | Class-weighting, threshold tuning, and Precision/Recall/F1/ROC-AUC instead of accuracy |
| Distribution shape | All numeric features non-normal (Shapiro-Wilk) | Non-parametric testing (Mann-Whitney U) |
| Cross-field consistency | `pdays = -1` count (36,954) equals `previous = 0` count (36,954) | Confirms the sentinel interpretation |
| Train/test comparability | Stratified split keeps 88.3 / 11.7 in both halves | Verified after the split |
| Post-encoding redundancy | Initial VIF values in the hundreds of millions | Four rounds of fixes, final VIF all below 12 |

**Why "unknown" was kept.** "The bank has no record of this" is information. Imputing it would invent data, and deleting `poutcome` (81.7% unknown) would remove the strongest predictor in the dataset.

---

## 5. Statistical Validation (Before Any Modelling)

Run on the full dataset, because this stage is exploratory and fits nothing. All fitted steps (encoding, scaling, VIF, models) used the training split only.

| Test | Used for | Result |
|---|---|---|
| **Chi-Square test of independence** | Each categorical feature vs `y` | All 9 significant. Strongest: `poutcome` (chi2 = 4,391.5) and `month` (chi2 = 3,061.8). Weakest: `default` (chi2 = 22.2, p = 2.5e-6) |
| **Shapiro-Wilk + Q-Q plots** | Normality of numeric features | All non-normal (p approx 0) |
| **Levene's test** | Equal variance between yes/no groups | Variances unequal for every numeric feature |
| **Mann-Whitney U** | Numeric feature vs `y` (the non-parametric choice, since the t-test's normality and equal-variance assumptions both failed) | Significant for balance, duration, campaign, pdays, previous. **Not significant for age (p = 0.063)** |

**Reading the age result.** Age is not irrelevant. Its relationship is non-monotonic, with older clients (60+) over-subscribing relative to middle-aged ones, and a rank-based test cannot detect that shape. This finding drove the decision to bucket age.

**Caveat.** With 45,211 rows, nearly any difference is "significant". The tests establish that a relationship exists, not how large it is. The effect-size view comes from the models' odds ratios and SHAP values.

---

## 6. Exploratory Findings

1. **The imbalance is severe**, so accuracy is a misleading metric (a "predict no" model scores 88.3% and finds no customers).
2. **Call duration separates yes from no almost perfectly.** That looked too good to be true, and it was (Section 9).
3. **Contact fatigue:** subscription likelihood drops sharply beyond about 3 contacts.
4. **Seasonality is inverted.** Volume is highest in May to August, but conversion is highest in March, September, October and December. Small-volume months (for example December, 214 contacts) carry wider uncertainty and should be read with that in mind.
5. **`poutcome = success` is the only previous-outcome category where "yes" rivals "no".**
6. **Correlation screen:** all numeric pairs are near zero except `pdays` and `previous` (0.45), which was re-examined formally through VIF.

*(Charts are in `images/` and in the illustrated report.)*

---

## 6A. SQL Analysis Layer (MySQL)

The same dataset is also loaded into a MySQL database so the data quality checks and business questions can be answered, and reproduced, in SQL. The script is `sql/bank_marketing_analysis_mysql.sql` (a setup block plus 22 statements for MySQL 8.0+, run in MySQL Workbench). A portable SQLite version of the same queries, `sql/bank_marketing_analysis.sql`, needs no server.

| Area | What the queries do | SQL features used |
|---|---|---|
| **Data quality** (1.1-1.6) | Row count, duplicate groups, NULL counts, `unknown` share per column, a 7-rule validation checklist, `pdays = -1` sentinel count and a `pdays`/`previous` cross-field consistency check | `GROUP BY`/`HAVING`, `CASE`, `UNION ALL` |
| **Baseline** (2.1) | Class balance with percentage share | Window function over an aggregate (`SUM() OVER ()`) |
| **Conversion analysis** (3.1-3.7) | Monthly conversion with volume and a small-sample flag, effort-versus-yield by month, lift by previous outcome, contact-fatigue buckets, age buckets, occupation ranking, debt-status segments | CTEs, `RANK()`, `CROSS JOIN`, bucketing with `CASE` |
| **Reusable views** (4.1-4.4) | `vw_bank_features` re-creates the Python feature logic (prior-contact flag, age group, campaign intensity, loan flag) in SQL. `vw_conversion_by_month` is a reporting-ready summary | `CREATE OR REPLACE VIEW`, querying views |
| **Model output** (5.1-5.4) | Decile gain and lift table from the XGBoost scores, the top-20% rule, predicted versus actual conversion by month, and a look at actual subscribers the model ranked in the bottom half | `NTILE()`, running totals with `SUM() OVER (ORDER BY ...)`, joins |

**Why SQL here.** Query 3.2 shows how much of the calling effort lands in each month next to how many subscribers each month actually yields. Query 5.1 turns the model into a decile table a campaign manager can read, and 5.2 expresses the top-20% rule in SQL. The `unknown`-category and sentinel checks (1.4-1.6) make the data quality assumptions explicit and re-runnable.

**How to load it.**

1. Export the cleaned data from the notebook:

```python
cols = ['age','job','marital','education','default','balance','housing','loan',
        'contact','day_of_week','month','duration','campaign','pdays','previous',
        'poutcome','y']
out = df_raw[cols].copy()                       # df_raw = features + target as loaded from UCI
for c in ['job','education','contact','poutcome']:
    out[c] = out[c].fillna('unknown')           # the loader returns "unknown" as blank
out = out.rename(columns={'default': 'credit_default'})   # DEFAULT is a reserved SQL word
out.insert(0, 'client_id', out.index)
out.to_csv('bank_marketing_clean.csv', index=False)

pd.DataFrame({'client_id': y_test.index, 'prob_yes': y_proba_xgb,
              'actual_yes': (y_test == 'yes').astype(int).values}).to_csv('lead_scores.csv', index=False)
```

2. In MySQL Workbench, run the SETUP block of the script (it creates `bank_marketing_db` and two empty tables).
3. Import each CSV with the Table Data Import Wizard, using the existing table.
4. Check the load: `bank_marketing` should hold 45,211 rows (39,922 no, 5,289 yes) and `lead_scores` 9,043 rows.
5. Run the queries one statement at a time.

**Dialect note.** The queries use standard constructs (CTEs, window functions, `CASE`, views), so porting to SQL Server or Snowflake needs only small changes (view syntax and identifier quoting). No stored procedures are included.

---

## 7. Why Age Was Handled the Way It Was

Mann-Whitney found no simple difference, the violin plot showed a heavier older tail for subscribers, and SHAP later showed age pushing predictions in both directions. Three lines of evidence point to a non-linear effect, so age was bucketed (`young` up to 30, `middle` 31-45, `senior` 46-60, `elderly` 60+). Logistic Regression cannot fit a bend with one coefficient, so bucketing lets it capture the effect. Tree models can find such splits themselves.

---

## 8. Feature Engineering

| Feature | Logic | Reason |
|---|---|---|
| `was_previously_contacted` | `pdays != -1` | Separates the sentinel meaning from real day counts |
| `age_group` | 4 buckets | Captures the non-linear age effect |
| `campaign_intensity` | low (up to 2) / medium (3-5) / high (6+) | Captures the diminishing-returns threshold |
| `contact_month_quarter` | month mapped to Q1-Q4 | Coarser seasonal grouping (later dropped as redundant) |
| `has_any_loan` | housing OR personal loan | Both pointed the same direction |
| `demographic_profile` | `age_group` x `job`, with combinations under 1% of rows folded into `other_profile` | Tests whether specific age-job combinations behave differently. Reduced from a possible 44 categories to 25. Chi-Square p approx 0 |

Dataset shape after engineering: 45,211 x 23. Every engineered feature was checked for statistical relevance before being kept.

---

## 9. Root Cause Analysis: Call-Duration Data Leakage

**Symptom.** Models trained on all columns scored ROC-AUC of 0.88-0.92, in line with most public work on this dataset.

**Root cause.** `duration` is the length of the call. It exists only after the call ends, but the decision the model supports (who to dial) is made before. Long calls are essentially engaged clients saying yes, so the feature restates the outcome.

**Verification.** Every model was trained twice on identical settings, with and without `duration`. The only difference between the two runs is that column.

| Model | ROC-AUC with duration | ROC-AUC without | Drop |
|---|---|---|---|
| Logistic Regression | 0.908 | 0.773 | 0.135 |
| Decision Tree | 0.880 | 0.753 | 0.127 |
| Random Forest | 0.916 | 0.785 | 0.131 |
| SVM | 0.911 | 0.767 | 0.144 |
| XGBoost | 0.923 | 0.776 | 0.147 |
| Naive Bayes | 0.819 | 0.749 | 0.070 |
| ANN | 0.914 | 0.771 | 0.143 |

The drop appears across linear, tree, kernel and neural models. That consistency is the evidence that this is leakage and not a quirk of one algorithm.

**Resolution.** The deployable pipeline excludes `duration`. The with-duration set is kept only as a documented upper-bound benchmark.

---

## 10. Pipeline Design

1. **Stratified 80/20 split** (36,168 train / 9,043 test), done after exploratory testing and before any fitted step.
2. **Encoding:** one-hot for nominal features (`drop='first'`, `handle_unknown='ignore'`), ordinal for `education` (primary < secondary < tertiary, unknown last). Fitted on train only.
3. **Multicollinearity (VIF) on the encoded training set.** See the root-cause log below.
4. **Scaling (MinMaxScaler, train-fitted)** for SVM and ANN only. Tree models use unscaled data.
5. **Imbalance:** `class_weight='balanced'` (sklearn models), `scale_pos_weight = 7.55` (XGBoost), a class-weight dictionary (Keras). SMOTE was deliberately not used, to avoid synthetic clients.
6. **Final feature set:** 55 columns with duration, 54 without.

---

## 11. Models and Results

**Why each model.** Logistic Regression (interpretable baseline, odds ratios), Decision Tree (readable rules), Random Forest (variance reduction), SVM (non-linear boundary), XGBoost (sequential boosting, strong on tabular data), plus an ANN and Naive Bayes as extra comparison points.

### Deployable set: without `duration`

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.760 | 0.275 | 0.641 | 0.385 | 0.773 |
| Decision Tree | 0.824 | 0.343 | 0.550 | 0.422 | 0.753 |
| Random Forest | 0.799 | 0.314 | 0.609 | 0.415 | 0.785 |
| SVM | 0.813 | 0.334 | 0.597 | 0.428 | 0.767 |
| XGBoost | 0.810 | 0.330 | 0.605 | 0.427 | 0.776 |
| Naive Bayes | 0.840 | 0.356 | 0.452 | 0.398 | 0.749 |
| ANN | 0.806 | 0.321 | 0.594 | 0.417 | 0.771 |

### Benchmark only: with `duration`

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.841 | 0.411 | 0.828 | 0.549 | 0.908 |
| Decision Tree | 0.833 | 0.397 | 0.831 | 0.537 | 0.880 |
| Random Forest | 0.829 | 0.396 | 0.873 | 0.545 | 0.916 |
| SVM | 0.846 | 0.419 | 0.827 | 0.557 | 0.911 |
| XGBoost | 0.855 | 0.437 | 0.841 | 0.575 | 0.923 |
| Naive Bayes | 0.854 | 0.404 | 0.526 | 0.457 | 0.819 |
| ANN | n/a | 0.393 | 0.854 | 0.538 | 0.914 |

### Why these metrics
Accuracy is misleading at 88/12. **Recall** measures how many real subscribers are found, **precision** measures how many calls are wasted, **F1** balances the two, and **ROC-AUC** measures ranking quality independent of any cutoff.

### Is the "best" model really best?

Bootstrapped 95% confidence intervals for F1 (1,000 resamples of the test set):

| Model | F1 | 95% CI |
|---|---|---|
| Logistic Regression | 0.384 | 0.363 - 0.406 |
| Decision Tree | 0.422 | 0.400 - 0.447 |
| Random Forest | 0.415 | 0.394 - 0.437 |
| SVM | 0.429 | 0.406 - 0.451 |
| XGBoost | 0.427 | 0.405 - 0.451 |
| Naive Bayes | 0.398 | 0.372 - 0.424 |

SVM, XGBoost, Decision Tree and Random Forest overlap almost completely.

**McNemar's test (SVM vs XGBoost, same 9,043 test clients):** 6,892 both correct, 1,252 both wrong, 463 only SVM correct, 436 only XGBoost correct. Statistic = 0.752, **p = 0.386**. The models cannot be separated, so the final choice rests on operational factors, not on a decimal place.

---

## 12. Threshold Tuning and Business Simulation

| Operating point (XGBoost, no duration) | Precision | Recall | F1 |
|---|---|---|---|
| Default cutoff 0.50 | 0.330 | 0.605 | 0.427 |
| F1-optimal cutoff 0.664 | 0.469 | 0.466 | 0.467 |
| **Call the top 20% by predicted probability (cutoff approx 0.516)** | **0.346** | **0.592** | n/a |

The top-20% rule is the figure a campaign manager can act on: **59.2% of subscribers reached by dialling one client in five.** The bank can slide along this precision-recall trade-off according to its own cost of a wasted call versus a missed sale.

---

## 13. Explainability

- **Logistic Regression odds ratios:** `poutcome_success` 8.57x; `month_mar` 2.30x, `month_oct` 2.26x, `month_sep` 1.85x; `contact_unknown` 0.31x; `month_nov` 0.43x; `has_any_loan` 0.67x.
- **XGBoost gain importance:** `poutcome_success` 0.264, more than 2.5x the next feature (`contact_unknown`, 0.091).
- **SHAP summary:** confirms direction. High `campaign` and unknown contact push toward "no", `poutcome_success` is the largest push toward "yes", and `age` shows a mixed, non-monotonic spread.
- **Method disagreement worth noting:** `demographic_profile` shows up in the Logistic Regression odds ratios but not in XGBoost's top 15. Trees find age-job splits on their own, so the hand-built interaction adds little for them.

Two structurally different methods agree on the top driver, which makes that finding far more trustworthy than either one alone.

---

## 14. Recommendations for the Business

1. **Rank and call the top 20%** of leads by predicted probability (about 59% of subscribers for 20% of the dialling effort).
2. **Prioritise past successes.** They convert at roughly 8.5x the odds.
3. **Cap attempts at about 3 per client.** Returns diminish after that.
4. **Rebalance the calendar** toward March, September, October and December. Validate with a pilot first, since these months have low historical volume and may not scale linearly.
5. **Improve capture of contact details.** An unknown contact channel cuts the odds of subscribing to roughly a third (odds ratio 0.31).
6. **Governance:** retain SHAP outputs so any individual lead score can be explained to a reviewer.

---

## 15. Root-Cause Log

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | Model scores far above realistic levels | `duration` leaks the outcome | Parallel with/without runs, excluded from deployable model |
| 2 | VIF in the hundreds of millions | Dummy-variable trap (`default_no` = 1 - `default_yes`) | `OneHotEncoder(drop='first')` |
| 3 | `month_*` and `contact_month_quarter_*` at VIF up to 1e15 | The quarter feature was derived from month | Dropped the quarter columns |
| 4 | `job_*`, `age_group_*` and `demographic_profile_*` all inflated (VIF 100-1,600) | The interaction was built from the same columns | Kept only `demographic_profile` |
| 5 | `was_previously_contacted` and `poutcome_unknown` at VIF about 1,088 | Both flag "never contacted before" (36,954 vs 36,959 rows) | Dropped the flag, kept the richer `poutcome` |
| 6 | SVM F1 of 0.249 versus about 0.42 for peers | Unscaled features: `balance` dominated the RBF distance | MinMaxScaler fitted on train, F1 recovered to 0.428 |
| 7 | "0 categorical columns" found | Pandas returned the newer `str` dtype, not `object` | Selected with `select_dtypes(include=['object','category','string'])` |

---

## 16. How This Maps to a Banking Data Analyst Role

| Role expectation | Evidence in this project |
|---|---|
| Ensure data quality, prepare data quality checklists | Section 4 checklist: duplicates, nulls, sentinels, cross-field consistency, split integrity |
| Root cause analysis for each issue | Sections 9 and 15: seven documented symptom-to-cause-to-fix cases |
| Analyse data, identify trends, give actionable insights | Sections 6 and 14: seasonality inversion, contact-fatigue threshold, lead-ranking rule |
| Document data dictionaries and processes | Section 3 data dictionary, plus a fully documented pipeline |
| Stakeholder communication | Section 1 and Section 14 translate model output into decisions, with caveats stated plainly |
| Python for automation and standardisation | Reproducible, split-safe pipeline (fit on train, transform on test) |
| SQL development (queries, views) | Section 6A: 22 MySQL statements covering data quality checks, CTEs, window functions (`RANK`, `NTILE`, running totals), joins and two reusable views |
| Banking domain | Retail direct-marketing banking, including lead prioritisation and explainable scoring |

**Not covered by this project:** SAS, Snowflake, SQL Server stored procedures, Tableau/Power BI/Sigma, Power Automate and ETL scheduling. The SQL here is run on MySQL, not SQL Server or Snowflake, and the syntax differences are small but real. A natural extension is pointing a BI tool at the `vw_conversion_by_month` view and the scored `lead_scores` table to build the dashboard layer.

---

## 17. Limitations and Next Steps

- Single stratified split with bootstrapped intervals. **K-fold cross-validation and GridSearchCV were not run**, and hyperparameters are sensible defaults rather than tuned values.
- No multiple-testing correction (for example Bonferroni) across the Chi-Square and Mann-Whitney runs. Conclusions would not change at these p-values, but it is the more rigorous standard.
- No effect-size measure for Chi-Square (Cramer's V), and no PR-AUC alongside ROC-AUC, though PR-AUC suits an 11.7% positive rate better.
- Logistic Regression's log-odds linearity (Box-Tidwell) and influential-point checks (Cook's distance) were not formally executed. VIF was.
- `balance` was not log-transformed despite heavy skew. An IQR-capping experiment for the distance-based models (Logistic Regression, SVM, ANN) is designed but not yet run.
- The data covers 2008-2010 from one Portuguese bank. Behaviour and seasonality may differ today, so any recommendation should be piloted before roll-out.
- A Streamlit scoring app is a planned extension.

---

## 18. Reproduce It

```bash
python -m venv venv            # use Python 3.12 (TensorFlow does not yet support 3.14)
venv\Scripts\activate          # Windows; use source venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

`requirements.txt`: pandas, numpy, scikit-learn, xgboost, tensorflow, matplotlib, seaborn, ucimlrepo, scipy, statsmodels, shap, ipykernel

Data loads directly from UCI with `fetch_ucirepo(id=222)`. No manual download is needed. Run the notebook top to bottom (Run All) so that every variable is rebuilt in order. The two CSVs used for the MySQL import (Section 6A) are generated files, so add `*.csv` to `.gitignore` rather than committing them.

**Suggested repo layout**
```
├── README.md
├── Bank_Marketing_Project_Report.md   # illustrated long-form report
├── requirements.txt
├── sql/
│   ├── bank_marketing_analysis.sql        # portable SQLite version, 22 queries
│   └── bank_marketing_analysis_mysql.sql  # MySQL 8.0+ setup block + 22 queries
├── images/                            # saved charts (savefig)
└── <your notebook>.ipynb
```

---

**Author:** *[Your name]* · M.Sc. Statistics · *[LinkedIn / email / GitHub]*
**Data source:** Moro, Cortez and Rita (2014), "A data-driven approach to predict the success of bank telemarketing", *Decision Support Systems*. UCI Machine Learning Repository, DOI 10.24432/C5K306 (CC BY 4.0).
