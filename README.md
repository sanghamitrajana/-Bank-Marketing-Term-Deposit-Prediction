# -Bank-Marketing-Term-Deposit-Prediction
Bank Marketing Subscription Prediction using Machine Learning Classification Models
# Bank Marketing Analytics: Term Deposit Prediction

An end-to-end analytics project on 45,211 bank marketing records. It covers data quality auditing, root cause analysis, statistical testing, predictive modelling and SQL-based lead ranking to understand **who subscribes to a term deposit** and **which customers a campaign should target first**.

## Business Problem
A bank runs telemarketing campaigns to sell term deposits. Calling every customer is costly, so the goal is to:
1. Understand which factors are associated with subscription.
2. Predict the likelihood of subscription.
3. Rank leads so the team can contact the most promising customers first.

## Key Results
| Area | Result |
|---|---|
| Data audited | 45,211 records |
| Multicollinearity | VIF reduced from above 108 to below 12 |
| Strongest association | Previous campaign success linked to **8.5x higher** subscription odds (logistic regression odds ratio) |
| Lead ranking (SQL) | Top 20% of scored leads captured **59.2%** of all subscribers |

## Tech Stack
Python (Pandas, NumPy, scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn) | MySQL

## Approach
1. **Data audit and quality checks**: validated data quality in Python and MySQL and documented data definitions (data dictionary).
2. **Root cause analysis**: investigated data leakage and multicollinearity. [Add one line: which feature caused leakage and what you did about it.]
3. **Statistical analysis**: Chi-Square tests for categorical variables, Mann-Whitney U tests for numeric variables, and logistic-regression odds ratios.
4. **Modelling**: compared classification models, using McNemar's test to check whether performance differences were significant. Interpreted XGBoost predictions with SHAP.
5. **SQL lead ranking**: scored leads and ranked them in MySQL to measure how many subscribers the top-scored segment captures.

## Project Structure
```
[Replace with your actual files/folders, for example:]
├── data/            # dataset (or link to source)
├── notebooks/       # analysis notebook(s)
├── sql/             # MySQL queries for audit and lead ranking
└── README.md
```

## How to Run
```bash
git clone https://github.com/sanghamitrajana/Bank-Marketing-Term-Deposit-Prediction.git
cd Bank-Marketing-Term-Deposit-Prediction
pip install -r requirements.txt
# open the notebook and run all cells
```

## Data
[https://archive.ics.uci.edu/dataset/222/bank+marketing]

## Author
**Sanghamitra Jana**, M.Sc. Statistics, Banaras Hindu University
[LinkedIn](https://www.linkedin.com/in/sanghamitrajana) | sanghamitra.bhu.stat27@gmail.com

