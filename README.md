# SLA Breach Prediction: XGBoost Early-Warning Dashboard for Operations Teams

A machine learning project that predicts which employees are likely to **breach their SLA** (service-level agreement), so a manager can step in early. It includes a model comparison, a tuning comparison, an explanation of what drives breaches, and a Streamlit dashboard with a live predictor.

![Model performance](images/01_model_performance.png)

<details>
<summary>More screenshots</summary>

![Risk flags](images/02_risk_flags.png)
![Predict SLA](images/03_predict_sla.png)
![EDA and insights](images/04_eda_insights.png)
![Data overview](images/05_data_overview.png)

</details>

**Demo video:** [link to your video here]  |  **Live app:** [link to Streamlit Cloud here, if deployed]

---

## Why this project matters

In BPO, KPO, support and content-moderation teams, every employee works against SLA targets such as turnaround time, quality and task completion. Managers usually find out about a breach after it has happened.

This project shows how a model could give an **early warning**:

- **Prioritise manager attention.** A ranked list of at-risk employees shows who to check first.
- **Find the causes.** Feature importance shows which measures (completion ratio, turnaround time, quality, rework) push risk up.
- **Be consistent.** Every employee is scored the same way, instead of by gut feeling.

This is a portfolio project on a synthetic dataset. No business savings were measured, and none are claimed.

## Dataset

10,000 records with 12 columns: `Employee_ID`, `Team` (Ads, Moderation, QA, Support), `Shift` (Day, Night), `Tasks_Assigned`, `Tasks_Completed`, `Avg_TAT_Hours`, `Error_Rate`, `Quality_Score`, `Escalations`, `Rework_Count`, `Experience_Years`, and the target `SLA_Status` (Met or Breached).

- Class balance: 60% Breached, 40% Met.
- 1,099 rows have missing `Avg_TAT_Hours`, `Error_Rate` and `Quality_Score` together.
- `Employee_ID` repeats, so rows are not one per employee. The column was dropped.

## What I did

1. **EDA:** class balance, team and shift effects, correlations, outliers.
2. **Split first, then impute.** The data was split 80/20 (stratified) before filling missing values with **training-set medians**, which avoids data leakage. The original notebook imputed before splitting.
3. **Feature engineering:** `completion_ratio` (completed / assigned) and a `was_missing` flag. Missing rows were more often "Met" (15% vs 8%), so missingness itself carries a signal.
4. **Model comparison** with 5-fold cross-validation on the training set.
5. **Hyperparameter tuning** of XGBoost with three methods and equal budgets.
6. **One final test** on 2,000 unseen records.
7. **Explainability** with feature importance, plus a check of the score without the task columns.
8. **Business action:** a risk score, with employees above 0.7 flagged for review.

## Tools and models

| Area | What was used |
|---|---|
| Language and data | Python, Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Models | Logistic Regression, K-Nearest Neighbours, Random Forest, XGBoost |
| Tuning | GridSearchCV, RandomizedSearchCV, Optuna |
| Evaluation | Macro-F1 (the classes are 60/40), classification report, confusion matrix, stratified cross-validation |
| App | Streamlit |
| Version control | Git and GitHub |

## Results

| Stage | Macro-F1 |
|---|---|
| Original notebook: KNN on raw features | 0.83 accuracy (0.90 after GridSearch) |
| Logistic Regression (5-fold CV) | 0.823 |
| KNN (5-fold CV) | 0.863 |
| Random Forest (5-fold CV) | 0.981 |
| XGBoost (5-fold CV) | 0.981 |
| XGBoost tuned, GridSearchCV (3-fold CV) | 0.976 |
| XGBoost tuned, RandomizedSearchCV (3-fold CV) | 0.978 |
| XGBoost tuned, Optuna (3-fold CV) | 0.980 |
| **Final test, 2,000 unseen records: Optuna-tuned XGBoost (notebook)** | **0.99** |
| Final test without task columns (notebook) | 0.93 |
| Final test: default XGBoost (retrained inside the dashboard) | 0.987 |
| Final test without task columns (dashboard) | 0.918 |

The notebook uses the Optuna-tuned parameters. The dashboard retrains XGBoost with default parameters so it can run from the CSV alone, which is why its scores differ slightly. That gap is another sign that tuning added almost nothing.

Notebook final test details: Breached precision 1.00 and recall 0.98; Met precision 0.98 and recall 0.99. With a 0.7 risk cutoff, 1,180 test records were flagged and 99.9% of them had actually breached.

## Key findings

- **Tree models beat linear and distance-based models** by 0.12 to 0.16 macro-F1. Random Forest and XGBoost tie.
- **Feature engineering and a clean split made the biggest difference.** Adding `completion_ratio` and avoiding leakage mattered more than tuning.
- **Tuning added almost nothing.** The three methods are within 0.004 of each other, and the default XGBoost already scored 0.98.
- **The score drops from 0.99 to 0.93 without the task columns.** The other columns (turnaround time, quality, errors, rework, escalations) still carry most of the signal.

## Limitations

- **The label appears rule-generated.** Scores this close to perfect suggest the model is recovering a rule built from these columns. This is not evidence of real-world accuracy.
- **The data looks synthetic**, and `Employee_ID` repeats.
- **Tuning gave no meaningful gain**, so the final model should not be described as "optimised".
- **No business impact was measured.** Any savings would need a real pilot.
- The 0.7 flagging cutoff was chosen as a reasonable manager-capacity setting, not tuned on the data.

## Interactive dashboard

`app.py` is a Streamlit dashboard with five tabs and a sidebar filter (team and shift) that updates the data and EDA views:

| Tab | What it shows |
|---|---|
| Data Overview | Raw data, descriptive statistics, missing values, column types |
| EDA & Insights | Team, shift and SLA distributions, correlation heatmap, outlier box plots |
| Model Performance | Test macro-F1, model comparison, tuning comparison, confusion matrix, feature importance, the "without task columns" check |
| Risk Flags | A risk-cutoff slider showing how many employees are flagged, how many of them really breached, and the 15 highest-risk records |
| Predict SLA | Enter one employee's numbers and get the predicted status and breach probability |

The model is trained inside the app from the CSV (a few seconds on first load), so no saved model file is needed. The model-comparison and tuning numbers come from the notebook and are stored as constants at the top of `app.py`.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Repository structure

```
├── employee_sla_status.csv
├── employee_project.ipynb     # EDA, models, tuning, evaluation
├── app.py                     # interactive Streamlit dashboard and predictor
├── requirements.txt
└── images/dashboard.png
```

## Author

Aleena Elizabeth Thomas | [LinkedIn](https://www.linkedin.com/in/aleena-elizabeth-thomas/) | [GitHub](https://github.com/aleenaelizabethth)
