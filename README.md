# Claim Settlement Bias Analysis Dashboard

A Streamlit dashboard for claim settlement officers to:

1. Run descriptive cross-tabulation of claim attributes against `POLICY_STATUS`.
2. Run diagnostic statistical tests (chi-square, Cramér's V, Mann-Whitney U) to
   surface attributes (age, income, team, occupation, etc.) where approval
   rates differ significantly — i.e. possible bias signals.
3. Train four supervised classifiers (KNN, Decision Tree, Random Forest,
   Gradient Boosting) with proper feature engineering to predict claim
   outcome.
4. Compare model train/test accuracy, precision, recall, F1, ROC curves and
   confusion matrices.
5. View an auto-generated findings & recommendations summary.

## Project structure

```
claim-bias-dashboard/
├── app.py                 # Streamlit app (5 pages, sidebar navigation)
├── utils/
│   ├── data_processing.py # cleaning, crosstabs, bias stats, feature engineering
│   └── modeling.py        # preprocessing pipeline, training, evaluation
├── data/
│   └── Insurance.csv      # sample dataset (replace or upload your own from the UI)
├── requirements.txt
└── README.md
```

## Run locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. Use the sidebar to navigate between
the five analysis pages, or upload your own CSV (same columns as
`data/Insurance.csv`) to analyze different data without touching the code.

## Deploy on Streamlit Community Cloud (free)

1. Create a new GitHub repository and push this folder to it:
   ```bash
   git init
   git add .
   git commit -m "Claim settlement bias dashboard"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, click **New app**.
3. Select your repo, branch `main`, and main file path `app.py`.
4. Click **Deploy**. The bundled `data/Insurance.csv` will be used by default;
   anyone using the app can also upload a different CSV from the sidebar.

> If your data is sensitive, do **not** commit it to a public repo — either
> use a private repo, or remove `data/Insurance.csv` and rely solely on the
> in-app uploader (nothing is persisted server-side between sessions).

## Expected CSV columns

`POLICY_NO, PI_NAME, PI_GENDER, SUM_ASSURED, ZONE, PAYMENT_MODE, EARLY_NON,
PI_OCCUPATION, MEDICAL_NONMED, PI_STATE, REASON_FOR_CLAIM, PI_AGE,
PI_ANNUAL_INCOME, POLICY_STATUS`

`POLICY_STATUS` is expected to contain exactly two outcome labels, by default
`"Approved Death Claim"` and `"Repudiate Death"`. If your data uses different
label text, update `APPROVED_LABEL` / `REJECTED_LABEL` in
`utils/data_processing.py`.

## Methodology notes

- **Cleaning**: `SUM_ASSURED` / `PI_ANNUAL_INCOME` are parsed from
  comma-formatted text to numeric; `ZONE` is normalized into a `TEAM` column
  to collapse case/whitespace duplicates (e.g. `South`, `SOUTH`, `South 2`).
- **Diagnostic tests**: chi-square test of independence for categorical
  attributes vs. outcome, with Cramér's V as effect size; Mann-Whitney U test
  for numeric attributes (age, income) since claim outcome is binary and
  these variables are not normally distributed.
- **Feature engineering**: numeric features are standardized; high-cardinality
  categorical features (state, occupation, reason, team) are capped to their
  top-N most frequent values + `"Other"`, then one-hot encoded — all inside a
  scikit-learn `Pipeline` fit only on the training fold to avoid leakage.
- **Models**: `KNeighborsClassifier`, `DecisionTreeClassifier`,
  `RandomForestClassifier`, `GradientBoostingClassifier` from scikit-learn,
  with adjustable hyperparameters in the sidebar.
- **Evaluation**: accuracy (train & test, to flag overfitting), precision,
  recall, F1, ROC-AUC, ROC curves, and confusion matrices for all four models.

## Disclaimer

Statistical association between an attribute and approval rate is a signal
worth investigating, not proof of intentional bias on its own — confirm
findings against case-level underwriting documentation before drawing
conclusions about individuals or teams.
