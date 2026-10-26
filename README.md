# Titanic Survival Prediction — Model Selection & Feature Engineering

A compact record of my work on Kaggle's [Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) competition. The goal was to predict passenger survival and understand whether additional feature engineering or more complex models would improve a solid baseline.

**Best public Kaggle accuracy: 0.79665** using **Gradient Boosting** with seven original passenger features. That score came from `submission_cv_best_v3.csv`.

## Approach

1. **Model selection:** Compared Logistic Regression, Random Forest, Extra Trees and Gradient Boosting on the same stratified five-fold CV splits.
2. **Feature engineering:** Tested names/titles, family information, passenger class and fare interactions, women/children indicators and cabin information.
3. **Robustness testing:** Repeated five-fold CV five times, using matched splits for the baseline, Title and Title + Family features.
4. **Alternative algorithm:** Compared CatBoost with Gradient Boosting over 15 matched validation folds.

All imputation, encoding and feature transformations used for the scikit-learn experiments are inside training pipelines, so they are fitted within the corresponding training fold rather than on the entire dataset.

## Results

### Selected cross-validation results

| Experiment | Validation approach | Mean accuracy |
| --- | --- | ---: |
| Gradient Boosting (seven original features) | Stratified 5-fold | **0.8272** |
| Random Forest (same features) | Stratified 5-fold | 0.8271 |
| Gradient Boosting + Title | Stratified 5-fold | 0.8361 |
| Gradient Boosting baseline | Repeated 5-fold × 5 | 0.8267 |
| Gradient Boosting + Title | Repeated 5-fold × 5 | 0.8269 |
| Gradient Boosting + Title & Family | Repeated 5-fold × 5 | 0.8285 |
| Gradient Boosting baseline | Repeated 5-fold × 3 | **0.8294** |
| CatBoost | Repeated 5-fold × 3 | 0.8230 |

**Note:** The three CV protocols above have different train/validation splits; compare models *within* a protocol rather than treating differences across protocols as controlled improvements. These numbers are from the original notebook runs; rerun the notebooks to regenerate the tables.

### Kaggle public leaderboard submissions

| Submission | Description | Public accuracy |
| --- | --- | ---: |
| `submission_cv_best_v3.csv` | Gradient Boosting, selected by CV | **0.79665** |
| `submission_rf_simple_v2.csv` | Earlier simple Random Forest run | 0.78229 |
| `submission_v1.csv` | First submission | 0.77990 |
| `submission_catboost_v6.csv` | CatBoost | 0.77511 |
| `submission_gb_title_family.csv` | Gradient Boosting, Title + Family | 0.76315 |
| `submission_gb_tuned_v4.csv` | Earlier Gradient Boosting tuning run | 0.76315 |

The scores above are historical Kaggle submission results, not notebook-computed CV metrics. The public leaderboard is a separate held-out test sample; its scores are not directly comparable to cross-validation estimates. This repository recreates only the three submission-generating experiments in notebooks 01, 03 and 04, not every earlier run.

## What worked, and what didn't

- **Worked:** A straightforward Gradient Boosting model using `Pclass`, `Sex`, `Age`, `Fare`, `SibSp`, `Parch` and `Embarked` produced the highest observed public score (**0.79665**). Its five-fold mean CV score was **0.8272**.
- **Initially promising:** The Title feature improved mean accuracy from **0.8272** to **0.8361** in the first five-fold feature experiment.
- **Less convincing after more testing:** With 25 matched folds, Title + Family improved mean CV accuracy only from **0.8267** to **0.8285** (**+0.179 percentage points**), and the matching leaderboard submission was worse (**0.76315**).
- **Did not help here:** Combining every engineered feature did not outperform Title alone in the first feature test (**0.8283** vs **0.8361**). CatBoost also trailed the baseline in both its matched repeated-CV comparison (**0.8230** vs **0.8294**) and its public Kaggle submission (**0.77511** vs **0.79665**).

**Takeaway:** Feature ideas that improve one validation split may not hold up under repeated evaluation or transfer to a separate leaderboard set. For this project, the simpler baseline remained the best-performing submitted model.

## Repository structure

```text
titanic-survival-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_model_comparison.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_robustness_testing.ipynb
│   └── 04_catboost_comparison.ipynb
├── src/
│   └── feature_engineering.py
└── data/
    └── README.md
```

Each notebook is self-contained aside from the small helper module used in notebooks 02–03 and the competition CSVs. The original experiment notebooks contained multiple workflows in single cells; these versions split them into readable, documented stages while preserving the underlying experiment settings.

## Reproduce

1. Clone this repository and enter its root directory.
2. Download [`train.csv` and `test.csv` from Kaggle](https://www.kaggle.com/competitions/titanic/data) and save both into `data/`.
3. Install dependencies and open Jupyter:

   ```bash
   python -m pip install -r requirements.txt
   jupyter notebook
   ```

4. Run the notebooks in order. They can be launched from the repository root or from `notebooks/`; the import and data paths resolve either way. Notebooks 01, 03 and 04 write optional Kaggle submissions into `submissions/` (ignored by Git).

**Data and generated submissions are excluded from Git**; the notebooks contain no private paths or raw passenger datasets. Re-running with newer package versions may produce small numeric differences.

## Tools

Python, pandas, NumPy, scikit-learn, CatBoost, matplotlib and Jupyter.
