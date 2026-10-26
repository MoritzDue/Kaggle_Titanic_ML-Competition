# Titanic Survival Prediction | Kaggle

A machine learning project to predict which Titanic passengers survived. I compared different models, tested new features, and checked whether improvements held up across multiple validation runs.

**Best Kaggle result: 79.7% accuracy** (public leaderboard score: **0.79665**) using **Gradient Boosting** with basic passenger information.

## What I tried

- Compared Logistic Regression, Random Forest, Extra Trees and Gradient Boosting.
- Created features based on passenger titles, family groups, cabins and fares.
- Used repeated cross-validation to check whether apparent improvements were reliable.
- Tested CatBoost against the best baseline.

## Results

| Approach | Kaggle accuracy |
| --- | ---: |
| **Gradient Boosting (basic features)** | **79.7%** |
| Random Forest | 78.2% |
| CatBoost | 77.5% |
| Gradient Boosting + Title & Family features | 76.3% |

**Key takeaway:** More complex models and extra features did not guarantee better predictions. Features such as passenger titles looked promising in initial validation, but their advantage was small in repeated testing and did not improve the Kaggle submission. The simpler Gradient Boosting model performed best.

## Notebooks

1. [Model comparison](notebooks/01_model_comparison.ipynb) — choose a baseline model.
2. [Feature engineering](notebooks/02_feature_engineering.ipynb) — test new passenger features.
3. [Robustness testing](notebooks/03_robustness_testing.ipynb) — check results across more data splits.
4. [CatBoost comparison](notebooks/04_catboost_comparison.ipynb) — compare a more advanced algorithm.

## Run the project

Download `train.csv` and `test.csv` from the [Kaggle competition](https://www.kaggle.com/competitions/titanic/data) and place them in `data/`. Then run:

```bash
pip install -r requirements.txt
jupyter notebook
```

**Tools:** Python, pandas, scikit-learn, CatBoost and Jupyter.
