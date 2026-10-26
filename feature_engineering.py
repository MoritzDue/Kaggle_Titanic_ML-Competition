"""Feature engineering helpers for Titanic CV experiments.

This is a focused refactor of the original ``preprocessingresult.py``.
The feature definitions and preprocessing choices are intentionally preserved.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

BASE_FEATURES = [
    'Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked',
]

FEATURE_SETS = {
    'Baseline': [],
    'Family': ['FamilySize', 'IsAlone', 'HasParentChildAboard'],
    'Women and children': [
        'IsChild', 'AgeMissing', 'AdultFemaleWithParch', 'ChildWithParch',
    ],
    'Class and fare': ['SexClass', 'FarePerFamilyMember'],
    'Title': ['Title'],
    'Cabin': ['Deck', 'CabinMissing'],
}
FEATURE_SETS['Combined'] = list(dict.fromkeys(
    feature for group in FEATURE_SETS.values() for feature in group
))
FEATURE_SETS['Title + Family'] = FEATURE_SETS['Title'] + FEATURE_SETS['Family']


def engineer_experiment_features(df):
    """Derive the same title, family and other features as the original runs."""
    df = df.copy()

    df['Title'] = (
        df['Name'].str.extract(r',\s*([^.]*)\.')[0]
        .fillna('Unknown').str.strip()
    )
    df['NameLength'] = df['Name'].str.len()
    ticket_clean = (
        df['Ticket'].fillna('Unknown').str.upper()
        .str.replace(r'[^A-Z0-9]', '', regex=True)
    )
    df['TicketPrefix'] = (
        ticket_clean.str.extract(r'^([A-Z]+)')[0].fillna('NONE')
    )
    df['TicketLength'] = df['Ticket'].str.len()
    df['Deck'] = df['Cabin'].str[0].fillna('Unknown')
    df['CabinMissing'] = df['Cabin'].isna().astype(int)
    df['AgeMissing'] = df['Age'].isna().astype(int)
    df = df.drop(columns=['Name', 'Ticket', 'Cabin'])

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['HasParentChildAboard'] = (df['Parch'] > 0).astype(int)
    df['IsChild'] = (df['Age'] < 16).astype(int)
    df['AdultFemaleWithParch'] = (
        (df['Sex'] == 'female') & (df['Age'] >= 16) & (df['Parch'] > 0)
    ).astype(int)
    df['ChildWithParch'] = (
        (df['Age'] < 16) & (df['Parch'] > 0)
    ).astype(int)
    df['SexClass'] = df['Sex'] + '_P' + df['Pclass'].astype(str)
    df['FarePerFamilyMember'] = df['Fare'] / df['FamilySize']
    return df


def create_experiment_pipeline(feature_set, classifier):
    """Return a full CV-safe pipeline for the requested feature group."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f'Unknown feature set {feature_set!r}')

    selected = BASE_FEATURES + FEATURE_SETS[feature_set]
    categorical = [
        col for col in selected
        if col in ['Pclass', 'Sex', 'Embarked', 'Title', 'Deck', 'SexClass']
    ]
    numerical = [col for col in selected if col not in categorical]

    preprocessor = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median'), numerical),
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ]), categorical),
    ], sparse_threshold=0.0)

    return Pipeline([
        ('feature_engineering', FunctionTransformer(
            engineer_experiment_features, validate=False,
        )),
        ('preprocessing', preprocessor),
        ('classifier', classifier),
    ])
