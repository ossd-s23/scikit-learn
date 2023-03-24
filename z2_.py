from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

import pandas as pd
import numpy as np

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    np.random.seed(1024)

    # Creating dummy X and y
    dummy_X = pd.DataFrame(
        np.arange(0, 3, 0.01).reshape((100, 3)), columns=["a", "b", "c"]
    )
    dummy_y = pd.DataFrame(np.random.choice([0, 1], size=(100, 1)), columns=["label"])

    # Create preprocessing pipeline
    prep_pipeline = Pipeline(
        [
            ("Imputer", SimpleImputer(strategy="median")),
            ("StdScalar", StandardScaler()),
        ]
    )

    # Use column transformer
    mod_prep_pipeline = ColumnTransformer(
        [("num", prep_pipeline, [0, 1])], remainder="passthrough"
    )

    # Create full pipeline
    model = Pipeline(
        [
            ("Preprocessing", mod_prep_pipeline),
            ("Classifier", KNeighborsClassifier()),
        ]
    )

    print(model)

    # Apply the sequential feature selector
    sfs = SequentialFeatureSelector(model)
    sfs.fit(dummy_X, dummy_y)
