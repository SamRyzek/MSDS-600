import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('../Assignment/data/new_churn_data.csv')
tpot_data = tpot_data.drop('customerID', axis=1)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: 0.7967416387132746
exported_pipeline = make_pipeline(
    PCA(iterated_power=8, svd_solver="randomized"),
    DecisionTreeClassifier(criterion="entropy", max_depth=9, min_samples_leaf=2, min_samples_split=12)
)
# Fix random state in exported estimator
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

print(results)
