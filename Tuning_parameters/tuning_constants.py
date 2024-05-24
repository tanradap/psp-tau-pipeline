from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.feature_selection import RFE

# Hyperparameter tuning
pipeline = Pipeline([
    ('normalizer', MinMaxScaler()),
    ('selector', RFE(RandomForestClassifier(random_state=42))),
    ('clf', BalancedRandomForestClassifier())
])

# Hyper parameter space

# Features
# features_to_select = [40, 42, 44, 46, 48, 50, 52, 54]
features_to_select = [28, 30, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54]

# Number of trees in random forest
n_estimators = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

# Number of features to consider at every split (sqrt(54) = ~7)
max_features = [0.2, 0.4, 0.6, 0.8, 1]
# max_features = ['sqrt', 0.2, 0.4, 0.6, 0.8, 1]

# Maximum number of levels in tree
max_depth = [5, 10, 15, 20]
max_depth.append(None)

# Minimum number of samples required to split an internal node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# sampling strategy
sampling_strategy = ['auto', 'all', 'not majority', 'majority'] # auto = not minority

# max_samples
max_samples = [0.25, 0.5, 0.75, None]

# # class weights
# class_weight = ['balanced', 'balanced_subsample', None]

# Create the random grid
random_grid = {'selector__n_features_to_select': features_to_select,
                'clf__n_estimators': n_estimators,
               'clf__max_features': max_features,
               'clf__max_depth': max_depth,
               'clf__min_samples_split': min_samples_split,
               'clf__min_samples_leaf': min_samples_leaf,
               'clf__random_state': [42],
               'clf__sampling_strategy': sampling_strategy,
               'clf__max_samples': max_samples,
               'clf__class_weight': ['balanced']
              }