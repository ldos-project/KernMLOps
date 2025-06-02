import os
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import OneHotEncoder


def parse_ycsb_runtime(filename: str):
    matching_lines = []
    with open(filename, 'r') as file:
        for line in file:
            if 'RunTime' in line:
                matching_lines.append(line.strip())
    cumulative_ms = 0
    for line in matching_lines:
        rt = int(line.split(',')[2])
        cumulative_ms += rt
    return cumulative_ms

encodings = {
    'compressor': {'lzo': 0, 'deflate': 1, '842': 2, 'lz4': 3, 'lz4hc': 4, 'zstd': 5},
    'zpool': {'zbud': 0, 'z3fold': 1, 'zsmalloc': 2},
    'Y': 1,
    'N': 0
}

config_runtimes = defaultdict(list)
for filename in os.listdir("./random/results"):
    if not filename.startswith("redis_"):
        continue
    filepath = os.path.join("./random/results", filename)
    if not os.path.isfile(filepath):
        continue
    runtime = parse_ycsb_runtime(filepath)
    name, _ = os.path.splitext(filename)
    parts = name.split('_')[1:-1]
    if parts[0] not in encodings['compressor'] or parts[1] not in encodings['zpool']:
        continue
    compressor = encodings['compressor'][parts[0]]
    zpool = encodings['zpool'][parts[1]]
    numeric = list(map(int, parts[2:4]))
    bools = [encodings[part] for part in parts[4:]]
    config_key = tuple([compressor, zpool] + numeric + bools)
    config_runtimes[config_key].append(runtime)

X = []
y = []
for config, runtimes in config_runtimes.items():
    X.append(list(config))
    y.append(sum(runtimes) / len(runtimes))

# Convert to numpy arrays for easier manipulation
X_np = np.array(X)
y_np = np.array(y)

# Create train/test split for evaluation
X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=42)

# Apply one-hot encoding to categorical features
categorical_features = [0, 1]  # Compressor and zpool indices
encoder = OneHotEncoder(sparse_output=False)  # Changed to sparse=False for older versions
X_train_cat = encoder.fit_transform(X_train[:, categorical_features])
X_test_cat = encoder.transform(X_test[:, categorical_features])
X_train_num = X_train[:, 2:]  # Numerical features
X_test_num = X_test[:, 2:]
X_train_transformed = np.hstack((X_train_cat, X_train_num))
X_test_transformed = np.hstack((X_test_cat, X_test_num))

# Setup cross-validation to evaluate model performance
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Create and fit basic model for comparison
base_model = RandomForestRegressor(random_state=42)
base_model.fit(X_train_transformed, y_train)

# Evaluate base model
base_pred = base_model.predict(X_test_transformed)
# Calculate RMSE manually
base_mse = mean_squared_error(y_test, base_pred)
base_rmse = np.sqrt(base_mse)  # Calculate RMSE manually
base_r2 = r2_score(y_test, base_pred)
print(f"Base model RMSE: {base_rmse:.2f}")
print(f"Base model R²: {base_r2:.3f}")

# Cross-validate base model
base_cv_scores = cross_val_score(base_model, X_train_transformed, y_train,
                                cv=cv, scoring='neg_mean_squared_error')
print(f"Cross-validation MSE: {-base_cv_scores.mean():.2f} ± {base_cv_scores.std():.2f}")

# Hyperparameter tuning
param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

random_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=10,
    cv=cv,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search.fit(X_train_transformed, y_train)
best_model = random_search.best_estimator_
print(f"\nBest parameters: {random_search.best_params_}")

# Evaluate best model
best_pred = best_model.predict(X_test_transformed)
best_mse = mean_squared_error(y_test, best_pred)
best_rmse = np.sqrt(best_mse)  # Calculate RMSE manually
best_r2 = r2_score(y_test, best_pred)
print(f"Tuned model RMSE: {best_rmse:.2f}")
print(f"Tuned model R²: {best_r2:.3f}")

# Get feature importances using permutation importance
# Need to map back feature names to account for one-hot encoding
cat_feature_names = []
for i, feat in enumerate(categorical_features):
    if feat == 0:  # compressor
        for comp_name in encodings['compressor'].keys():
            cat_feature_names.append(f"compressor_{comp_name}")
    elif feat == 1:  # zpool
        for zpool_name in encodings['zpool'].keys():
            cat_feature_names.append(f"zpool_{zpool_name}")

numeric_feature_names = [
    "max_pool_percent", "accept_threshold_percent",
    "shrinker_enabled", "exclusive_loads",
    "same_filled_pages_enabled", "non_same_filled_pages_enabled"
]

transformed_feature_names = cat_feature_names + numeric_feature_names

# Calculate permutation importance on test set
result = permutation_importance(best_model, X_test_transformed, y_test,
                               n_repeats=10, random_state=42)
perm_importances = result.importances_mean

print("\nPermutation importances (transformed features):")
for imp, name in sorted(zip(perm_importances, transformed_feature_names),
                        key=lambda x: x[0], reverse=True):
    print(f"{name}: {imp:.3f}")

# Original feature importances from the model (for the transformed features)
print("\nModel feature importances (transformed features):")
for imp, name in sorted(zip(best_model.feature_importances_, transformed_feature_names),
                        key=lambda x: x[0], reverse=True):
    print(f"{name}: {imp:.3f}")
