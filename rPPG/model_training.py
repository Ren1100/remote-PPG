import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from data_preprocessing import load_data

data_dir = 'data/PURE/'
window_size = 30  # Example window size
X, y = load_data(data_dir, window_size)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning with Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Evaluate the model
accuracy = best_model.score(X_test, y_test)
print(f'Test Accuracy: {accuracy:.2f}')

# Save the model
joblib.dump(best_model, 'rppg_model.pkl')
