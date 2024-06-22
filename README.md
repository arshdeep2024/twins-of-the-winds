# TWINS OF THE WINDS
### Project Report: Random Forest Regressor for Temperature Prediction

#### Introduction
This project involves building a predictive model using a Random Forest Regressor to predict the sea surface temperature (`s.s.temp.`) based on several meteorological variables. The key steps include data preprocessing, model training, hyperparameter tuning, and model evaluation.

#### Data Preprocessing
1. **Loading Data**: The dataset `train.csv` was loaded using pandas.
2. **Handling Missing Values**: Missing values in `humidity`, `zon.winds`, `mer.winds`, and `air temp.` were filled with the mean of their respective columns.
3. **Feature Selection**: The target variable was `s.s.temp.`, and features included `humidity`, `zon.winds`, `mer.winds`, and `air temp.`. The feature set without humidity was also prepared but not used in the final model.
4. **Splitting Data**: The dataset was split into training and testing sets with a test size of 9% using `train_test_split`.

#### Feature Selection
**Target and Features**:
1. The target variable (y1) was s.s.temp.
2. The features (X1) included all columns except s.s.temp.

#### Model Training and Hyperparameter Tuning
1. **Model Selection**: A `RandomForestRegressor` was chosen for its robustness and ability to handle missing values.
2. **Grid Search**: A grid search was performed to tune the hyperparameters:
   - `n_estimators`: Number of trees in the forest (200, 300)
   - `max_depth`: Maximum depth of the tree (None, 10)
   - `min_samples_split`: Minimum number of samples required to split an internal node (5, 7)
   - `min_samples_leaf`: Minimum number of samples required to be at a leaf node (1, 2)
3. **Scoring**: The model was evaluated using mean squared error (MSE) as the scoring metric, with higher scores indicating better performance.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
import pandas as pd
import numpy as np

# Load data
data = pd.read_csv("train.csv")

# Handle missing values
X1 = data.drop(["s.s.temp."], axis=1)
y1 = data["s.s.temp."]
X1.humidity.fillna(value=np.mean(X1.humidity), inplace=True)
X1["zon.winds"].fillna(value=np.mean(X1["zon.winds"]), inplace=True)
X1["mer.winds"].fillna(value=np.mean(X1["mer.winds"]), inplace=True)
X1["air temp."].fillna(value=np.mean(X1["air temp."]), inplace=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.09)

# Define model
clf = RandomForestRegressor(n_jobs=-1)

# Define hyperparameter grid
grid = {
    'n_estimators': [200, 300],
    'max_depth': [None, 10],
    'min_samples_split': [5, 7],
    'min_samples_leaf': [1, 2]
}

# Perform grid search
scorer = make_scorer(mean_squared_error, greater_is_better=False)
gs_clf = GridSearchCV(estimator=clf, param_grid=grid, cv=7, scoring=scorer, n_jobs=-1, verbose=2)
gs_clf.fit(X_train, y_train)
```

#### Model Performance
The model was evaluated on the test set using the best parameters found by the grid search. The performance was measured in terms of mean squared error (MSE).

```python
# Model evaluation
mse = mean_squared_error(y_test, gs_clf.predict(X_test))
print(f'Mean Squared Error: {mse}')
```

#### Predictions
The trained model was used to make predictions on separate datasets (`data_1997_1998.csv`) and (`evaluation.csv`).

### Conclusion
The project successfully built and tuned a Random Forest Regressor to predict sea surface temperature. The use of grid search allowed for optimal hyperparameter tuning, resulting in a model with a reasonable performance as evaluated by mean squared error.
