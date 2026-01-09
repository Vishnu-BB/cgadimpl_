import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import joblib



df = pd.read_csv('advertising.csv')
df

X = df.drop(columns=['Sales'])
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)




pipeline = Pipeline(steps=[
    ('scaler',StandardScaler()),
    ('model', LinearRegression())
])



param_grid = {
    'model__fit_intercept' : [True, False]
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator = pipeline,
    param_grid = param_grid,
    scoring="neg_root_mean_squared_error",
    n_jobs = -1,
    cv=cv
)

grid = grid.fit(X_train, y_train)
grid



best_model = grid.best_estimator_

print("Best Parameters :", grid.best_params_)
print("Best CV RMSE :", grid.best_score_)



y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("RMSE :", rmse)

baseline_pred = np.mean(y_train)
baseline_rmse = np.sqrt(np.mean((y_test - baseline_pred)**2))
print("Baseline RMSE :",baseline_rmse)

if rmse < baseline_rmse:
    print("Model is good")
else:
    print("Model is not good")

