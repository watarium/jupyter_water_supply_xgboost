import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('water_supply.csv')
X_train_validate, X_test, y_train_validate, y_test = train_test_split(df.drop('water', axis=1), df['water'].ravel())

dtrain = xgb.DMatrix(X_train_validate.values, label=y_train_validate.tolist())

fit_params = {
    'eval_metric': 'rmse',
    'eval_set': [[X_train_validate, y_train_validate]]
}

# Range of grid search
params = {
    'learning_rate': list(np.arange(0.05, 0.41, 0.05)),
    'max_depth': list(np.arange(3, 11, 1))
}

# Fit for Grid search
def GSfit(params):
    regressor = xgb.XGBRegressor(n_estimators=100)
    grid = GridSearchCV(regressor, params, cv=3, fit_params=fit_params, scoring='neg_mean_squared_error', verbose=2, return_train_score=True)
    grid.fit(X_train_validate,y_train_validate)
    return grid

# Grid search
grid = GSfit(params)
grid_best_params = grid.best_params_
grid_scores_df = pd.DataFrame(grid.cv_results_)

# Best n by Cross Validation
cv=xgb.cv(grid_best_params, dtrain, num_boost_round=200, nfold=3)
n_best = cv[cv['test-rmse-mean'] == cv['test-rmse-mean'].min()]['test-rmse-mean'].index[0]
grid_best_params['n_estimators'] = n_best + 1

# Fit by best params
regressor = xgb.XGBRegressor(learning_rate=grid_best_params['learning_rate'],
                             max_depth=grid_best_params['max_depth'],
                             n_estimators=grid_best_params['n_estimators'])
regressor.fit(X_train_validate, y_train_validate, verbose=False)

# Save model
pickle.dump(regressor, open('water_supply.pkl', 'wb'))

# Prediction
for hour in range(1,24):
    for temperature in range(5,40):
        input = pd.DataFrame([[hour, temperature]], columns=['hour', 'temperature'])
        result = regressor.predict(input)
        print('Hour: ' + str(hour) + '  Temperature: ' + str(temperature) + '  Prediction: ' + str(result[0]))