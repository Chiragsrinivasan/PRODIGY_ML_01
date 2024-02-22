import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

X_train = train_data.drop(columns=['SalePrice'])
y_train = train_data['SalePrice']

numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),('scaler', StandardScaler())])

categorical_features = X_train.select_dtypes(include=['object']).columns
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                          ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
                                               ('cat', categorical_transformer, categorical_features)])

rf_regressor = Pipeline(steps=[('preprocessor', preprocessor),('regressor', RandomForestRegressor())])

rf_regressor.fit(X_train, y_train)

y_train_pred = rf_regressor.predict(X_train)
train_rmse = sqrt(mean_squared_error(y_train, y_train_pred))

predictions = rf_regressor.predict(test_data)

submission = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': predictions})

submission.to_csv('submission.csv', index=False)
