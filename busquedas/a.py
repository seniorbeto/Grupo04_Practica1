# 'n_estimators': 331, 'max_depth': 26, 'min_samples_split': 4, 'min_samples_leaf': 1, 'bootstrap': True

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import pandas as pd

# definicion de constantes usadas a lo largo del proyecto
SEED = 100472050  # la semilla debe ser el NIA de uno de los integrantes
wind_ava = pd.read_csv("data/wind_ava.csv", index_col=0)
aux = wind_ava[wind_ava.columns[wind_ava.columns.str.endswith('13')]]
# añadir la columna energy a wind_ava
aux.insert(0, "energy", wind_ava["energy"])     
wind_ava = aux
print(wind_ava.head())

# Dividimos los datos en entrenamiento y test
X = wind_ava.drop(columns='energy')
y = wind_ava['energy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)


params = {
    'n_estimators': 331,
    'max_depth': 26,
    'min_samples_split': 4,
    'min_samples_leaf': 1,
    'bootstrap': True
}

model = RandomForestRegressor(**params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = metrics.r2_score(y_test, y_pred)
rmse = metrics.root_mean_squared_error(y_test, y_pred)
print(f"RMSE: {rmse}")
print(f"R2: {r2}")


# open bayes_rf.pkl (model) and compare the results with the ones obtained here
import pickle
with open('bayes_rf.pkl', 'rb') as f:
    model = pickle.load(f)
y_pred = model.predict(X_test)
r2 = metrics.r2_score(y_test, y_pred)
rmse = metrics.root_mean_squared_error(y_test, y_pred)
print(f"RMSE: {rmse}")
print(f"R2: {r2}")
