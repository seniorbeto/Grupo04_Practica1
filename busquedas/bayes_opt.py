import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import optuna
import numpy as np

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


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 250),
        'max_depth': trial.suggest_int('max_depth', 10, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
    }
    # print(f"Trial {trial.number} - Iniciando entrenamiento con hiperparámetros: {params}")
    
    model = RandomForestRegressor(**params)
    rmse_scores = []
    for _ in range(5):  # 5-fold cross-validation
        X_fold_train, X_fold_val, y_fold_train, y_fold_val = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED)
        model.fit(X_fold_train, y_fold_train)
        y_pred = model.predict(X_fold_val)
        rmse = metrics.root_mean_squared_error(y_fold_val, y_pred)
        rmse_scores.append(rmse)
    
    # print(f"Trial {trial.number} - RMSE: {np.mean(rmse_scores)}")
    return np.mean(rmse_scores)


def progress_callback(study, trial):
    print(f"Trial {trial.number} finished with value: {trial.value} and parameters: {trial.params}")
    print(f"Best trial so far: {study.best_trial.number}")

study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))

print("Iniciando búsqueda de hiperparámetros...")
study.optimize(objective, n_trials=100, n_jobs=-1, callbacks=[progress_callback], show_progress_bar=True)

best_params = study.best_params
print("Mejores hiperparámetros encontrados:")
print(best_params)

df = study.trials_dataframe()

# Guarda el DataFrame en un archivo CSV
df.to_csv('optuna_results.csv', index=False)

# Entrenar el mejor modelo con los mejores hiperparámetros
best_model = RandomForestRegressor(**best_params)
best_model.fit(X_train, y_train)

# Predecir los valores de test
y_test_pred = best_model.predict(X_test)

# Calcular métricas
rmse_rf_cv = metrics.mean_squared_error(y_test, y_test_pred, squared=False)
r2_rf_cv = best_model.score(X_test, y_test)

print(f"RMSE: {rmse_rf_cv}")
print(f"R2: {r2_rf_cv}")

# exportar el mejor modelo
import pickle
with open('bayes_rf.pkl', 'wb') as f:
    pickle.dump(best_model, f)

