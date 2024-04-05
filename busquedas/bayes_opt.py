import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)


def objective(trial):
    # parametros para un svr
    params = {
        'C': trial.suggest_loguniform('C', 1e-2, 1e2),
        'epsilon': trial.suggest_loguniform('epsilon', 1e-2, 1e2),
        'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
        'degree': trial.suggest_int('degree', 1, 5),
        'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
        'coef0': trial.suggest_loguniform('coef0', 1e-2, 1e2)
    }
    # print(f"Trial {trial.number} - Iniciando entrenamiento con hiperparámetros: {params}")
    
    model = SVR(**params)
    rmse_scores = []
    for _ in range(10):  # 10-fold cross-validation
        X_fold_train, X_fold_val, y_fold_train, y_fold_val = train_test_split(X_train, y_train, test_size=0.1, random_state=SEED)
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
study.optimize(objective, n_trials=1000, n_jobs=-1, callbacks=[progress_callback], show_progress_bar=True)

best_params = study.best_params
print("Mejores hiperparámetros encontrados:")
print(best_params)

df = study.trials_dataframe()

# Guarda el DataFrame en un archivo CSV
df.to_csv('optuna_results.csv', index=False)

# Entrenar el mejor modelo con los mejores hiperparámetros
best_model = SVR(**best_params)
best_model.fit(X_train, y_train)

# Predecir los valores de test
y_test_pred = best_model.predict(X_test)

# Calcular métricas
rmse_rf_cv = metrics.root_mean_squared_error(y_test, y_test_pred)
r2_rf_cv = best_model.score(X_test, y_test)

print(f"SVR RMSE: {rmse_rf_cv}")
print(f"SVR R2: {r2_rf_cv}")

# exportar el mejor modelo
try:
    import pickle
    with open(f'bayes_svr_{rmse_rf_cv}.pkl', 'wb') as f:
        pickle.dump(best_model, f)
except:
    try:
        import joblib
        joblib.dump(best_model, f'bayes_svr_{rmse_rf_cv}.pkl')
    except:
        pass
    
print(f"Parametros del mejor model (svr): {best_params}")
    
    
# mejor random forest:
# {'n_estimators': 333, 'max_depth': 34, 'min_samples_split': 2, 'min_samples_leaf': 2, 'bootstrap': True}

# mejor svr: