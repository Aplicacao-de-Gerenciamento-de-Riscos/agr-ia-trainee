# train_and_tune_models.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import joblib
import pandas as pd
import optuna
from data_processing import load_data
from database import get_db
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


def hyperparameter_tuning(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Melhores Parâmetros para {model.__class__.__name__}:", grid_search.best_params_)
    return grid_search.best_estimator_


def xgboost_optimization(X_train, y_train):
    def objective(trial):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
            'lambda': trial.suggest_loguniform('lambda', 1e-3, 1),
            'alpha': trial.suggest_loguniform('alpha', 1e-3, 1),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300)
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_train)
        return accuracy_score(y_train, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    print("Melhores Parâmetros para XGBClassifier:", study.best_params)
    return XGBClassifier(**study.best_params)


def predict_delay(model, version_data, epic_feature_names, threshold=0.5):
    """
    Prevê o risco de atraso e identifica os épicos que mais impactam a previsão.

    Parameters:
    - model: O modelo XGBoost treinado.
    - version_data: DataFrame com os dados da versão específica para predição.
    - epic_feature_names: Lista de features relacionadas aos épicos.
    - threshold: Limite de probabilidade para considerar atraso (default=0.5).

    Returns:
    - delay_risk_percentage: Percentual de risco de atraso.
    - top_impacting_epics: Lista de épicos que mais impactam o atraso.
    """
    # 1. Obtenha a probabilidade de atraso
    probability_of_delay = model.predict_proba(version_data)[:, 1]  # Probabilidade da classe de atraso
    delay_risk_percentage = probability_of_delay.mean() * 100  # Percentual médio de risco de atraso

    return delay_risk_percentage


def train_and_evaluate_models():
    # Carregar e preparar os dados
    db = next(get_db())
    data = load_data(db)
    data_df = pd.DataFrame(data)

    # Exportar os dados para um arquivo xlsx
    data_df.to_excel('data.xlsx', index=False)

    # Definir variáveis preditoras e variável target
    X = data_df.drop(columns=['cod_version', 'cod_project'])
    y = (data_df['timespent'] > data_df['time_original_estimate']).astype(int)

    # Dividir dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Aplicar SMOTE para lidar com o desbalanceamento
    smote = SMOTE(k_neighbors=4, random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Balanceamento no XGBoost
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

    # Treinar e avaliar RandomForest e GradientBoosting com GridSearchCV
    models_params = {
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'class_weight': ['balanced', None]
            }
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }
    }

    for model_name, mp in models_params.items():
        best_model = hyperparameter_tuning(mp['model'], mp['params'], X_train, y_train)
        y_pred = best_model.predict(X_test)
        print(f"\nModelo Tunado: {model_name}")
        print("Acurácia:", accuracy_score(y_test, y_pred))
        print("Relatório de Classificação:\n", classification_report(y_test, y_pred, zero_division=1))

    # Treinar e avaliar XGBoost com otimização
    def xgboost_objective(trial):
        params = {
            'objective': 'binary:logistic',
            'scale_pos_weight': scale_pos_weight,
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
            'lambda': trial.suggest_loguniform('lambda', 1e-3, 1),
            'alpha': trial.suggest_loguniform('alpha', 1e-3, 1),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300)
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_train)
        return accuracy_score(y_train, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(xgboost_objective, n_trials=50)
    best_xgb_params = study.best_params
    best_xgb_params['scale_pos_weight'] = scale_pos_weight
    best_xgb_model = XGBClassifier(**best_xgb_params)
    best_xgb_model.fit(X_train, y_train)
    y_pred_xgb = best_xgb_model.predict(X_test)
    print("\nModelo Tunado: XGBoost")
    print("Acurácia:", accuracy_score(y_test, y_pred_xgb))
    print("Relatório de Classificação:\n", classification_report(y_test, y_pred_xgb, zero_division=1))


if __name__ == "__main__":
    train_and_evaluate_models()
