# python MLOps_start_HW_2.py
import mlflow
import os
import pandas as pd
import logging

from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

#os.getenv("MLFLOW_TRACKING_URI", "No env")
#mlflow.get_registry_uri()

_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

### Соберем модели для будущего цикла ###
_LOG.info("1. Download and init data")
model_names = ["random_forest", "linear_regression", "desicion_tree"]
models = dict(
    zip(model_names, [
        RandomForestRegressor(),
        LinearRegression(),
        DecisionTreeRegressor(),
    ]))


### Загрузка и предобработка данных ###
data = fetch_openml(name="house_prices", as_frame=True, parser='auto')
data = pd.concat([data['data'], data['target']], axis=1)

_LOG.info("2. Prepaid data")
y = data['SalePrice']
X = data.drop(columns='SalePrice')

# Оставлю тольку булевые и числовые признаки
X = X.select_dtypes(include=['number', 'bool'])
X_columns = X.columns

# Изюавлюсь от пропусков - найдем колонки с пропуском и заполним значение модой
none_in_columns = (X.isna().sum()>0)*1
none_in_columns = none_in_columns[none_in_columns>0].index

for col_with_nan in none_in_columns:
    inpit_val = X[col_with_nan].mode()[0]
    X[col_with_nan] = X[col_with_nan].fillna(inpit_val)

assert X.isna().sum().sum() + y.isna().sum() == 0, 'Естьпропуски в Х или у!'

### Выделяем тренировочные и тестовые наборы ###
_LOG.info("3. Split data")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Стандартизация признаков для линейной модели, нарочно не стандартизирую y
s_scaller = StandardScaler()
X_train_ss = s_scaller.fit_transform(X_train)
X_test_ss = s_scaller.transform(X_test)

X_train_ss = pd.DataFrame(X_train_ss, columns=X_columns)
X_test_ss = pd.DataFrame(X_test_ss, columns=X_columns)

### Создаем/подключаемся к эксперементу и выполняем run ###

# Поиск эксперемента
experiment_name = 'Danila_Iugai'
experiment = mlflow.get_experiment_by_name(experiment_name)
_LOG.info(f"4. Try to find experiment_name: {experiment_name}")

# Создадим или подключимся к эксперементу, если такой существует
if experiment:
    exp_id = mlflow.set_experiment('Danila_Iugai').experiment_id
    _LOG.info(f"4.1 set experiment: {experiment_name}")
else:
    exp_id = mlflow.create_experiment('Danila_Iugai')
    _LOG.info(f"4.1 create experiment: {experiment_name}")


# Создадим parent run
with mlflow.start_run(run_name="dangennadevich", experiment_id = exp_id, description = "parent") as parent_run:
    for model_nm in models.keys():
        # Запустим child run на каждую модель
        with mlflow.start_run(run_name=model_nm, experiment_id=exp_id, nested=True) as child_run:
            model = models[model_nm]

            # Определяем X_train_iter и X_test_iter в зависимости от модели (стандартизированные ли признаки)
            if model_nm == 'linear_regression':
                X_train_iter = X_train_ss
                X_test_iter = X_test_ss
            else:
                X_train_iter = X_train
                X_test_iter = X_test

            # Модель и предсказание
            model.fit(X_train_iter, y_train)
            y_pred = model.predict(X_test_iter)        

            # Создадим датасет для mlflow.evaluate
            eval_df = X_test_iter.copy()
            eval_df["target"] = y_test.values
            
            # Сохраним результаты обучения с помощью MLFlow.
            signature = infer_signature(X_test_iter, y_pred)
            model_info = mlflow.sklearn.log_model(sk_model = model, artifact_path = model_nm, signature=signature)
            mlflow.evaluate(
                model=model_info.model_uri,
                data=eval_df,
                targets="target",
                model_type="regressor",
                evaluators=["default"],
            )