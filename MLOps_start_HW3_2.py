import mlflow
import os

from airflow.models import DAG, Variable
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from typing import Any, Dict, Literal
# YOUR IMPORTS HERE
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.datasets import fetch_openml
from datetime import datetime, timedelta

from mlflow.models import infer_signature

import json
import pandas as pd
import logging
import io
import tempfile
import os

_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

NAME = 'Danila_Iugai'
BUCKET = Variable.get("S3_BUCKET")
DEFAULT_ARGS = {
    "owner": "Iugai Danila",
    "retry": 3,
    "retry_delay": timedelta(minutes=1)
}

dag = DAG(
    dag_id=NAME,
    tags=["mlops"],
    schedule_interval="0 1 * * *",
    start_date=days_ago(2),
    catchup=False,
    default_args=DEFAULT_ARGS,
    concurrency=5
)

model_names = ["random_forest", "linear_regression", "decision_tree"]
models = dict(
    zip(model_names, [
        RandomForestRegressor(),
        LinearRegression(),
        DecisionTreeRegressor(),
    ]))

def configure_mlflow():
    for key in [
        "MLFLOW_TRACKING_URI",
        "AWS_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ]:
        os.environ[key] = Variable.get(key)

# Отключаем сообщение о git в MLflow
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'


def init() -> Dict[str, str]:
    """Лог старта пайплайна и название модели"""
    metrics = dict()
    metrics['start_timestamp'] = datetime.now().strftime('%Y-%m-%d %H%:%M %H%:%M:%S')
    
    _LOG.info(f"Train pipeline started.")
    
    # Инициализируем mlflow
    configure_mlflow()

    # Получаем информацию об эксперименте
    experiment_name = NAME
    metrics['experiment_name'] = experiment_name
    _LOG.info(f"Try to find experiment_name: {experiment_name}")

    existing_exp = mlflow.get_experiment_by_name(experiment_name)

    if not existing_exp:
        _LOG.info("Creating new experiment")
        experiment_id = mlflow.create_experiment(experiment_name)
        metrics['experiment_id'] = experiment_id
        mlflow.set_experiment(experiment_name)
    else:
        _LOG.info("Found experiment")
        current_experiment=dict(existing_exp)
        experiment_id = current_experiment['experiment_id']
        metrics['experiment_id'] = experiment_id

    with mlflow.start_run(run_name="dangennadevich", experiment_id = experiment_id, description = "parent") as parent_run:
        run_id = parent_run.info.run_id
        metrics['run_id'] = run_id

        return metrics


def get_data(**kwargs) -> Dict[str, Any]:
    """Загрузка данных и отправка в s3"""

    ti = kwargs['ti']
    metrics = ti.xcom_pull(task_ids='init')

    metrics['time_of_start_get_data'] = datetime.now().strftime('%d.%m.%y %H:%M:%S')

    _LOG.info(f"Start download.")


    # Загрузим данные из sklearn
    data = fetch_openml(name="house_prices", as_frame=True)
    data = pd.concat([data['data'], data['target']], axis=1)

    # Выгрузим данные в s3  
    s3_hook = S3Hook("s3_connection")
    filebuffer = io.BytesIO()
    data.to_pickle(filebuffer)
    filebuffer.seek(0)

    s3_key = f"Danila_Iugai/project/datasets/not_prepared_data.pkl" # Путь к файлу на S3
    s3_hook.load_file_obj(
        file_obj=filebuffer,
        key=s3_key,
        bucket_name=BUCKET,
        replace=True,
    )

    metrics['time_of_end_get_data'] = datetime.now().strftime('%H:%M:%S %d.%m.%y')
    _LOG.info("Download finish.")


    metrics['file_path'] = s3_key
    metrics['data_size'] = data.shape 

    return metrics



def prepare_data(**kwargs) -> Dict[str, Any]:
    """Обработка данных, подготовка для моделирвоания"""

    # Получаем данные из XCom из задачи get_data
    ti = kwargs['ti']
    metrics = ti.xcom_pull(task_ids='get_data')

    metrics['time_of_start_prepare_data'] = datetime.now().strftime('%d.%m.%y %H:%M:%S')

    _LOG.info("Download data.")


    # Загрузим датасет из S3
    s3_hook = S3Hook("s3_connection")
    file = s3_hook.download_file(key=metrics['file_path'], bucket_name=BUCKET)
    data = pd.read_pickle(file)
    
    _LOG.info("Download finish. Start prepaid data.")
    # Разделим данные на X и y
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

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Для линейной регрессии стандратизируем признаки, для остальных просто разбиение
    _LOG.info("Prepaid for LR")
    
    # Стандартизация признаков, нарочно не стандартизирую y
    s_scaller = StandardScaler()
    X_train_lr = s_scaller.fit_transform(X_train)
    X_test_lr = s_scaller.transform(X_test)

    # Преобразую в DF и сложу в diсt для удобства
    X_train_lr = pd.DataFrame(X_train_lr, columns=X_columns)
    X_test_lr = pd.DataFrame(X_test_lr, columns=X_columns)

    dataset_dict = {
        'X_train_lr' : X_train_lr,
        'X_test_lr' : X_test_lr,
    }


    _LOG.info("Prepaid for wood model")
    # Тут просто разделим данные и сложим в diсt
    dataset_dict['X_train_wood'] = X_train
    dataset_dict['X_test_wood'] = X_test

    _LOG.info("Prepaid y_true data")
    dataset_dict['y_train'] = y_train
    dataset_dict['y_test'] = y_test


    _LOG.info("Prepaid end. Send to s3")
    # Выгрузим данные в s3, каждый DataFrame в цикле
    for df_name in dataset_dict.keys():
        
        filebuffer = io.BytesIO()
        dataset_dict[df_name].to_pickle(filebuffer)
        filebuffer.seek(0)

        s3_key = f"Danila_Iugai/project/datasets/{df_name}_prepared_data.pkl" # Путь к файлу на S3
        s3_hook.load_file_obj(
            file_obj=filebuffer,
            key=s3_key,
            bucket_name=BUCKET,
            replace=True,
        )

    metrics['X_columns'] = json.dumps({'nums': X_columns.tolist()})

    _LOG.info("Upload to s3 done!")
    metrics['time_of_end_prepare_data'] = datetime.now().strftime('%d.%m.%y %H:%M:%S')

    return metrics


def train_model(**kwargs) -> Dict[str, Any]:
    """Обучения одной модели
    
    :model_name: название модели, одно из "random_forest", "linear_regression", "decision_tree"
    """

    # Получаем данные из XCom из задачи prepare_data
    ti = kwargs['ti']
    metrics = ti.xcom_pull(task_ids='prepare_data')

    metrics['time_of_start_train_model'] = datetime.now().strftime('%d.%m.%y %H:%M:%S')

    # Инициализируем mlflow
    configure_mlflow()

    # Извлечение model_name из kwargs
    model_name = kwargs.get('model_name')

    with mlflow.start_run(run_id=metrics['run_id'], experiment_id=metrics['experiment_id']) as parent_run:
        with mlflow.start_run(run_name=model_name, experiment_id=metrics['experiment_id'], nested=True) as child_run:
            
            dataset_dict = {}
            _LOG.info("Download data.")

            # Загрузим датасет из S3
            s3_hook = S3Hook("s3_connection")

            if model_name == 'linear_regression':
                datasets = ['X_train_lr', 'X_test_lr', 'y_train', 'y_test']
            else:
                datasets = ['X_train_wood', 'X_test_wood', 'y_train', 'y_test']

            for df_name in datasets:
                s3_key = f"Danila_Iugai/project/datasets/{df_name}_prepared_data.pkl"

                file = s3_hook.download_file(key=s3_key, bucket_name=BUCKET)
                dataset_dict[df_name] = pd.read_pickle(file)

            model = models[model_name]

            model.fit(X=dataset_dict[f'{datasets[0]}'], y=dataset_dict[f'{datasets[2]}'])

            eval_df = dataset_dict[f'{datasets[1]}'].copy()
            y_pred = model.predict(X=eval_df)

            # Создадим датасет для mlflow.evaluate
            eval_df["target"] = dataset_dict[f'{datasets[3]}'].values
        
            # Сохраним результаты обучения с помощью MLFlow.
            signature = infer_signature(eval_df, y_pred)
            model_info = mlflow.sklearn.log_model(sk_model = model, artifact_path = model_name, signature=signature)

            mlflow.evaluate(
                model=model_info.model_uri,
                data=eval_df,
                targets="target",
                model_type="regressor",
                evaluators=["default"],
            )

            # отдельно метрики для S3
            metrics[f'MSE_{model_name}'] = mean_squared_error(y_true=dataset_dict[f'{datasets[3]}'], y_pred=y_pred)
            metrics[f'MAE_{model_name}'] = mean_absolute_error(y_true=dataset_dict[f'{datasets[3]}'], y_pred=y_pred)

            metrics['time_of_end_train_model'] = datetime.now().strftime('%d.%m.%y %H:%M:%S')

            return metrics


def save_results(**kwargs) -> None:
    """Сохранение метрик"""

    # Получаем данные из XCom из задачи train_model
    ti = kwargs['ti']
    metrics = ti.xcom_pull(task_ids='train_model')
    
    _LOG.info("Start upload data to s3")

    # Создадим временный файл
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        json.dump(metrics, temp_file)

    # Загрузим S3
    s3_hook = S3Hook("s3_connection")
    s3_key = f"Danila_Iugai/project/results/json_metrics.pkl"
    s3_hook.load_file(temp_file.name, key=s3_key, bucket_name=BUCKET, replace=True)
    
    # Удаляем временный файл
    os.unlink(temp_file.name)

    _LOG.info("END!")
   
task_init = PythonOperator(task_id=f"init", python_callable=init, provide_context=True, dag=dag) 

task_get_data = PythonOperator(task_id=f"get_data", python_callable=get_data, provide_context=True, dag=dag)

task_prepare_data = PythonOperator(task_id=f"prepare_data", python_callable=prepare_data, provide_context=True, dag=dag)

training_model_tasks = [
    PythonOperator(task_id=f"random_forest_train_model", python_callable=train_model, provide_context=True, dag=dag, op_kwargs={'model_name': 'random_forest'}),
    PythonOperator(task_id=f"linear_regression_train_model", python_callable=train_model, provide_context=True, dag=dag, op_kwargs={'model_name': 'linear_regression'}),
    PythonOperator(task_id=f"decision_tree_train_model", python_callable=train_model, provide_context=True, dag=dag, op_kwargs={'model_name': 'decision_tree'})
    ]

task_save_results = PythonOperator(task_id=f"save_results", python_callable=save_results, provide_context=True, dag=dag)

task_init >> task_get_data >> task_prepare_data >> training_model_tasks >> task_save_results