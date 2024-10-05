from airflow.models import DAG, Variable
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import DAG, Variable
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Any, Dict, Literal
from datetime import datetime, timedelta

import json
import pandas as pd
import logging
import io
import tempfile
import os


_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

BUCKET = Variable.get("S3_BUCKET")
DEFAULT_ARGS = {
    "owner": "Iugai Danila",
    "retry": 3,
    "retry_delay": timedelta(minutes=1)
}

model_names = ["random_forest", "linear_regression", "desicion_tree"]
models = dict(
    zip(model_names, [
        RandomForestRegressor(),
        LinearRegression(),
        DecisionTreeRegressor(),
    ]))



def init(m_name: Literal["random_forest", "linear_regression", "decision_tree"]) -> None:
    """Лог старта пайплайна и название модели"""
    time_of_start_pipeline = datetime.now().strftime('%H:%M:%S %d.%m.%y')
    _LOG.info(f"Train pipeline for model {m_name} started.")
    
    return {
        "start_pipeline_time" : time_of_start_pipeline, # таймстемп запуска пайплайна
        "model_name" : m_name # название обучаемой модели
        }

def get_data(**kwargs) -> Dict[str, Any]:
    """Загрузка данных и отправка в s3"""
    time_of_start_get_data = datetime.now().strftime('%d.%m.%y %H:%M:%S')

    # Получаем данные из XCom из задачи init
    task_id = kwargs['task_instance'].task_id.replace("_get_data", "_init")
    xcom_value = kwargs['ti'].xcom_pull(task_ids=task_id)
    model_name = xcom_value.get('model_name') if xcom_value else 'ERROR'

    _LOG.info(f"Start download.")

    # Загрузим данные из sklearn
    data = fetch_openml(name="house_prices", as_frame=True)
    data = pd.concat([data['data'], data['target']], axis=1)

    # Выгрузим данные в s3  
    s3_hook = S3Hook("s3_connection")
    filebuffer = io.BytesIO()
    data.to_pickle(filebuffer)
    filebuffer.seek(0)

    s3_key = f"Danila_Iugai/{model_name}/datasets/not_prepared_data.pkl" # Путь к файлу на S3
    s3_hook.load_file_obj(
        file_obj=filebuffer,
        key=s3_key,
        bucket_name=BUCKET,
        replace=True,
    )

    time_of_end_get_data = datetime.now().strftime('%H:%M:%S %d.%m.%y')
    _LOG.info("Download finish.")

    return {
        'time_of_start_get_data': time_of_start_get_data,
        'time_of_end_get_data': time_of_end_get_data,
        'file_path': s3_key,  
        'data_size': data.shape 
    }



def prepare_data(**kwargs) -> Dict[str, Any]:
    """Обработка данных, подготовка для моделирвоания"""
    _LOG.info("Download data.")
    time_of_start_prepare_data = datetime.now().strftime('%d.%m.%y %H:%M:%S')

    # Получаем данные из XCom из задачи init
    task_id = kwargs['task_instance'].task_id.replace("_prepare_data", "_init")
    xcom_value = kwargs['ti'].xcom_pull(task_ids=task_id)
    model_name = xcom_value.get('model_name') if xcom_value else 'ERROR'

    # Получаем данные из XCom из задачи get_data
    task_id = kwargs['task_instance'].task_id.replace("_prepare_data", "_get_data")
    xcom_value = kwargs['ti'].xcom_pull(task_ids=task_id)
    s3_path_not_prepaid_data = xcom_value.get('file_path') if xcom_value else 'ERROR'

    # Загрузим датасет из S3
    s3_hook = S3Hook("s3_connection")
    file = s3_hook.download_file(key=s3_path_not_prepaid_data, bucket_name=BUCKET)
    data = pd.read_pickle(file)
    
    _LOG.info("Download finish. Start prepaid data.")
    # Разделим данные на X и y
    y = data['SalePrice']
    X = data.drop(columns='SalePrice')

    # оставлю тольку булевые и числовые признаки
    X = X.select_dtypes(include=['number', 'bool'])
    X_columns = X.columns

    # Изюавлюсь от пропусков - найдем колонки с пропуском и заполним значение модой
    none_in_columns = (X.isna().sum()>0)*1
    none_in_columns = none_in_columns[none_in_columns>0].index

    for col_with_nan in none_in_columns:
        inpit_val = X[col_with_nan].mode()[0]
        X[col_with_nan] = X[col_with_nan].fillna(inpit_val)

    # Для линейной регрессии стандратизируем признаки, для остальных просто разбиение
    if model_name == 'linear_regression':
        _LOG.info("Prepaid for LR")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Стандартизация признаков, нарочно не стандартизирую y
        s_scaller = StandardScaler()
        X_train = s_scaller.fit_transform(X_train)
        X_test = s_scaller.transform(X_test)

        # Преобразую в DF и сложу в diсt для удобства
        X_train = pd.DataFrame(X_train, columns=X_columns)
        X_test = pd.DataFrame(X_test, columns=X_columns)

        dataset_dict = {
            'X_train' : X_train,
            'X_test' : X_test,
            'y_train' : y_train,
            'y_test' : y_test
        }

    # Закостылим пропуски

    else:
        _LOG.info("Prepaid for wood model")

        # Тут просто разделим данные и сложим в diсt
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        dataset_dict = {
            'X_train' : X_train,
            'X_test' : X_test,
            'y_train' : y_train,
            'y_test' : y_test
        }

    _LOG.info("Prepaid end. Send to s3")
    # Выгрузим данные в s3, каждый DataFrame в цикле
    for df_name in dataset_dict.keys():
        
        filebuffer = io.BytesIO()
        dataset_dict[df_name].to_pickle(filebuffer)
        filebuffer.seek(0)

        s3_key = f"Danila_Iugai/{model_name}/datasets/{df_name}_prepared_data.pkl" # Путь к файлу на S3
        s3_hook.load_file_obj(
            file_obj=filebuffer,
            key=s3_key,
            bucket_name=BUCKET,
            replace=True,
        )

    X_columns = json.dumps({'nums': X_columns.tolist()})

    _LOG.info("Upload to s3 done!")
    time_of_end_prepare_data = datetime.now().strftime('%d.%m.%y %H:%M:%S')

    return {
        'time_of_start_prepare_data': time_of_start_prepare_data,
        'time_of_end_prepare_data': time_of_end_prepare_data,
        'feature_names': X_columns
    }


def train_model(**kwargs) -> Dict[str, Any]:
    """Обучения модели"""
    _LOG.info("Download data.")
    time_of_start_train_model = datetime.now().strftime('%d.%m.%y %H:%M:%S')

    # Получаем данные из XCom из задачи init
    task_id = kwargs['task_instance'].task_id.replace("_train_model", "_init")
    xcom_value = kwargs['ti'].xcom_pull(task_ids=task_id)
    model_name = xcom_value.get('model_name') if xcom_value else 'ERROR'

    # Тут будут датасеты
    dataset_dict = {}

    # Загрузим датасет из S3
    s3_hook = S3Hook("s3_connection")

    for df_name in ['X_train', 'X_test', 'y_train', 'y_test']:
        s3_key = f"Danila_Iugai/{model_name}/datasets/{df_name}_prepared_data.pkl"

        file = s3_hook.download_file(key=s3_key, bucket_name=BUCKET)
        dataset_dict[df_name] = pd.read_pickle(file)

    model = models[model_name]

    model.fit(X = dataset_dict['X_train'], y = dataset_dict['y_train'])
    y_pred = model.predict(X = dataset_dict['X_test'])

    MSE = mean_squared_error(y_true=dataset_dict['y_test'], y_pred=y_pred)
    MAE = mean_absolute_error(y_true=dataset_dict['y_test'], y_pred=y_pred)

    _LOG.info(f"mean_squared_error = {MSE}, mean_absolute_error = {MAE}")

    time_of_end_train_model = datetime.now().strftime('%d.%m.%y %H:%M:%S')

    return {
        'time_of_start_train_model': time_of_start_train_model,
        'time_of_end_train_model': time_of_end_train_model,
        'MSE': MSE,
        'MAE': MAE
    }

def save_results(**kwargs) -> None:
    """Сохранение метрик"""
    _LOG.info("Start load data from Xcom")

    # Получаем данные из XCom из задачи init
    task_id = kwargs['task_instance'].task_id.replace("_save_results", "_init")
    xcom_value = kwargs['ti'].xcom_pull(task_ids=task_id)
    model_name = xcom_value.get('model_name') if xcom_value else 'ERROR'
    start_pipeline_time = xcom_value.get('start_pipeline_time') if xcom_value else 'ERROR'

    # Получаем данные из XCom из задачи get_data
    task_id = kwargs['task_instance'].task_id.replace("_save_results", "_get_data")
    xcom_value = kwargs['ti'].xcom_pull(task_ids=task_id)
    time_of_start_get_data = xcom_value.get('time_of_start_get_data') if xcom_value else 'ERROR'
    time_of_end_get_data = xcom_value.get('time_of_end_get_data') if xcom_value else 'ERROR'
    data_size = xcom_value.get('data_size') if xcom_value else 'ERROR'

    # Получаем данные из XCom из задачи prepare_data
    task_id = kwargs['task_instance'].task_id.replace("_save_results", "_prepare_data")
    xcom_value = kwargs['ti'].xcom_pull(task_ids=task_id)
    time_of_start_prepare_data = xcom_value.get('time_of_start_prepare_data') if xcom_value else 'ERROR'
    time_of_end_prepare_data = xcom_value.get('time_of_end_prepare_data') if xcom_value else 'ERROR'
    feature_names = xcom_value.get('feature_names') if xcom_value else 'ERROR'

    # Получаем данные из XCom из задачи train_model
    task_id = kwargs['task_instance'].task_id.replace("_save_results", "_train_model")
    xcom_value = kwargs['ti'].xcom_pull(task_ids=task_id)
    time_of_start_train_model = xcom_value.get('time_of_start_train_model') if xcom_value else 'ERROR'
    time_of_end_train_model = xcom_value.get('time_of_end_train_model') if xcom_value else 'ERROR'
    MSE = xcom_value.get('MSE') if xcom_value else 'ERROR'
    MAE = xcom_value.get('MAE') if xcom_value else 'ERROR'

    metrics_dag_and_model = {
        'model_name' : model_name,
        'MSE' : MSE,
        'MAE' : MAE,
        'start_pipeline_time' : start_pipeline_time,
        'time_of_end_pipeline' : datetime.now().strftime('%d.%m.%y %H:%M:%S'),
        'data_size' : data_size,
        'feature_names' : feature_names,
        'time_of_start_get_data' : time_of_start_get_data,
        'time_of_end_get_data' : time_of_end_get_data,
        'time_of_start_prepare_data' : time_of_start_prepare_data,
        'time_of_end_prepare_data' : time_of_end_prepare_data,
        'time_of_start_train_model' : time_of_start_train_model,
        'time_of_end_train_model' : time_of_end_train_model
    }

    for key in metrics_dag_and_model.keys():
        assert metrics_dag_and_model[key] != 'ERROR', f'Value for key={key} == "ERROR"'
    
    _LOG.info("Start upload data to s3")

    # Создадим временный файл
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        json.dump(metrics_dag_and_model, temp_file)

    # Загрузим S3
    s3_hook = S3Hook("s3_connection")
    s3_key = f"Danila_Iugai/{model_name}/results/json_metrics.pkl"
    s3_hook.load_file(temp_file.name, key=s3_key, bucket_name=BUCKET, replace=True)
    
    # Удаляем временный файл
    os.unlink(temp_file.name)

    _LOG.info("END!")

def create_dag(dag_id: str, m_name: Literal["random_forest", "linear_regression", "decision_tree"]) -> DAG:
    dag = DAG(
        dag_id=dag_id,
        tags=["mlops"],
        schedule_interval="0 1 * * *",
        start_date=days_ago(2),
        catchup=False,
        default_args=DEFAULT_ARGS,
    )

    with dag:
        task_init = PythonOperator(task_id=f"{m_name}_init", python_callable=init, provide_context=True, dag=dag, op_kwargs={"m_name": m_name}) 
        
        task_get_data = PythonOperator(task_id=f"{m_name}_get_data", python_callable=get_data, provide_context=True, dag=dag)

        task_prepare_data = PythonOperator(task_id=f"{m_name}_prepare_data", python_callable=prepare_data, provide_context=True, dag=dag)

        task_train_model = PythonOperator(task_id=f"{m_name}_train_model", python_callable=train_model, provide_context=True, dag=dag)

        task_save_results = PythonOperator(task_id=f"{m_name}_save_results", python_callable=save_results, provide_context=True, dag=dag)

        # Соединяем задачи
        task_init >> task_get_data >> task_prepare_data >> task_train_model >> task_save_results
    return dag




for model_name in models.keys():
    create_dag(f"Danila_Iugai_{model_name}", model_name)