# HSE MLOps start
Проекты в ВШЭ-МОВС по дисциплине "MLOps начало"

__Проект__

Проект курса находится в ветке main, в рамках проекта необходимо реализовать обучение 3-х 
мадолей при помощи Airflow, а так же внедрить логирование MLFlow.

Airflow Dag нужно реализовать таким способом, что бы все 3 моделли обучались паралельно.
MLflow - в рамках одного проекта должен запускаться 1 ран с 3 дочерними ранами, где каждый 
дочерний ран - обучение одной модели.

Задача моделирования произвольная.
Проект строится на основе HW1 и HW2.

___ДЗ1___
Находится на ветке hw1

Необходимо написать 3 DAG'а внутри 1 py-файла. ML-задача не имеет значения. Необходимо обучить 3 модели: Линейная регрессия, Дерево решений и Случайный лес. 

3 DAG'a -- это 3 разных ML-модели. 
Каждый DAG -- это 5 шагов пайплайна: init, get_gata, prepare_data, train_model, save_results.

Данные между шагами (метрики пайплайна) передаются с помощью XCOM.
Данные для обучения между шагами гоняем через S3.


___ДЗ2___
Находится на ветке hw2

В рамках ДЗ2 необходимо обучить 3 модели и залогировать это с помощью 
MLFlow.

Обучение 3 моделей в рамках 1 run и 1 эксперимента, используя child run 
для каждой модели

Артефакты сохранить на s3, отобразить метрики в MLFlow.
