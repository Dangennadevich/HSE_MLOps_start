# HSE_MLOps_start
Проекты в ВШЭ-МОВС по дисциплине "MLOps начало"

___ДЗ1___

Необходимо написать 3 DAG'а внутри 1 py-файла. ML-задача не имеет значения. Необходимо обучить 3 модели: Линейная регрессия, Дерево решений и Случайный лес. 

3 DAG'a -- это 3 разных ML-модели. 
Каждый DAG -- это 5 шагов пайплайна: init, get_gata, prepare_data, train_model, save_results.

Данные между шагами (метрики пайплайна) передаются с помощью XCOM.
Данные для обучения между шагами гоняем через S3.
