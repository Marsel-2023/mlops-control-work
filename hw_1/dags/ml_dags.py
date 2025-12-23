import json
import time
import pickle
import io
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.hooks.S3_hook import S3Hook
from airflow.models import Variable
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Настройки DAG
default_args = {
    'owner': 'Bakiev Marsel',
    'start_date': datetime(2023, 1, 1),
    'retries': 3,
    'retry_delay': timedelta(minutes=1),
}

# Конфигурация моделей
MODEL_CONFIGS = {
    'linear_regression': LinearRegression,
    'decision_tree': DecisionTreeRegressor,
    'random_forest': RandomForestRegressor
}

def init_task(model_name, **kwargs):
    """Шаг 1: Инициализация пайплайна"""
    ti = kwargs['ti']
    timestamp = datetime.now().isoformat()
    
    metrics = {
        'model_name': model_name,
        'pipeline_start_time': timestamp,
        'owner': 'Bakiev Marsel',
        'tag': 'mlops'
    }
    
    ti.xcom_push(key='init_metrics', value=metrics)
    print(f"Init completed for {model_name}")
    return metrics

def get_data_task(**kwargs):
    """Шаг 2: Загрузка данных"""
    ti = kwargs['ti']
    init_metrics = ti.xcom_pull(task_ids='init', key='init_metrics')
    model_name = init_metrics['model_name']
    
    start_time = time.time()
    # Загружаем данные из открытого источника
    data = fetch_california_housing()
    X, y, feature_names = data.data, data.target, data.feature_names
    end_time = time.time()
    
    # Сохраняем данные в S3
    bucket = Variable.get("S3_BUCKET")
    s3_path = f"Marsel-2023/{model_name}/datasets/raw_data.pkl"
    hook = S3Hook('s3_connection')
    
    buffer = io.BytesIO()
    pickle.dump((X, y, feature_names), buffer)
    buffer.seek(0)
    
    hook.load_file_obj(
        file_obj=buffer,
        key=s3_path,
        bucket_name=bucket,
        replace=True
    )
    print(f"Data saved to S3: {s3_path}")
    
    metrics = {
        'start_time': start_time,
        'end_time': end_time,
        'duration': end_time - start_time,
        'dataset_size': X.shape[0],
        'dataset_shape': X.shape,
        'target_range': [float(y.min()), float(y.max())]
    }
    
    ti.xcom_push(key='get_data_metrics', value=metrics)
    return metrics

def prepare_data_task(**kwargs):
    """Шаг 3: Подготовка данных"""
    ti = kwargs['ti']
    init_metrics = ti.xcom_pull(task_ids='init', key='init_metrics')
    model_name = init_metrics['model_name']
    
    bucket = Variable.get("S3_BUCKET")
    s3_path = f"Marsel-2023/{model_name}/datasets/raw_data.pkl"
    hook = S3Hook('s3_connection')
    
    # Читаем данные из S3
    file_bytes = hook.read_key(key=s3_path, bucket_name=bucket)
    X, y, feature_names = pickle.loads(file_bytes.encode('latin1'))
    
    start_time = time.time()
    # Нормализация данных
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    end_time = time.time()
    
    # Сохраняем обработанные данные
    processed_path = f"Marsel-2023/{model_name}/datasets/processed_data.pkl"
    buffer = io.BytesIO()
    pickle.dump((X_scaled, y, feature_names), buffer)
    buffer.seek(0)
    
    hook.load_file_obj(
        file_obj=buffer,
        key=processed_path,
        bucket_name=bucket,
        replace=True
    )
    print(f"Processed data saved to S3: {processed_path}")
    
    metrics = {
        'start_time': start_time,
        'end_time': end_time,
        'duration': end_time - start_time,
        'feature_names': list(feature_names),
        'processed_shape': X_scaled.shape,
        'feature_means': scaler.mean_.tolist(),
        'feature_stds': scaler.scale_.tolist()
    }
    
    ti.xcom_push(key='prepare_data_metrics', value=metrics)
    return metrics

def train_model_task(**kwargs):
    """Шаг 4: Обучение модели"""
    ti = kwargs['ti']
    init_metrics = ti.xcom_pull(task_ids='init', key='init_metrics')
    model_name = init_metrics['model_name']
    
    bucket = Variable.get("S3_BUCKET")
    s3_path = f"Marsel-2023/{model_name}/datasets/processed_data.pkl"
    hook = S3Hook('s3_connection')
    
    # Читаем обработанные данные
    file_bytes = hook.read_key(key=s3_path, bucket_name=bucket)
    X, y, feature_names = pickle.loads(file_bytes.encode('latin1'))
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    start_time = time.time()
    # Обучение модели
    model_class = MODEL_CONFIGS[model_name]
    model = model_class()
    model.fit(X_train, y_train)
    end_time = time.time()
    
    # Предсказания и оценка
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Сохраняем модель
    model_path = f"Marsel-2023/{model_name}/results/model.pkl"
    buffer = io.BytesIO()
    pickle.dump(model, buffer)
    buffer.seek(0)
    
    hook.load_file_obj(
        file_obj=buffer,
        key=model_path,
        bucket_name=bucket,
        replace=True
    )
    print(f"Model saved to S3: {model_path}")
    
    metrics = {
        'start_time': start_time,
        'end_time': end_time,
        'duration': end_time - start_time,
        'mse': float(mse),
        'r2': float(r2),
        'train_score': float(model.score(X_train, y_train)),
        'test_score': float(model.score(X_test, y_test)),
        'model_params': str(model.get_params())
    }
    
    ti.xcom_push(key='train_model_metrics', value=metrics)
    return metrics

def save_results_task(**kwargs):
    """Шаг 5: Сохранение результатов"""
    ti = kwargs['ti']
    init_metrics = ti.xcom_pull(task_ids='init', key='init_metrics')
    model_name = init_metrics['model_name']
    
    # Собираем все метрики
    all_metrics = {
        'init': ti.xcom_pull(task_ids='init', key='init_metrics'),
        'get_data': ti.xcom_pull(task_ids='get_data', key='get_data_metrics'),
        'prepare_data': ti.xcom_pull(task_ids='prepare_data', key='prepare_data_metrics'),
        'train_model': ti.xcom_pull(task_ids='train_model', key='train_model_metrics')
    }
    
    bucket = Variable.get("S3_BUCKET")
    s3_path = f"Marsel-2023/{model_name}/results/metrics.json"
    hook = S3Hook('s3_connection')
    
    hook.load_string(
        string_data=json.dumps(all_metrics, indent=2),
        key=s3_path,
        bucket_name=bucket,
        replace=True
    )
    
    print(f"Metrics saved to S3: {s3_path}")
    print("Pipeline completed successfully!")
    return f"Results saved for {model_name}"

# Создание 3 DAG'ов
for model_name in MODEL_CONFIGS.keys():
    with DAG(
        dag_id=f"ml_{model_name}",
        default_args=default_args,
        schedule_interval='0 1 * * *',  # Ежедневно в 1 час ночи
        tags=['mlops'],
        catchup=False,
        description=f'ML Pipeline for {model_name.replace("_", " ").title()}'
    ) as dag:
        
        init = PythonOperator(
            task_id='init',
            python_callable=init_task,
            op_kwargs={'model_name': model_name}
        )
        
        get_data = PythonOperator(
            task_id='get_data',
            python_callable=get_data_task
        )
        
        prepare_data = PythonOperator(
            task_id='prepare_data',
            python_callable=prepare_data_task
        )
        
        train_model = PythonOperator(
            task_id='train_model',
            python_callable=train_model_task
        )
        
        save_results = PythonOperator(
            task_id='save_results',
            python_callable=save_results_task
        )
        
        # Определяем порядок выполнения
        init >> get_data >> prepare_data >> train_model >> save_results
        
        # Регистрируем DAG в глобальном пространстве имен
        globals()[f"dag_{model_name}"] = dag