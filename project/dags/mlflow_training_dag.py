import json
import time
import pickle
import io
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.hooks.S3_hook import S3Hook
from airflow.models import Variable
import mlflow
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from mlflow.models import infer_signature
from mlflow.models.evaluation import EvaluationResult

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

def init_mlflow_training(**kwargs):
    """Инициализация MLFlow и пайплайна"""
    ti = kwargs['ti']
    timestamp = datetime.now().isoformat()
    
    # Настройка MLFlow
    mlflow_tracking_uri = Variable.get("MLFLOW_TRACKING_URI", default_var="http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    print(f"MLFlow tracking URI: {mlflow_tracking_uri}")
    
    # Проверка и создание эксперимента
    experiment_name = "Marsel-2023"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Creating new experiment: {experiment_name}")
        mlflow.create_experiment(experiment_name)
    else:
        print(f"Using existing experiment: {experiment_name}")
    mlflow.set_experiment(experiment_name)
    
    metrics = {
        'pipeline_start_time': timestamp,
        'experiment_name': experiment_name,
        'parent_run_name': 'lizvladii',
        'mlflow_tracking_uri': mlflow_tracking_uri
    }
    
    ti.xcom_push(key='init_metrics', value=metrics)
    print("MLFlow initialized successfully!")
    return metrics

def get_and_prepare_data_mlflow(**kwargs):
    """Загрузка и подготовка данных с логированием в MLFlow"""
    ti = kwargs['ti']
    init_metrics = ti.xcom_pull(task_ids='init_mlflow_training', key='init_metrics')
    
    with mlflow.start_run(run_name=init_metrics['parent_run_name'], nested=True):
        # Загрузка данных
        print("Loading California housing dataset...")
        data = fetch_california_housing()
        X, y, feature_names = data.data, data.target, data.feature_names
        
        # Логирование параметров датасета
        mlflow.log_params({
            'dataset': 'california_housing',
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'feature_names': json.dumps(list(feature_names)),
            'target_range': [float(y.min()), float(y.max())]
        })
        
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Нормализация
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Сохранение в S3
        bucket = Variable.get("S3_BUCKET")
        hook = S3Hook('s3_connection')
        
        processed_data = {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names,
            'scaler': scaler
        }
        
        buffer = io.BytesIO()
        pickle.dump(processed_data, buffer)
        buffer.seek(0)
        
        hook.load_file_obj(
            file_obj=buffer,
            key="Marsel-2023/project/datasets/processed_data.pkl",
            bucket_name=bucket,
            replace=True
        )
        print("Processed data saved to S3")
        
        metrics = {
            'dataset_size': X.shape[0],
            'train_size': X_train_scaled.shape[0],
            'test_size': X_test_scaled.shape[0],
            'feature_names': list(feature_names),
            'target_mean': float(y.mean()),
            'target_std': float(y.std())
        }
        
        ti.xcom_push(key='data_metrics', value=metrics)
        return metrics

def train_model_with_mlflow(model_name, model_class, **kwargs):
    """Обучение одной модели с полным логированием в MLFlow"""
    ti = kwargs['ti']
    init_metrics = ti.xcom_pull(task_ids='init_mlflow_training', key='init_metrics')
    data_metrics = ti.xcom_pull(task_ids='get_and_prepare_data_mlflow', key='data_metrics')
    
    # Загрузка данных из S3
    bucket = Variable.get("S3_BUCKET")
    hook = S3Hook('s3_connection')
    
    file_bytes = hook.read_key(
        key="Marsel-2023/project/datasets/processed_data.pkl",
        bucket_name=bucket
    )
    processed_data = pickle.loads(file_bytes.encode('latin1'))
    
    X_train = processed_data['X_train']
    X_test = processed_data['X_test']
    y_train = processed_data['y_train']
    y_test = processed_data['y_test']
    feature_names = processed_data['feature_names']
    
    # MLFlow tracking
    with mlflow.start_run(nested=True, run_name=model_name):
        run_id = mlflow.active_run().info.run_id
        print(f"\n{'='*50}")
        print(f"Training {model_name.replace('_', ' ').title()}")
        print(f"Run ID: {run_id}")
        print(f"{'='*50}")
        
        # Логирование параметров
        mlflow.log_params({
            'model_type': model_name,
            'features': json.dumps(list(feature_names)),
            'train_size': X_train.shape[0],
            'test_size': X_test.shape[0],
            'random_state': 42
        })
        
        # Обучение модели
        print("Training model...")
        start_time = time.time()
        model = model_class()
        model.fit(X_train, y_train)
        end_time = time.time()
        training_time = end_time - start_time
        
        # Предсказания
        print("Making predictions...")
        y_pred = model.predict(X_test)
        
        # Логирование модели
        print("Logging model to MLFlow...")
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=model_name,
            signature=signature,
            registered_model_name=f"Marsel-2023_{model_name}"
        )
        
        # Оценка модели
        print("Evaluating model with mlflow.evaluate()...")
        eval_results = mlflow.evaluate(
            model=model,
            data=X_test,
            targets=y_test,
            model_type="regressor",
            evaluators=["default"],
            feature_names=feature_names.tolist(),
            evaluator_config={
                "log_model_explainability": True,
                "explainability_algorithm": "kernel"
            }
        )
        
        # Сбор метрик
        metrics = eval_results.metrics
        metrics.update({
            'training_time': training_time,
            'training_score': model.score(X_train, y_train),
            'n_features': X_train.shape[1],
            'n_samples': X_train.shape[0]
        })
        
        # Логирование метрик
        print("Logging evaluation metrics...")
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
            print(f"  {metric_name}: {value:.4f}")
        
        # Логирование важности признаков
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            feature_importance = dict(zip(feature_names, np.abs(model.coef_)))
        
        if feature_importance:
            mlflow.log_dict(
                feature_importance, 
                f"{model_name}_feature_importance.json"
            )
            print(f"Logged feature importance for {model_name}")
        
        # Сохранение модели в S3 через MLFlow
        model_uri = f"runs:/{run_id}/{model_name}"
        print(f"Model URI: {model_uri}")
        
        print(f"✅ Model {model_name} trained and logged successfully!")
        return metrics

def create_mlflow_project_dag():
    """Создание DAG для проекта"""
    with DAG(
        dag_id="mlflow_project_training",
        default_args=default_args,
        schedule_interval='0 1 * * *',  # Ежедневно в 1 час ночи
        tags=['mlops', 'mlflow', 'project'],
        catchup=False,
        description='Project DAG: ML training with MLFlow integration'
    ) as dag:
        
        init = PythonOperator(
            task_id='init_mlflow_training',
            python_callable=init_mlflow_training
        )
        
        get_data = PythonOperator(
            task_id='get_and_prepare_data_mlflow',
            python_callable=get_and_prepare_data_mlflow
        )
        
        # Обучение 3 моделей параллельно
        train_tasks = []
        for model_name, model_class in MODEL_CONFIGS.items():
            task = PythonOperator(
                task_id=f'train_{model_name}_mlflow',
                python_callable=train_model_with_mlflow,
                op_kwargs={
                    'model_name': model_name,
                    'model_class': model_class
                }
            )
            train_tasks.append(task)
        
        # Определение порядка выполнения
        init >> get_data
        for task in train_tasks:
            get_data >> task
        
        return dag

# Создание и регистрация DAG
dag_mlflow_project = create_mlflow_project_dag()
globals()['dag_mlflow_project'] = dag_mlflow_project

print("✅ Project DAG created successfully!")
print("📚 DAG ID: mlflow_project_training")
print("⏰ Schedule: Daily at 1:00 AM")