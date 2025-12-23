import mlflow
import json
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from mlflow.models import infer_signature
from mlflow.models.evaluation import EvaluationResult

# Проверка и создание эксперимента
experiment_name = "Marsel-2023"
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    print(f"Creating new experiment: {experiment_name}")
    mlflow.create_experiment(experiment_name)
else:
    print(f"Using existing experiment: {experiment_name}")
mlflow.set_experiment(experiment_name)

# Parent run с вашим ником в Telegram
with mlflow.start_run(run_name="lizvladii") as parent_run:
    print("Starting parent run: lizvladii")
    
    # Загрузка данных
    print("Loading California housing dataset...")
    data = fetch_california_housing()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # Нормализация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Конфигурация моделей
    models_config = [
        ("linear_regression", LinearRegression),
        ("decision_tree", DecisionTreeRegressor),
        ("random_forest", RandomForestRegressor)
    ]
    
    # Обучение 3 моделей в child runs
    for model_name, model_class in models_config:
        print(f"\n{'='*50}")
        print(f"Training {model_name.replace('_', ' ').title()}")
        print(f"{'='*50}")
        
        with mlflow.start_run(nested=True, run_name=model_name) as child_run:
            run_id = child_run.info.run_id
            print(f"Child run started: {run_id}")
            
            # Логирование параметров
            mlflow.log_params({
                'model_type': model_name,
                'features': json.dumps(list(feature_names)),
                'train_size': X_train_scaled.shape[0],
                'test_size': X_test_scaled.shape[0],
                'random_state': 42
            })
            
            # Обучение модели
            print("Training model...")
            model = model_class()
            model.fit(X_train_scaled, y_train)
            
            # Предсказания
            print("Making predictions...")
            y_pred = model.predict(X_test_scaled)
            
            # Подготовка сигнатуры для MLFlow
            signature = infer_signature(X_test_scaled, y_pred)
            
            # Логирование модели
            print("Logging model to MLFlow...")
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=model_name,
                signature=signature,
                registered_model_name=f"Marsel-2023_{model_name}"
            )
            
            # Оценка модели с помощью mlflow.evaluate()
            print("Evaluating model with mlflow.evaluate()...")
            eval_results = mlflow.evaluate(
                model=model,
                data=X_test_scaled,
                targets=y_test,
                model_type="regressor",
                evaluators=["default"],
                feature_names=feature_names.tolist(),
                evaluator_config={
                    "log_model_explainability": True,
                    "explainability_algorithm": "kernel"
                }
            )
            
            # Логирование метрик из результатов оценки
            print("Logging evaluation metrics...")
            for metric_name, value in eval_results.metrics.items():
                mlflow.log_metric(metric_name, value)
                print(f"  {metric_name}: {value:.4f}")
            
            # Дополнительные метрики
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            mlflow.log_metrics({
                'training_score': train_score,
                'testing_score': test_score,
                'n_features': X_train_scaled.shape[1],
                'n_samples': X_train_scaled.shape[0]
            })
            
            print(f"Training R²: {train_score:.4f}")
            print(f"Testing R²: {test_score:.4f}")
            
            # Логирование артефактов
            print("Logging artifacts...")
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
            
            print(f"✅ Model {model_name} trained and logged successfully!")
            print(f"🔗 Model URI: runs:/{run_id}/{model_name}")

print("\n🎉 All models trained and logged successfully!")
print(f"📊 View results at: http://localhost:5000/#/experiments/{experiment_name}")