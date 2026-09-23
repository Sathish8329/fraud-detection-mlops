
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule


# --------------------------------------------------
# 1. Validate data
# --------------------------------------------------

def validate_data():
    print("Validating fraud transaction data...")


# --------------------------------------------------
# 2. Get training result
# --------------------------------------------------

def get_training_result(**context):
    import json

    with open("/opt/airflow/models/training_result.json", "r") as f:
        result = json.load(f)

    run_id = result["run_id"]
    accuracy = result["accuracy"]
    model_version = result["model_version"]

    print(f"MLflow Run ID: {run_id}")
    print(f"Accuracy: {accuracy}")
    print(f"Model Version: {model_version}")

    return result


# --------------------------------------------------
# 3. Evaluate model
# --------------------------------------------------

def evaluate_model(**context):

    result = context["ti"].xcom_pull(
        task_ids="get_training_result"
    )

    run_id = result["run_id"]
    accuracy = result["accuracy"]
    model_version = result["model_version"]

    print(f"MLflow Run ID: {run_id}")
    print(f"Accuracy: {accuracy}")
    print(f"Model Version: {model_version}")

    # Model quality threshold
    threshold = 0.90

    print(f"Required Accuracy: {threshold}")

    if accuracy >= threshold:
        print("✅ MODEL APPROVED")
        print(f"Model version {model_version} passed evaluation.")

        return {
            "status": "approved",
            "run_id": run_id,
            "accuracy": accuracy,
            "model_version": model_version,
        }

    else:
        print("❌ MODEL REJECTED")
        print(f"Model version {model_version} failed evaluation.")

        raise ValueError(
            f"Model rejected. Accuracy {accuracy} is below "
            f"required threshold {threshold}."
        )


# --------------------------------------------------
# 4. Register model
# --------------------------------------------------

def register_model(**context):

    result = context["ti"].xcom_pull(
        task_ids="evaluate_model"
    )

    if not result:
        raise ValueError("No approved model information found.")

    status = result["status"]
    run_id = result["run_id"]
    accuracy = result["accuracy"]
    model_version = result["model_version"]

    if status != "approved":
        raise ValueError("Model was not approved.")

    print("===================================")
    print("MODEL REGISTRATION")
    print("===================================")

    print(f"Status        : {status}")
    print(f"MLflow Run ID : {run_id}")
    print(f"Accuracy      : {accuracy}")
    print(f"Model Version : {model_version}")

    print(
        f"🚀 Model version {model_version} is approved "
        "and ready for deployment."
    )

# --------------------------------------------------
# 5. Failure notification
# --------------------------------------------------

def failure_notification():
    print("🚨 Fraud MLOps pipeline failed!")


# --------------------------------------------------
# DAG
# --------------------------------------------------

with DAG(
    dag_id="fraud_mlops",
    start_date=datetime(2026, 9, 19),
    schedule=None,
    catchup=False,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=2),
    },
) as dag:

    # --------------------------------------------------
    # Validate data
    # --------------------------------------------------

    validate = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )

    # --------------------------------------------------
    # Train model
    # --------------------------------------------------

    train = BashOperator(
        task_id="train_model",
        bash_command="cd /opt/airflow && python3 src/train.py",
    )

    # --------------------------------------------------
    # Read MLflow training result
    # --------------------------------------------------

    training_result = PythonOperator(
        task_id="get_training_result",
        python_callable=get_training_result,
    )

    # --------------------------------------------------
    # Evaluate model
    # --------------------------------------------------

    evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    # --------------------------------------------------
    # Register model
    # --------------------------------------------------

    register = PythonOperator(
        task_id="register_model",
        python_callable=register_model,
    )

    # --------------------------------------------------
    # Failure notification
    # --------------------------------------------------

    failure = PythonOperator(
        task_id="failure_notification",
        python_callable=failure_notification,
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    # --------------------------------------------------
    # Dependencies
    # --------------------------------------------------

    validate >> train >> training_result >> evaluate >> register

    train >> failure
