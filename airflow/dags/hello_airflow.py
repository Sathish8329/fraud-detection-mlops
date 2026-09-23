from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime


def say_hello():
    print("Hello from Airflow!")


def say_mlops():
    print("Learning MLOps with Airflow!")


with DAG(
    dag_id="hello_mlops",
    start_date=datetime(2026, 9, 19),
    schedule=None,
    catchup=False,
) as dag:

    hello = PythonOperator(
        task_id="say_hello",
        python_callable=say_hello,
    )

    mlops = PythonOperator(
        task_id="say_mlops",
        python_callable=say_mlops,
    )

    hello >> mlops
