"""
Airflow Lab 1: K-means Clustering Pipeline
DAG Definition
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.lab import load_data, data_preprocessing, build_save_model, load_model_elbow

# Default arguments for all tasks
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 10, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG
with DAG(
    'Airflow_Lab1',
    default_args=default_args,
    description='K-means clustering pipeline with elbow method',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=['lab1', 'clustering', 'kmeans'],
) as dag:
    
    # Task 1: Load data
    load_data_task = PythonOperator(
        task_id='load_data_task',
        python_callable=load_data,
        provide_context=True,
    )
    
    # Task 2: Preprocess data
    preprocess_task = PythonOperator(
        task_id='data_preprocessing_task',
        python_callable=data_preprocessing,
        provide_context=True,
    )
    
    # Task 3: Build and save model
    build_model_task = PythonOperator(
        task_id='build_save_model_task',
        python_callable=build_save_model,
        provide_context=True,
    )
    
    # Task 4: Load model and find optimal clusters
    load_model_task = PythonOperator(
        task_id='load_model_task',
        python_callable=load_model_elbow,
        provide_context=True,
    )
    
    # Define task dependencies (execution order)
    load_data_task >> preprocess_task >> build_model_task >> load_model_task
