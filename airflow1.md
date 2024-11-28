To master Apache Airflow, it's important to understand its core concepts and how they are applied in Python. Below, I’ll go through some key concepts in Airflow and provide exercises along with solutions.

### 1. **DAG (Directed Acyclic Graph)**
A DAG is a collection of tasks with defined dependencies that Airflow uses to manage and schedule the execution of tasks.

#### Exercise 1: Create a simple DAG
**Objective**: Write a DAG that runs two tasks sequentially.

**Solution**:
```python
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from datetime import datetime

# Define the DAG
dag = DAG(
    'simple_dag',
    description='A simple Airflow DAG',
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
)

# Define the tasks
start_task = DummyOperator(
    task_id='start',
    dag=dag,
)

end_task = DummyOperator(
    task_id='end',
    dag=dag,
)

# Define the task sequence
start_task >> end_task
```

### 2. **Tasks**
Tasks are the basic units of work in Airflow. They are executed according to the DAG's schedule.

#### Exercise 2: Create a task that prints "Hello, Airflow!"
**Objective**: Create a Python task that prints a message when executed.

**Solution**:
```python
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime

def hello_airflow():
    print("Hello, Airflow!")

# Define the DAG
dag = DAG(
    'hello_airflow_dag',
    description='A DAG to print Hello, Airflow!',
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
)

# Define the task
hello_task = PythonOperator(
    task_id='hello_task',
    python_callable=hello_airflow,
    dag=dag,
)
```

### 3. **Operators**
Operators are pre-built tasks in Airflow that perform specific operations. The most common ones are `PythonOperator`, `BashOperator`, and `DummyOperator`.

#### Exercise 3: Use `BashOperator` to execute a shell command
**Objective**: Create a task that runs a shell command using `BashOperator`.

**Solution**:
```python
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime

# Define the DAG
dag = DAG(
    'bash_operator_dag',
    description='A DAG to run a Bash command',
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
)

# Define the task
bash_task = BashOperator(
    task_id='bash_task',
    bash_command='echo "Hello from Bash!"',
    dag=dag,
)
```

### 4. **Task Dependencies**
Task dependencies determine the order in which tasks are executed in the DAG.

#### Exercise 4: Set task dependencies in a DAG
**Objective**: Create three tasks where Task A runs first, followed by Task B, and finally Task C.

**Solution**:
```python
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from datetime import datetime

# Define the DAG
dag = DAG(
    'task_dependencies_dag',
    description='DAG with task dependencies',
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
)

# Define the tasks
task_a = DummyOperator(
    task_id='task_a',
    dag=dag,
)

task_b = DummyOperator(
    task_id='task_b',
    dag=dag,
)

task_c = DummyOperator(
    task_id='task_c',
    dag=dag,
)

# Set task dependencies
task_a >> task_b >> task_c
```

### 5. **XComs (Cross-Communication)**
XComs allow tasks to share data. This is useful when one task needs to pass data to another.

#### Exercise 5: Use XCom to pass data between tasks
**Objective**: Pass a value from one Python task to another using XCom.

**Solution**:
```python
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime

def push_value(**kwargs):
    value = "Hello, XCom!"
    kwargs['ti'].xcom_push(key='message', value=value)

def pull_value(**kwargs):
    value = kwargs['ti'].xcom_pull(task_ids='push_task', key='message')
    print(f"Pulled value: {value}")

# Define the DAG
dag = DAG(
    'xcom_example_dag',
    description='DAG with XCom for passing data',
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
)

# Define the tasks
push_task = PythonOperator(
    task_id='push_task',
    python_callable=push_value,
    provide_context=True,
    dag=dag,
)

pull_task = PythonOperator(
    task_id='pull_task',
    python_callable=pull_value,
    provide_context=True,
    dag=dag,
)

# Set task dependencies
push_task >> pull_task
```

### 6. **DAG Parameters (Dynamic DAGs)**
DAGs can be dynamic, allowing you to create multiple instances of tasks based on parameters.

#### Exercise 6: Create dynamic tasks using a loop
**Objective**: Create multiple tasks dynamically using a loop.

**Solution**:
```python
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from datetime import datetime

# Define the DAG
dag = DAG(
    'dynamic_tasks_dag',
    description='DAG with dynamic tasks',
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
)

# List of task names
task_names = ['task_1', 'task_2', 'task_3']

# Create tasks dynamically
for task_name in task_names:
    task = DummyOperator(
        task_id=task_name,
        dag=dag,
    )
    # Set dependencies (all tasks will run in parallel)
    if 'prev_task' in locals():
        prev_task >> task
    prev_task = task
```

### 7. **Triggering DAGs**
You can trigger a DAG either manually, via the UI, or programmatically.

#### Exercise 7: Trigger a DAG programmatically
**Objective**: Use Airflow’s CLI to trigger a DAG manually.

**Solution**:
To trigger a DAG manually via CLI, you can use this command:
```bash
airflow dags trigger -d <DAG_ID>
```

For example:
```bash
airflow dags trigger -d hello_airflow_dag
```

### 8. **Sensors**
Sensors are a type of operator that waits for a certain condition to be met before proceeding.

#### Exercise 8: Use `FileSensor` to wait for a file to be available
**Objective**: Wait for a file to be available before proceeding with the task.

**Solution**:
```python
from airflow import DAG
from airflow.operators.sensors import FileSensor
from airflow.operators.dummy_operator import DummyOperator
from datetime import datetime

# Define the DAG
dag = DAG(
    'file_sensor_dag',
    description='DAG with FileSensor',
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
)

# Define the FileSensor
file_sensor = FileSensor(
    task_id='file_sensor',
    filepath='/path/to/your/file.txt',
    poke_interval=10,
    timeout=60 * 5,  # Wait for up to 5 minutes
    dag=dag,
)

# Define a task that runs after the file is available
end_task = DummyOperator(
    task_id='end_task',
    dag=dag,
)

# Set task dependencies
file_sensor >> end_task
```

### Conclusion
These exercises cover some of the core concepts in Apache Airflow, including DAGs, tasks, dependencies, XComs, sensors, and dynamic DAG creation. As you build more complex workflows, you'll combine these concepts in more intricate ways to handle real-world use cases.


Here are some more **advanced concepts** in Apache Airflow, along with exercises and solutions to help you deepen your understanding of how to build more complex workflows and leverage the full power of Airflow. These topics cover features such as **Dynamic DAG Generation**, **SubDAGs**, **Task Groups**, **Branching**, **Trigger Rules**, **Airflow Variables**, and **External Task Sensors**.

---

### 1. **Dynamic DAG Generation**

**Objective**: Generate multiple similar tasks dynamically using a loop to create several tasks with different parameters.

#### Solution:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# Function to process each task
def process_task(task_name):
    print(f"Processing task: {task_name}")

# Define the DAG
dag = DAG(
    'dynamic_dag_example',
    description='DAG with dynamic task generation',
    schedule_interval=None,
    start_date=datetime(2024, 11, 28),
    catchup=False,
)

# List of task names to create dynamically
task_names = ['task_1', 'task_2', 'task_3']

# Dynamically create tasks
for task_name in task_names:
    PythonOperator(
        task_id=task_name,
        python_callable=process_task,
        op_args=[task_name],
        dag=dag,
    )
```

**Explanation**:
- Instead of manually defining each task, we loop over a list of task names and dynamically generate tasks using the `PythonOperator`.
- This is useful for creating similar tasks without repeating code.

---

### 2. **SubDAGs**

**Objective**: Use SubDAGs to group related tasks into a smaller DAG that can be executed within a parent DAG.

#### Solution:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.subdag import SubDagOperator
from datetime import datetime

# Function for tasks in the subDAG
def process_subdag(parent_dag_name, child_dag_name, args):
    subdag = DAG(
        dag_id=child_dag_name,
        default_args=args,
        schedule_interval=None,
    )

    with subdag:
        task_1 = PythonOperator(
            task_id='subdag_task_1',
            python_callable=lambda: print("SubDAG task 1"),
            dag=subdag,
        )
        task_2 = PythonOperator(
            task_id='subdag_task_2',
            python_callable=lambda: print("SubDAG task 2"),
            dag=subdag,
        )
        task_1 >> task_2

    return subdag

# Define the parent DAG
dag = DAG(
    'subdag_example',
    description='Parent DAG with a SubDAG',
    schedule_interval=None,
    start_date=datetime(2024, 11, 28),
    catchup=False,
)

start = DummyOperator(task_id='start', dag=dag)

# Create the SubDAG
subdag_task = SubDagOperator(
    task_id='process_subdag',
    subdag=process_subdag('subdag_example', 'subdag_example.process_subdag', dag.default_args),
    dag=dag,
)

end = DummyOperator(task_id='end', dag=dag)

# Set dependencies
start >> subdag_task >> end
```

**Explanation**:
- A `SubDagOperator` allows you to create a subDAG for a group of related tasks.
- SubDAGs help keep complex DAGs modular and easier to manage.

---

### 3. **Task Groups**

**Objective**: Group related tasks for better visualization in the Airflow UI using Task Groups.

#### Solution:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime

# Function to simulate task processing
def process_task(task_name):
    print(f"Processing task: {task_name}")

# Define the DAG
dag = DAG(
    'task_group_example',
    description='DAG with Task Groups',
    schedule_interval=None,
    start_date=datetime(2024, 11, 28),
    catchup=False,
)

# Create task groups
with TaskGroup("group_1", dag=dag) as group_1:
    task_1 = PythonOperator(
        task_id='task_1',
        python_callable=process_task,
        op_args=['task_1'],
    )
    task_2 = PythonOperator(
        task_id='task_2',
        python_callable=process_task,
        op_args=['task_2'],
    )

with TaskGroup("group_2", dag=dag) as group_2:
    task_3 = PythonOperator(
        task_id='task_3',
        python_callable=process_task,
        op_args=['task_3'],
    )
    task_4 = PythonOperator(
        task_id='task_4',
        python_callable=process_task,
        op_args=['task_4'],
    )

# Define task dependencies
group_1 >> group_2
```

**Explanation**:
- `TaskGroup` allows you to logically group tasks together in the Airflow UI, which is useful for organizing DAGs with many tasks.
- Tasks within a group can have their own dependencies, and the group itself can be treated as a single unit.

---

### 4. **Branching**

**Objective**: Dynamically choose which task to execute next based on some condition.

#### Solution:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.branch import BranchPythonOperator
from datetime import datetime

# Function to decide which branch to execute
def choose_branch(**kwargs):
    if kwargs['execution_date'].minute % 2 == 0:
        return 'branch_1_task'
    else:
        return 'branch_2_task'

# Define the DAG
dag = DAG(
    'branching_example',
    description='DAG with Branching',
    schedule_interval=None,
    start_date=datetime(2024, 11, 28),
    catchup=False,
)

start = DummyOperator(task_id='start', dag=dag)

branching_task = BranchPythonOperator(
    task_id='branching_task',
    python_callable=choose_branch,
    provide_context=True,  # Pass context to the function
    dag=dag,
)

branch_1_task = DummyOperator(task_id='branch_1_task', dag=dag)
branch_2_task = DummyOperator(task_id='branch_2_task', dag=dag)

end = DummyOperator(task_id='end', dag=dag)

# Define task dependencies
start >> branching_task
branching_task >> [branch_1_task, branch_2_task]
branch_1_task >> end
branch_2_task >> end
```

**Explanation**:
- `BranchPythonOperator` allows you to dynamically choose the next task to execute based on some condition, which is useful in scenarios where workflows have conditional execution paths.

---

### 5. **Trigger Rules**

**Objective**: Control task execution based on the state of upstream tasks using trigger rules.

#### Solution:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime

# Define functions for the tasks
def task_1_func():
    print("Task 1 succeeded")

def task_2_func():
    print("Task 2 failed")

def task_3_func():
    print("Task 3 executed")

# Define the DAG
dag = DAG(
    'trigger_rule_example',
    description='DAG with Trigger Rules',
    schedule_interval=None,
    start_date=datetime(2024, 11, 28),
    catchup=False,
)

start = DummyOperator(task_id='start', dag=dag)

task_1 = PythonOperator(
    task_id='task_1',
    python_callable=task_1_func,
    dag=dag,
)

task_2 = PythonOperator(
    task_id='task_2',
    python_callable=task_2_func,
    dag=dag,
)

task_3 = PythonOperator(
    task_id='task_3',
    python_callable=task_3_func,
    dag=dag,
    trigger_rule='all_failed',  # This task runs if all upstream tasks fail
)

end = DummyOperator(task_id='end', dag=dag)

# Set dependencies
start >> [task_1, task_2] >> task_3 >> end
```

**Explanation**:
- `trigger_rule`: This determines when a task should run. In this example, `task_3` runs if all its upstream tasks fail (`all_failed`).
- Other trigger rules include `all_success`, `one_failed`, `none_failed`, etc.

---

### 6. **External Task Sensor**

**Objective**: Wait for a task in another DAG to complete before proceeding with the current DAG.

#### Solution:

```python
from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime

# Define the parent DAG
dag_parent = DAG(
    'parent_dag',
    description='Parent DAG for External Task Sensor Example',
    schedule_interval=None,
    start_date=datetime(2024, 11, 28),
    catchup=False,
)

start_parent = DummyOperator(task_id='start', dag=dag_parent)
end_parent = DummyOperator(task_id='end', dag=dag_parent)

# Define the child DAG
dag_child = DAG(
    'child_dag',
    description='Child DAG for External Task Sensor Example',
    schedule_interval=None,
    start_date=datetime(2024, 11, 

28),
    catchup=False,
)

start_child = DummyOperator(task_id='start', dag=dag_child)
end_child = DummyOperator(task_id='end', dag=dag_child)

# Define the External Task Sensor in the parent DAG
sensor_task = ExternalTaskSensor(
    task_id='sensor_task',
    external_dag_id='child_dag',  # DAG ID to monitor
    external_task_id='end',
    mode='poke',  # You can use poke or reschedule mode
    poke_interval=10,
    timeout=600,
    poke_delay=60,
    dag=dag_parent,
)

# Define dependencies
start_parent >> sensor_task >> end_parent
```

**Explanation**:
- `ExternalTaskSensor`: Waits for a specific task in another DAG to finish before continuing with the current DAG. In this case, the parent DAG waits for the `end` task of the child DAG.

---

### Conclusion

These exercises cover several advanced concepts in Apache Airflow. Mastering these concepts will allow you to:
- Dynamically generate tasks and manage complex DAGs.
- Use subDAGs and task groups for better DAG organization.
- Implement conditional task execution with branching and trigger rules.
- Handle cross-DAG dependencies using external task sensors.

Airflow is incredibly flexible, and with these advanced techniques, you can design highly modular, maintainable, and robust workflows.


Below is an example of how you can create a simple Airflow DAG (Directed Acyclic Graph) that orchestrates a data pipeline using Python.

### Problem:
You need to create an Airflow DAG to orchestrate a data pipeline. The pipeline should:
1. Extract data from a file (for simplicity, assume it's a CSV file).
2. Process the data (e.g., filter rows or compute some transformation).
3. Load the processed data to a destination (e.g., a database or output file).

### Steps:
1. **Install Airflow**: 
   If you don't have Airflow installed, you can install it using pip:
   ```bash
   pip install apache-airflow
   ```

2. **Set up the Airflow environment**: 
   Initialize the Airflow database (once):
   ```bash
   airflow db init
   ```

3. **Define the DAG**: 

   The DAG will contain three tasks:
   - Extract task: Extracts data from a CSV file.
   - Process task: Processes the data (e.g., filters out empty rows).
   - Load task: Loads the processed data to another location (e.g., stores it back in a different file).

4. **Airflow DAG Code**:

   ```python
   from airflow import DAG
   from airflow.operators.python import PythonOperator
   from datetime import datetime
   import pandas as pd

   # Function to extract data from a CSV file
   def extract_data(**kwargs):
       # Simulating extraction from a file
       data = pd.read_csv('/path/to/input_data.csv')
       kwargs['ti'].xcom_push(key='raw_data', value=data)
       print("Data Extracted")
       return data

   # Function to process the data
   def process_data(**kwargs):
       # Get raw data from the previous task
       ti = kwargs['ti']
       raw_data = ti.xcom_pull(task_ids='extract_data', key='raw_data')
       
       # Simulate processing by filtering rows with missing values
       processed_data = raw_data.dropna()
       kwargs['ti'].xcom_push(key='processed_data', value=processed_data)
       print("Data Processed")
       return processed_data

   # Function to load the data (write to a new CSV file in this example)
   def load_data(**kwargs):
       # Get processed data from the previous task
       ti = kwargs['ti']
       processed_data = ti.xcom_pull(task_ids='process_data', key='processed_data')
       
       # Write the processed data to a new CSV
       processed_data.to_csv('/path/to/output_data.csv', index=False)
       print("Data Loaded")

   # Define the DAG
   default_args = {
       'owner': 'airflow',
       'retries': 1,
       'start_date': datetime(2024, 11, 28),
   }

   with DAG(
       dag_id='simple_data_pipeline',
       default_args=default_args,
       description='A simple data pipeline orchestrated by Airflow',
       schedule_interval=None,  # Runs manually
       catchup=False,
   ) as dag:
       
       # Define the tasks
       extract_task = PythonOperator(
           task_id='extract_data',
           python_callable=extract_data,
           provide_context=True,
       )
       
       process_task = PythonOperator(
           task_id='process_data',
           python_callable=process_data,
           provide_context=True,
       )
       
       load_task = PythonOperator(
           task_id='load_data',
           python_callable=load_data,
           provide_context=True,
       )
       
       # Set up task dependencies
       extract_task >> process_task >> load_task
   ```

### Explanation:
- **DAG definition**: The `DAG` is defined using the `DAG` class in Airflow. The DAG has an ID (`simple_data_pipeline`) and is set to run manually (`schedule_interval=None`).
- **PythonOperator**: Each task in the pipeline is created using the `PythonOperator`. These operators execute Python functions when triggered.
- **XCom**: Airflow uses **XCom** (short for "cross-communication") to allow tasks to share data. In this case, XCom is used to pass the extracted and processed data between tasks.
- **Task dependencies**: `extract_task >> process_task >> load_task` sets up the sequence in which the tasks should run. The process task will not run until the extract task has completed, and similarly, the load task will only run after the process task is done.

### How to run:
1. **Start Airflow**:
   Start the Airflow web server and scheduler:
   ```bash
   airflow webserver --port 8080
   airflow scheduler
   ```

2. **Trigger the DAG**:
   You can trigger the DAG manually using the Airflow UI or from the command line:
   ```bash
   airflow dags trigger simple_data_pipeline
   ```

3. **Monitor the DAG**:
   - Open the Airflow UI by visiting `http://localhost:8080` in your browser.
   - You will be able to monitor the status of your DAG and its tasks from there.

### Solution Highlights:
- **Data Extraction**: The `extract_data` function reads data from a CSV file using `pandas.read_csv()`.
- **Data Processing**: The `process_data` function filters out rows with missing values using `dropna()`.
- **Data Loading**: The `load_data` function saves the processed data to a new CSV file using `to_csv()`.

This is a basic example of how Airflow can be used to orchestrate a data pipeline. You can extend it by adding more complex tasks like data validation, transformations, or loading to different data stores (databases, APIs, etc.).


Here’s an advanced Apache Airflow exercise for building a data pipeline, along with its solution:

### **Exercise: Build a Data Pipeline with Apache Airflow**

**Problem Statement:**

You need to build a data pipeline in Apache Airflow that performs the following tasks:

1. **Extract**: Fetch data from a public API (for instance, an API that returns weather data).
2. **Transform**: Clean and transform the data. This may involve filtering out unnecessary fields, renaming columns, and converting data types.
3. **Load**: Store the transformed data into a PostgreSQL database (or another relational database).
4. **Error Handling**: Ensure that the pipeline handles failures gracefully and retries a specified number of times in case of errors.
5. **Logging**: Log the important steps (data extraction, transformation, and loading) and store logs for future reference.
6. **Scheduling**: Schedule the pipeline to run every hour.

---

### **Solution:**

Here’s how to implement this with Apache Airflow.

#### **1. Setup the environment**

Ensure the following dependencies are installed:

```bash
pip install apache-airflow
pip install apache-airflow-providers-postgres
pip install requests
pip install pandas
```

#### **2. Create the DAG**

```python
from airflow import DAG
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.hooks.postgres_hook import PostgresHook
import requests
import pandas as pd
import logging
from datetime import timedelta

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'retries': 3,  # Retry 3 times in case of failure
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,
    'email_on_retry': False,
}

# Define the DAG
dag = DAG(
    'weather_data_pipeline',
    default_args=default_args,
    description='A data pipeline to extract, transform, and load weather data',
    schedule_interval=timedelta(hours=1),
    start_date=days_ago(1),
    catchup=False,
)

# Function to extract weather data from API
def extract_weather_data():
    url = "https://api.openweathermap.org/data/2.5/weather?q=London&appid=your_api_key"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        logging.info("Data extraction successful.")
        return data
    except requests.exceptions.RequestException as e:
        logging.error(f"Error while fetching data: {e}")
        raise

# Function to transform the extracted data
def transform_weather_data(data):
    try:
        # Convert to a pandas DataFrame
        df = pd.json_normalize(data)
        
        # Extract relevant fields
        df = df[['name', 'main.temp', 'main.humidity', 'wind.speed', 'weather.0.description']]
        
        # Rename columns
        df.columns = ['city', 'temperature', 'humidity', 'wind_speed', 'description']
        
        # Convert temperature to Celsius
        df['temperature'] = df['temperature'] - 273.15
        
        logging.info("Data transformation successful.")
        return df
    except Exception as e:
        logging.error(f"Error during data transformation: {e}")
        raise

# Function to load the transformed data into PostgreSQL
def load_weather_data_to_db(df):
    try:
        # PostgreSQL connection
        hook = PostgresHook(postgres_conn_id='my_postgres_conn')
        conn = hook.get_conn()
        cursor = conn.cursor()
        
        # Insert data into PostgreSQL table
        for _, row in df.iterrows():
            cursor.execute("""
                INSERT INTO weather_data (city, temperature, humidity, wind_speed, description)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (city) DO UPDATE SET
                    temperature = EXCLUDED.temperature,
                    humidity = EXCLUDED.humidity,
                    wind_speed = EXCLUDED.wind_speed,
                    description = EXCLUDED.description;
            """, (row['city'], row['temperature'], row['humidity'], row['wind_speed'], row['description']))
        
        conn.commit()
        logging.info("Data loaded successfully into PostgreSQL.")
    except Exception as e:
        logging.error(f"Error during data loading: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

# Define the tasks
extract_task = PythonOperator(
    task_id='extract_weather_data',
    python_callable=extract_weather_data,
    dag=dag,
)

transform_task = PythonOperator(
    task_id='transform_weather_data',
    python_callable=transform_weather_data,
    op_args=['{{ task_instance.xcom_pull(task_ids="extract_weather_data") }}'],
    dag=dag,
)

load_task = PythonOperator(
    task_id='load_weather_data',
    python_callable=load_weather_data_to_db,
    op_args=['{{ task_instance.xcom_pull(task_ids="transform_weather_data") }}'],
    dag=dag,
)

# Task dependencies
extract_task >> transform_task >> load_task
```

### **Explanation:**

1. **DAG definition**: The DAG is scheduled to run every hour with `schedule_interval=timedelta(hours=1)`. We also specify default arguments for retries, email alerts, etc.

2. **Extract task**: This PythonOperator calls the `extract_weather_data` function, which fetches data from the weather API.

3. **Transform task**: The data fetched from the API is passed to the `transform_weather_data` function via `XCom`, where it is cleaned and transformed into the required format (e.g., converting temperature to Celsius).

4. **Load task**: The transformed data is loaded into PostgreSQL using the `load_weather_data_to_db` function.

5. **Error handling**: If any step fails (during extraction, transformation, or loading), the task will retry based on the specified retry logic.

6. **Logging**: Logs are generated at each step, so you can track the progress of the data pipeline.

7. **PostgreSQL**: The database is used to store the processed weather data. You need to configure the PostgreSQL connection (`my_postgres_conn`) in Airflow's Admin interface (`Connections`).

---

### **PostgreSQL Table Structure:**

You need a PostgreSQL table to store the data. Here's an SQL script to create it:

```sql
CREATE TABLE weather_data (
    id SERIAL PRIMARY KEY,
    city VARCHAR(255),
    temperature FLOAT,
    humidity INT,
    wind_speed FLOAT,
    description VARCHAR(255),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(city)
);
```

### **Execution Flow:**

- **Extract**: Every hour, the pipeline pulls the weather data from the API.
- **Transform**: The data is transformed to calculate temperature in Celsius and is filtered to include only the necessary fields.
- **Load**: The transformed data is inserted into the PostgreSQL table. If the data already exists for a city, it updates the record.

This pipeline is modular and can be easily extended with more data sources, transformations, or different storage backends.

---

### **Summary:**

This solution demonstrates a basic but scalable ETL pipeline in Apache Airflow using Python and PostgreSQL. You can build more complex workflows by adding more tasks for data validation, enrichment, or notifications.

