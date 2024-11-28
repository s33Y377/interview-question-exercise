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
