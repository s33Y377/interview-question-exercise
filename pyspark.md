Here are some advanced PySpark exercises designed to help you master PySpark's DataFrame API, RDD transformations, and advanced data processing tasks like aggregations, window functions, machine learning, and optimization.

---

### 1. **Advanced Aggregation with GroupBy and Window Functions**
Use PySpark to perform an advanced aggregation operation by combining the `groupBy` method and window functions. Calculate the average salary per department and rank the employees within each department based on their salary.

**Requirements:**
- Create a DataFrame with the following columns: `employee_id`, `department_id`, `salary`.
- Calculate the average salary per department.
- Rank employees within each department based on their salary (use `row_number()` window function).

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, row_number
from pyspark.sql.window import Window

# Initialize Spark session
spark = SparkSession.builder.master("local").appName("Advanced Aggregation").getOrCreate()

# Sample data
data = [
    (1, 1, 70000),
    (2, 1, 80000),
    (3, 1, 90000),
    (4, 2, 60000),
    (5, 2, 65000),
    (6, 2, 70000)
]

# Define schema
columns = ["employee_id", "department_id", "salary"]

# Create DataFrame
df = spark.createDataFrame(data, columns)

# Calculate average salary per department
avg_salary_df = df.groupBy("department_id").agg(avg("salary").alias("avg_salary"))

# Define a window spec to rank employees within each department based on salary
window_spec = Window.partitionBy("department_id").orderBy(df["salary"].desc())

# Add row_number() to rank employees by salary within department
df_with_ranks = df.withColumn("rank", row_number().over(window_spec))

# Show results
df_with_ranks.show()
```

---

### 2. **Joins with Complex Conditions**
Perform a join between two DataFrames with complex conditions. Consider two DataFrames: `orders` and `products`. You need to join them on `product_id` and filter the result based on the product's price and order quantity.

**Requirements:**
- Create two DataFrames: `orders` (columns: `order_id`, `product_id`, `quantity`) and `products` (columns: `product_id`, `product_name`, `price`).
- Join them on `product_id` and filter orders where the `quantity` is greater than 5 and the `price` is above $100.

```python
from pyspark.sql.functions import col

# Sample data
orders_data = [
    (1, 101, 10),
    (2, 102, 3),
    (3, 103, 8),
    (4, 104, 1)
]

products_data = [
    (101, "Product A", 150),
    (102, "Product B", 50),
    (103, "Product C", 120),
    (104, "Product D", 90)
]

# Define schemas
orders_columns = ["order_id", "product_id", "quantity"]
products_columns = ["product_id", "product_name", "price"]

# Create DataFrames
orders_df = spark.createDataFrame(orders_data, orders_columns)
products_df = spark.createDataFrame(products_data, products_columns)

# Perform the join
joined_df = orders_df.join(products_df, "product_id")

# Filter orders where quantity > 5 and price > 100
filtered_df = joined_df.filter((col("quantity") > 5) & (col("price") > 100))

# Show results
filtered_df.show()
```

---

### 3. **Handling Missing Data and Data Cleaning**
Handle missing data in a large dataset by using various techniques like dropping missing values, filling missing values, and using interpolation.

**Requirements:**
- Create a DataFrame with missing values.
- Perform the following:
  - Drop rows with any missing values.
  - Fill missing values with the mean of the respective column.
  - Interpolate missing values.

```python
from pyspark.sql.functions import mean

# Sample data with missing values
data_with_missing = [
    (1, "Alice", None),
    (2, "Bob", 50),
    (3, "Charlie", 45),
    (4, "David", None)
]

columns = ["id", "name", "age"]

# Create DataFrame
df_with_missing = spark.createDataFrame(data_with_missing, columns)

# 1. Drop rows with any missing values
df_no_missing = df_with_missing.dropna()

# 2. Fill missing values with mean (age column)
mean_age = df_with_missing.select(mean("age")).collect()[0][0]
df_filled = df_with_missing.fillna({"age": mean_age})

# 3. Interpolate missing values (PySpark doesn't directly support interpolation, but this can be done in pandas for example)
# Example: Fill missing values using forward fill in pandas (this step requires conversion to pandas)
pandas_df = df_with_missing.toPandas()
pandas_df["age"].fillna(method="ffill", inplace=True)

# Convert back to Spark DataFrame
df_interpolated = spark.createDataFrame(pandas_df)

# Show results
df_no_missing.show()
df_filled.show()
df_interpolated.show()
```

---

### 4. **Working with UDFs (User Defined Functions)**
Use a UDF to transform a column of data. For example, you could create a UDF to calculate the length of each string in a column and add a new column with the string lengths.

**Requirements:**
- Create a DataFrame with a column containing string values.
- Define a UDF to calculate the length of each string.
- Add a new column to the DataFrame with the string lengths.

```python
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType

# Sample data
data = [("Alice",), ("Bob",), ("Charlie",)]
columns = ["name"]

# Create DataFrame
df = spark.createDataFrame(data, columns)

# Define UDF to calculate string length
def string_length(name):
    return len(name)

# Register UDF
length_udf = udf(string_length, IntegerType())

# Add new column with string lengths
df_with_length = df.withColumn("name_length", length_udf(df["name"]))

# Show results
df_with_length.show()
```

---

### 5. **Building a Recommendation System with ALS**
Use PySpark’s `Alternating Least Squares (ALS)` to build a collaborative filtering recommendation system.

**Requirements:**
- Create a DataFrame with user-item ratings data.
- Train an ALS model to predict ratings.
- Recommend top 3 products for each user.

```python
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# Sample ratings data
ratings_data = [
    (1, 101, 5),
    (1, 102, 3),
    (2, 101, 4),
    (2, 103, 5),
    (3, 102, 2),
    (3, 103, 5)
]

columns = ["user_id", "product_id", "rating"]

# Create DataFrame
ratings_df = spark.createDataFrame(ratings_data, columns)

# Split data into training and test sets
train_df, test_df = ratings_df.randomSplit([0.8, 0.2])

# Build ALS model
als = ALS(maxIter=10, regParam=0.01, userCol="user_id", itemCol="product_id", ratingCol="rating", coldStartStrategy="drop")
model = als.fit(train_df)

# Evaluate the model
predictions = model.transform(test_df)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print(f"Root-Mean-Square Error (RMSE): {rmse}")

# Make recommendations for each user
user_recommendations = model.recommendForAllUsers(3)

# Show the recommendations
user_recommendations.show()
```

---

### 6. **Optimizing Performance with Partitioning**
Optimize a large dataset by adjusting the number of partitions. Perform an action on a large dataset and experiment with different partitioning strategies to improve performance.

**Requirements:**
- Create a large dataset.
- Perform a transformation that requires shuffling (like `groupBy`).
- Experiment with repartitioning before the transformation and observe performance changes.

```python
# Sample data with large size
data = [(i, i % 3) for i in range(1000000)]
columns = ["id", "group"]

# Create DataFrame
df = spark.createDataFrame(data, columns)

# Perform transformation (groupBy) without repartitioning
df_no_repartition = df.groupBy("group").count()

# Repartition the DataFrame to optimize performance
df_repartitioned = df.repartition(8).groupBy("group").count()

# Show the results (timing can be observed via Spark UI or logging)
df_no_repartition.show()
df_repartitioned.show()
```

---

### 7. **Using Spark SQL for Complex Queries**
Write a complex SQL query using Spark SQL to perform a join, aggregation, and filtering operations on a dataset.

**Requirements:**
- Create two DataFrames with relevant data.
- Register the DataFrames as temporary SQL tables.
- Write and execute a complex SQL query that joins the two tables, aggregates data, and filters the result.

```python
# Create DataFrames for sales and

 products
sales_data = [(1, 101, 2), (2, 102, 3), (3, 101, 1), (4, 103, 4)]
products_data = [(101, "Product A"), (102, "Product B"), (103, "Product C")]

columns_sales = ["sale_id", "product_id", "quantity"]
columns_products = ["product_id", "product_name"]

sales_df = spark.createDataFrame(sales_data, columns_sales)
products_df = spark.createDataFrame(products_data, columns_products)

# Register DataFrames as temporary SQL tables
sales_df.createOrReplaceTempView("sales")
products_df.createOrReplaceTempView("products")

# SQL query to join, aggregate, and filter
query = """
SELECT p.product_name, SUM(s.quantity) AS total_sales
FROM sales s
JOIN products p ON s.product_id = p.product_id
GROUP BY p.product_name
HAVING total_sales > 3
"""

# Execute the query
result = spark.sql(query)
result.show()
```

---

These exercises will help you explore the advanced capabilities of PySpark, including aggregations, window functions, UDFs, joins, recommendation systems, performance tuning, and more. Experiment with different optimizations and real-world scenarios to build more efficient data pipelines!
