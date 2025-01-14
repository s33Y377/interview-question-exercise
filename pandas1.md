```python
import pandas as pd

# Sample data for logon times
logon_data = {
    'user_id': [1, 2, 3],
    'logon_time': ['2025-01-14 08:00:00', '2025-01-14 09:00:00', '2025-01-14 10:00:00']
}

# Sample data for logoff times
logoff_data = {
    'user_id': [1, 2, 3],
    'logoff_time': ['2025-01-14 09:00:00', '2025-01-14 10:30:00', '2025-01-14 11:00:00']
}

# Convert logon_time and logoff_time to datetime format
logon_df = pd.DataFrame(logon_data)
logon_df['logon_time'] = pd.to_datetime(logon_df['logon_time'])

logoff_df = pd.DataFrame(logoff_data)
logoff_df['logoff_time'] = pd.to_datetime(logoff_df['logoff_time'])

# Merge logon and logoff data on 'user_id'
merged_df = pd.merge(logon_df, logoff_df, on='user_id')

# Calculate session duration by subtracting logon_time from logoff_time
merged_df['session_duration'] = merged_df['logoff_time'] - merged_df['logon_time']

# Display the result
print(merged_df)
```

Once you have the logon and logoff data in a pandas DataFrame, there are many additional analyses, transformations, and operations you can perform depending on your use case. Below are some examples of what can be done with this data:

### 1. **Basic Data Analysis**
   - **Summary Statistics**: You can calculate basic summary statistics such as the total number of sessions, average session duration, etc.
     ```python
     # Total number of sessions
     total_sessions = merged_df['user_id'].nunique()
     
     # Average session duration
     avg_session_duration = merged_df['session_duration'].mean()
     
     print("Total Sessions:", total_sessions)
     print("Average Session Duration:", avg_session_duration)
     ```

### 2. **Session Duration Analysis**
   - **Longest and Shortest Sessions**: Find the longest and shortest session durations.
     ```python
     longest_session = merged_df.loc[merged_df['session_duration'].idxmax()]
     shortest_session = merged_df.loc[merged_df['session_duration'].idxmin()]
     
     print("Longest Session:", longest_session)
     print("Shortest Session:", shortest_session)
     ```

   - **Session Duration Distribution**: Plot a histogram or other visualizations to understand the distribution of session durations.
     ```python
     import matplotlib.pyplot as plt

     merged_df['session_duration'].dt.total_seconds().plot(kind='hist', bins=10)
     plt.title('Session Duration Distribution')
     plt.xlabel('Duration (seconds)')
     plt.ylabel('Frequency')
     plt.show()
     ```

### 3. **Time-Based Analysis**
   - **Sessions by Time of Day**: Analyze the number of sessions that start at different times of the day (e.g., morning, afternoon).
     ```python
     merged_df['logon_time_hour'] = merged_df['logon_time'].dt.hour
     sessions_by_time_of_day = merged_df.groupby('logon_time_hour').size()
     print(sessions_by_time_of_day)
     ```

   - **Sessions by Day of Week**: Check which days of the week users log in the most.
     ```python
     merged_df['logon_day_of_week'] = merged_df['logon_time'].dt.day_name()
     sessions_by_day = merged_df.groupby('logon_day_of_week').size()
     print(sessions_by_day)
     ```

### 4. **Session Frequency Analysis**
   - **Frequent Users**: Find which users have the most number of logon/logoff sessions.
     ```python
     frequent_users = merged_df.groupby('user_id').size().sort_values(ascending=False)
     print(frequent_users)
     ```

### 5. **Identify Longest and Shortest Active Periods**
   - **Active Period Analysis**: Calculate the difference between the first logon and last logoff time for each user, to check their total active time span.
     ```python
     merged_df['logon_time'] = pd.to_datetime(merged_df['logon_time'])
     merged_df['logoff_time'] = pd.to_datetime(merged_df['logoff_time'])

     # Group by user and get the first logon and last logoff time
     user_active_periods = merged_df.groupby('user_id').agg(
         first_logon=('logon_time', 'min'),
         last_logoff=('logoff_time', 'max')
     )

     user_active_periods['active_period'] = user_active_periods['last_logoff'] - user_active_periods['first_logon']
     print(user_active_periods)
     ```

### 6. **Time Gap Between Sessions**
   - **Gaps Between Sessions**: Calculate the gap between consecutive logon and logoff sessions for users. This can help identify idle times or breaks between sessions.
     ```python
     merged_df['time_gap'] = merged_df['logon_time'].shift(-1) - merged_df['logoff_time']
     print(merged_df[['user_id', 'logon_time', 'logoff_time', 'time_gap']])
     ```

### 7. **Session Overlap Analysis**
   - **Overlapping Sessions**: If there are multiple users, you might want to analyze if there are overlapping sessions. This involves checking if a user's session overlaps with another user's session during the same time frame.
     ```python
     # To find overlaps, you'll need to compare logon and logoff times
     def is_overlap(row1, row2):
         return row1['logon_time'] < row2['logoff_time'] and row1['logoff_time'] > row2['logon_time']

     overlaps = []
     for idx1, row1 in merged_df.iterrows():
         for idx2, row2 in merged_df.iterrows():
             if idx1 != idx2 and is_overlap(row1, row2):
                 overlaps.append((row1['user_id'], row2['user_id'], row1['logon_time'], row1['logoff_time'], row2['logon_time'], row2['logoff_time']))
     
     overlap_df = pd.DataFrame(overlaps, columns=['user_id_1', 'user_id_2', 'logon_1', 'logoff_1', 'logon_2', 'logoff_2'])
     print(overlap_df)
     ```

### 8. **Handling Missing Data**
   - **Handle Missing Logoff Times**: Sometimes users may not have a recorded logoff time. You can handle this by filling or imputing missing values.
     ```python
     merged_df['logoff_time'].fillna(pd.Timestamp('2025-01-14 23:59:59'), inplace=True)
     ```

   - **Handle Missing Logon Times**: Similarly, missing logon times could be handled by backfilling or other methods.
     ```python
     merged_df['logon_time'].fillna(method='bfill', inplace=True)
     ```

### 9. **Exporting Results**
   - **Export to CSV/Excel**: You can export your final DataFrame to CSV or Excel for further analysis or reporting.
     ```python
     merged_df.to_csv('session_analysis.csv', index=False)
     merged_df.to_excel('session_analysis.xlsx', index=False)
     ```

### 10. **Advanced Time Series Analysis**
   - **Rolling Windows**: You can analyze user activity within rolling time windows (e.g., 7-day activity).
     ```python
     merged_df['logon_time'] = pd.to_datetime(merged_df['logon_time'])
     merged_df.set_index('logon_time', inplace=True)
     rolling_sessions = merged_df['session_duration'].rolling('7D').sum()
     print(rolling_sessions)
     ```

### 11. **User Retention Analysis**
   - **Session Frequency Over Time**: Track the frequency of sessions per user over time and analyze how often they log on (e.g., daily, weekly).
     ```python
     merged_df['logon_date'] = merged_df['logon_time'].dt.date
     sessions_per_day = merged_df.groupby('logon_date').size()
     print(sessions_per_day)
     ```

---

### Conclusion
The pandas DataFrame provides a rich platform for conducting detailed session analysis, identifying user behavior, monitoring session durations, and performing various data processing tasks. Depending on your needs, you can perform statistical analysis, time-based analysis, or even create sophisticated visualizations. The key is to tailor your analysis to the specific goals you're trying to achieve (e.g., identifying user trends, improving system performance, etc.).


---
---


Pandas is a powerful library in Python used for data manipulation and analysis. It is widely used in data science, machine learning, and general data processing tasks. Below is a comprehensive overview of key concepts in pandas, from basic to advanced, along with examples and expected outputs.

---

## **1. Basics of Pandas:**

### **1.1. Importing Pandas**
```python
import pandas as pd
```

### **1.2. Creating a DataFrame**
A DataFrame is the primary data structure in pandas, which can be thought of as a table with rows and columns.

```python
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [24, 27, 22, 32],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Miami']
}

df = pd.DataFrame(data)
print(df)
```

**Output:**
```
       Name  Age         City
0     Alice   24     New York
1       Bob   27  Los Angeles
2   Charlie   22      Chicago
3     David   32        Miami
```

### **1.3. Accessing Columns**
You can access columns by their names.

```python
# Access a single column
print(df['Name'])
```

**Output:**
```
0      Alice
1        Bob
2    Charlie
3      David
Name: Name, dtype: object
```

### **1.4. Accessing Rows**
Rows can be accessed using `.iloc[]` or `.loc[]`.

```python
# Access a row by index position (iloc)
print(df.iloc[2])

# Access a row by label (loc)
print(df.loc[1])
```

**Output for `.iloc[2]`:**
```
Name     Charlie
Age           22
City      Chicago
Name: 2, dtype: object
```

**Output for `.loc[1]`:**
```
Name              Bob
Age               27
City    Los Angeles
Name: 1, dtype: object
```

### **1.5. Descriptive Statistics**
Pandas provides simple functions to get summary statistics.

```python
print(df.describe())
```

**Output:**
```
             Age
count   4.000000
mean   26.250000
std     4.535429
min    22.000000
25%    23.000000
50%    25.500000
75%    28.500000
max    32.000000
```

---

## **2. Intermediate Concepts**

### **2.1. Sorting Data**
You can sort data by column values.

```python
# Sort by Age in ascending order
print(df.sort_values('Age'))
```

**Output:**
```
       Name  Age         City
2   Charlie   22      Chicago
0     Alice   24     New York
1       Bob   27  Los Angeles
3     David   32        Miami
```

### **2.2. Filtering Data**
Filtering data is often used to select rows based on conditions.

```python
# Select rows where Age > 25
print(df[df['Age'] > 25])
```

**Output:**
```
    Name  Age         City
1    Bob   27  Los Angeles
3  David   32        Miami
```

### **2.3. Adding New Columns**
You can add new columns by assigning values.

```python
# Add a new column 'Salary'
df['Salary'] = [50000, 60000, 45000, 70000]
print(df)
```

**Output:**
```
       Name  Age         City  Salary
0     Alice   24     New York   50000
1       Bob   27  Los Angeles   60000
2   Charlie   22      Chicago   45000
3     David   32        Miami   70000
```

### **2.4. Handling Missing Data**
Pandas provides methods to handle missing data (`NaN` values).

```python
# Creating a DataFrame with missing values
df2 = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', None],
    'Age': [24, 27, None, 32],
    'City': ['New York', None, 'Chicago', 'Miami']
})

# Fill missing values with a default value
df2_filled = df2.fillna('Unknown')
print(df2_filled)
```

**Output:**
```
       Name    Age         City
0     Alice   24.0     New York
1       Bob   27.0     Unknown
2   Charlie   Unknown   Chicago
3   Unknown   32.0        Miami
```

### **2.5. Grouping and Aggregating**
You can group data by one or more columns and then apply aggregation functions.

```python
# Group by 'City' and calculate the average age
grouped = df.groupby('City')['Age'].mean()
print(grouped)
```

**Output:**
```
City
Chicago        22.0
Los Angeles    27.0
Miami          32.0
New York       24.0
Name: Age, dtype: float64
```

---

## **3. Advanced Concepts**

### **3.1. Merging DataFrames**
You can merge multiple DataFrames similar to SQL joins.

```python
df3 = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Country': ['USA', 'USA', 'USA', 'USA']
})

merged_df = pd.merge(df, df3, on='Name')
print(merged_df)
```

**Output:**
```
       Name  Age         City  Salary Country
0     Alice   24     New York   50000     USA
1       Bob   27  Los Angeles   60000     USA
2   Charlie   22      Chicago   45000     USA
3     David   32        Miami   70000     USA
```

### **3.2. Pivot Tables**
Pivot tables help to summarize data by aggregating values.

```python
# Pivoting the data to summarize by 'City' and 'Age'
pivot = df.pivot_table(values='Age', index='City', aggfunc='mean')
print(pivot)
```

**Output:**
```
              Age
City             
Chicago        22
Los Angeles    27
Miami          32
New York       24
```

### **3.3. Applying Functions**
You can apply custom functions to columns or rows using `.apply()`.

```python
# Apply a function to each element of the 'Age' column
df['Age_in_5_years'] = df['Age'].apply(lambda x: x + 5)
print(df)
```

**Output:**
```
       Name  Age         City  Salary  Age_in_5_years
0     Alice   24     New York   50000              29
1       Bob   27  Los Angeles   60000              32
2   Charlie   22      Chicago   45000              27
3     David   32        Miami   70000              37
```

### **3.4. Working with Time Series Data**
Pandas offers powerful tools for working with dates and times.

```python
# Create a DateTime index
date_range = pd.date_range(start='2023-01-01', periods=4, freq='D')
df_time = pd.DataFrame({
    'Date': date_range,
    'Temperature': [32, 31, 30, 29]
})
print(df_time)
```

**Output:**
```
        Date  Temperature
0 2023-01-01           32
1 2023-01-02           31
2 2023-01-03           30
3 2023-01-04           29
```

### **3.5. Handling Categorical Data**
Pandas offers efficient handling of categorical variables.

```python
# Convert 'City' to categorical type
df['City'] = pd.Categorical(df['City'])
print(df.dtypes)
```

**Output:**
```
Name              object
Age                int64
City           category
Salary             int64
dtype: object
```

---

## **4. Optimizing Performance**

### **4.1. Using `Cython` or `NumPy` for Speed**
Pandas is built on top of NumPy, and operations can be optimized by using `NumPy` functions directly on DataFrames or Series when possible.

```python
import numpy as np
# Example: Apply NumPy's square root function to a column
df['Age_sqrt'] = np.sqrt(df['Age'])
print(df)
```

**Output:**
```
       Name  Age         City  Salary  Age_sqrt
0     Alice   24     New York   50000  4.898979
1       Bob   27  Los Angeles   60000  5.196152
2   Charlie   22      Chicago   45000  4.690415
3     David   32        Miami   70000  5.656854
```

### **4.2. Working with Large Datasets Efficiently**
You can read large datasets in chunks and process them efficiently.

```python
# Reading a CSV file in chunks
chunksize = 10000


for chunk in pd.read_csv('large_file.csv', chunksize=chunksize):
    print(chunk.head())
```

---

## Conclusion:
This covers the essential concepts in pandas, from basic DataFrame creation to advanced techniques like merging, pivoting, and handling time series data. Mastering pandas opens up the possibility for efficient data analysis and manipulation in Python.



---
---




