Here's an advanced SQL interview exercise that covers multiple concepts, including complex joins, subqueries, window functions, aggregation, and optimization techniques. The problem is designed to test your ability to handle a wide range of SQL scenarios.

### Problem: Sales Analysis and Reporting

**Background:**

You are given the following tables:

1. **`Customers`**
   - `customer_id` (INT)
   - `customer_name` (VARCHAR)
   - `join_date` (DATE)
   - `country` (VARCHAR)

2. **`Products`**
   - `product_id` (INT)
   - `product_name` (VARCHAR)
   - `category` (VARCHAR)
   - `price` (DECIMAL)

3. **`Orders`**
   - `order_id` (INT)
   - `customer_id` (INT)
   - `order_date` (DATE)
   - `total_amount` (DECIMAL)

4. **`Order_Items`**
   - `order_item_id` (INT)
   - `order_id` (INT)
   - `product_id` (INT)
   - `quantity` (INT)
   - `price_at_purchase` (DECIMAL)

---

### Requirements:

1. **Customer's Lifetime Value (LTV)**:
   Write a query to calculate the total lifetime value (LTV) of each customer. LTV is the total amount spent by the customer across all their orders.

2. **Top 5 Customers by LTV**:
   Write a query to return the top 5 customers by lifetime value, including their `customer_name`, `LTV`, and the `country` they are from. Sort the results in descending order of LTV.

3. **Monthly Revenue Trend**:
   Write a query to show the total revenue for each month, ordered by the month and year. Include the `month_year` (formatted as `YYYY-MM`), and total revenue for that month.

4. **Products with the Highest Revenue**:
   Write a query to identify the top 5 products by revenue. For each product, return the `product_name`, `category`, and the total revenue generated by the product. The revenue should be the sum of `(price_at_purchase * quantity)` for all orders where that product was purchased.

5. **Yearly Growth in Revenue**:
   Calculate the year-over-year revenue growth percentage. Return the year, total revenue for that year, and the percentage change from the previous year (if applicable). Ensure that the first year shows `NULL` for the percentage change.

6. **Average Order Value (AOV) by Country**:
   Write a query to calculate the average order value by country. Include the `country` and the average order value for each country. Order the result by the average order value in descending order.

7. **Top 3 Selling Products by Region**:
   Write a query to find the top 3 selling products in each country. Return the `country`, `product_name`, `category`, and the total quantity sold for each product. For each country, only the top 3 products by quantity should be returned.

---

### Sample Data:

#### `Customers`
| customer_id | customer_name | join_date  | country |
|-------------|---------------|------------|---------|
| 1           | Alice         | 2021-06-01 | USA     |
| 2           | Bob           | 2022-02-15 | UK      |
| 3           | Carol         | 2023-01-10 | USA     |
| 4           | Dave          | 2021-03-22 | Canada  |

#### `Products`
| product_id | product_name   | category   | price |
|------------|----------------|------------|-------|
| 101        | Laptop         | Electronics| 1000  |
| 102        | Smartphone     | Electronics| 500   |
| 103        | TV             | Electronics| 1500  |
| 104        | Headphones     | Accessories| 200   |

#### `Orders`
| order_id | customer_id | order_date | total_amount |
|----------|-------------|------------|--------------|
| 1001     | 1           | 2023-01-01 | 2000         |
| 1002     | 2           | 2023-01-05 | 1500         |
| 1003     | 3           | 2023-02-11 | 500          |
| 1004     | 1           | 2023-03-15 | 1000         |

#### `Order_Items`
| order_item_id | order_id | product_id | quantity | price_at_purchase |
|---------------|----------|------------|----------|-------------------|
| 1             | 1001     | 101        | 1        | 1000              |
| 2             | 1001     | 102        | 1        | 500               |
| 3             | 1002     | 103        | 1        | 1500              |
| 4             | 1003     | 102        | 1        | 500               |
| 5             | 1004     | 101        | 1        | 1000              |

---

### Tips for Solving:
- For **LTV**, you can join the `Orders` and `Order_Items` tables to calculate the total spent by each customer.
- To calculate **monthly revenue**, you might need to extract the month and year from the `order_date` and group by that.
- For **revenue by product**, use `JOIN` between `Order_Items` and `Products` to calculate the total revenue for each product.
- For **year-over-year growth**, use window functions (`LAG`) to get the revenue for the previous year and calculate the growth.
- **Top N products by region** could be solved using `ROW_NUMBER()` and `PARTITION BY` for each country.

### Advanced SQL Techniques to Use:
- **Window Functions**: `ROW_NUMBER()`, `LAG()`, and `RANK()`.
- **Aggregation**: `SUM()`, `AVG()`, `COUNT()`, and grouping by date or other fields.
- **Joins**: INNER JOIN, LEFT JOIN to combine multiple tables.
- **Subqueries**: For filtering or complex calculations.
- **Date Functions**: `YEAR()`, `MONTH()`, `DATE_FORMAT()` for formatting and extracting parts of dates.

Feel free to attempt solving this, and I can help guide you through any parts you're having trouble with!




Here is an advanced SQL interview exercise, along with solutions for each query, covering multiple SQL concepts such as aggregation, window functions, joins, subqueries, and date handling.

### Problem: Sales Analysis and Reporting

**Background:**

We have the following tables:

1. **`Customers`**
   - `customer_id` (INT)
   - `customer_name` (VARCHAR)
   - `join_date` (DATE)
   - `country` (VARCHAR)

2. **`Products`**
   - `product_id` (INT)
   - `product_name` (VARCHAR)
   - `category` (VARCHAR)
   - `price` (DECIMAL)

3. **`Orders`**
   - `order_id` (INT)
   - `customer_id` (INT)
   - `order_date` (DATE)
   - `total_amount` (DECIMAL)

4. **`Order_Items`**
   - `order_item_id` (INT)
   - `order_id` (INT)
   - `product_id` (INT)
   - `quantity` (INT)
   - `price_at_purchase` (DECIMAL)

---

### Exercise Solutions:

#### 1. **Customer's Lifetime Value (LTV)**:
The lifetime value (LTV) is the total amount spent by a customer across all their orders.

**Solution:**
```sql
SELECT
    c.customer_id,
    c.customer_name,
    SUM(oi.quantity * oi.price_at_purchase) AS lifetime_value
FROM
    Customers c
JOIN
    Orders o ON c.customer_id = o.customer_id
JOIN
    Order_Items oi ON o.order_id = oi.order_id
GROUP BY
    c.customer_id, c.customer_name
ORDER BY
    lifetime_value DESC;
```

- **Explanation**: This query joins the `Customers`, `Orders`, and `Order_Items` tables to calculate the total amount spent by each customer. The `SUM()` function is used to aggregate the total spent on all their orders.

---

#### 2. **Top 5 Customers by LTV**:
Find the top 5 customers by lifetime value, including their `customer_name`, `LTV`, and `country`.

**Solution:**
```sql
SELECT
    c.customer_name,
    SUM(oi.quantity * oi.price_at_purchase) AS lifetime_value,
    c.country
FROM
    Customers c
JOIN
    Orders o ON c.customer_id = o.customer_id
JOIN
    Order_Items oi ON o.order_id = oi.order_id
GROUP BY
    c.customer_name, c.country
ORDER BY
    lifetime_value DESC
LIMIT 5;
```

- **Explanation**: This query is similar to the first one but limits the result to the top 5 customers based on their lifetime value. The `LIMIT` clause ensures that only the top 5 customers are returned.

---

#### 3. **Monthly Revenue Trend**:
Calculate the total revenue for each month, ordered by the month and year.

**Solution:**
```sql
SELECT
    DATE_FORMAT(o.order_date, '%Y-%m') AS month_year,
    SUM(oi.quantity * oi.price_at_purchase) AS total_revenue
FROM
    Orders o
JOIN
    Order_Items oi ON o.order_id = oi.order_id
GROUP BY
    month_year
ORDER BY
    month_year;
```

- **Explanation**: The `DATE_FORMAT()` function is used to extract the `YYYY-MM` format from the `order_date` field. Then we group by `month_year` and calculate the total revenue for each month using `SUM()`.

---

#### 4. **Products with the Highest Revenue**:
Identify the top 5 products by revenue, returning their `product_name`, `category`, and total revenue.

**Solution:**
```sql
SELECT
    p.product_name,
    p.category,
    SUM(oi.quantity * oi.price_at_purchase) AS total_revenue
FROM
    Products p
JOIN
    Order_Items oi ON p.product_id = oi.product_id
GROUP BY
    p.product_name, p.category
ORDER BY
    total_revenue DESC
LIMIT 5;
```

- **Explanation**: We join `Products` with `Order_Items` to calculate the total revenue for each product. The `SUM()` function is used to aggregate revenue, and the `LIMIT` clause returns the top 5 products by revenue.

---

#### 5. **Yearly Growth in Revenue**:
Calculate the year-over-year revenue growth percentage.

**Solution:**
```sql
WITH YearlyRevenue AS (
    SELECT
        YEAR(o.order_date) AS year,
        SUM(oi.quantity * oi.price_at_purchase) AS total_revenue
    FROM
        Orders o
    JOIN
        Order_Items oi ON o.order_id = oi.order_id
    GROUP BY
        year
)
SELECT
    year,
    total_revenue,
    LAG(total_revenue) OVER (ORDER BY year) AS previous_year_revenue,
    CASE
        WHEN LAG(total_revenue) OVER (ORDER BY year) IS NULL THEN NULL
        ELSE ((total_revenue - LAG(total_revenue) OVER (ORDER BY year)) / LAG(total_revenue) OVER (ORDER BY year)) * 100
    END AS revenue_growth_percentage
FROM
    YearlyRevenue
ORDER BY
    year;
```

- **Explanation**: This query uses a common table expression (CTE) to first calculate the total revenue for each year. Then, the `LAG()` window function is used to get the revenue of the previous year for each row. The percentage growth is calculated using the formula:  
  \[
  \text{{growth\_percentage}} = \frac{{\text{{current\_revenue}} - \text{{previous\_revenue}}}}{{\text{{previous\_revenue}}}} \times 100
  \]
  If the previous year's revenue is `NULL` (i.e., for the first year), the percentage growth is also `NULL`.

---

#### 6. **Average Order Value (AOV) by Country**:
Calculate the average order value by country.

**Solution:**
```sql
SELECT
    c.country,
    AVG(o.total_amount) AS average_order_value
FROM
    Customers c
JOIN
    Orders o ON c.customer_id = o.customer_id
GROUP BY
    c.country
ORDER BY
    average_order_value DESC;
```

- **Explanation**: We calculate the average order value (`AVG(total_amount)`) for each country by joining the `Customers` and `Orders` tables. The `GROUP BY` clause ensures that the calculation is done per country.

---

#### 7. **Top 3 Selling Products by Region**:
Find the top 3 selling products by country, showing `country`, `product_name`, `category`, and the total quantity sold for each product.

**Solution:**
```sql
WITH ProductSales AS (
    SELECT
        c.country,
        p.product_name,
        p.category,
        SUM(oi.quantity) AS total_quantity_sold
    FROM
        Customers c
    JOIN
        Orders o ON c.customer_id = o.customer_id
    JOIN
        Order_Items oi ON o.order_id = oi.order_id
    JOIN
        Products p ON oi.product_id = p.product_id
    GROUP BY
        c.country, p.product_name, p.category
)
SELECT
    country,
    product_name,
    category,
    total_quantity_sold
FROM (
    SELECT
        country,
        product_name,
        category,
        total_quantity_sold,
        ROW_NUMBER() OVER (PARTITION BY country ORDER BY total_quantity_sold DESC) AS row_num
    FROM
        ProductSales
) AS ranked
WHERE
    row_num <= 3
ORDER BY
    country, total_quantity_sold DESC;
```

- **Explanation**: We use a CTE (`ProductSales`) to calculate the total quantity sold per product in each country. The `ROW_NUMBER()` window function assigns a rank to each product within each country, ordered by the total quantity sold. The outer query filters to return only the top 3 products for each country.

---

### Summary:
These solutions cover a variety of advanced SQL techniques such as:
- **Aggregation** (`SUM()`, `AVG()`)
- **Joins** (`JOIN`, `INNER JOIN`)
- **Window functions** (`ROW_NUMBER()`, `LAG()`)
- **Subqueries and CTEs** for complex aggregations
- **Date functions** for extracting year/month information
- **Ranking** with `ROW_NUMBER()` for top N queries

Let me know if you'd like further clarification on any of these queries!