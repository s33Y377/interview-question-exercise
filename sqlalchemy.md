SQLAlchemy ORM (Object-Relational Mapping) is a powerful tool for interacting with relational databases using Python objects. Here is a comprehensive guide to common query syntaxes used in SQLAlchemy ORM for querying the database.

### 1. **Basic Setup and Session Creation**

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Database connection URL (for example, SQLite)
engine = create_engine('sqlite:///example.db')

# Create a session
Session = sessionmaker(bind=engine)
session = Session()
```

### 2. **Basic Queries**

#### a. **Query all records from a table**

```python
# Assuming `User` is a mapped class
users = session.query(User).all()
```

#### b. **Query one record by primary key**

```python
user = session.query(User).get(1)  # Get user by primary key (ID)
```

#### c. **Query first record (or None)**

```python
user = session.query(User).first()
```

#### d. **Query with filtering (e.g., where clause)**

```python
# Get users older than 30
users = session.query(User).filter(User.age > 30).all()
```

#### e. **Query with multiple conditions**

```python
# Get users with age > 30 and city 'New York'
users = session.query(User).filter(User.age > 30, User.city == 'New York').all()
```

#### f. **Query using `filter_by()` for simple equality conditions**

```python
users = session.query(User).filter_by(city='New York').all()
```

### 3. **Ordering and Limiting Results**

#### a. **Order by a column**

```python
users = session.query(User).order_by(User.name.asc()).all()  # Ascending
# or
users = session.query(User).order_by(User.name.desc()).all()  # Descending
```

#### b. **Limit the number of results**

```python
users = session.query(User).limit(10).all()  # Limit to first 10 records
```

#### c. **Offset and Limit**

```python
users = session.query(User).offset(10).limit(10).all()  # Skip first 10 records, then limit to 10 more
```

### 4. **Selecting Specific Columns**

```python
# Select only specific columns (like name and age)
users = session.query(User.name, User.age).all()
```

### 5. **Aggregates and Grouping**

#### a. **Count the number of records**

```python
from sqlalchemy import func

# Count users
user_count = session.query(func.count(User.id)).scalar()
```

#### b. **Group by**

```python
# Group by city and count the number of users in each city
users_by_city = session.query(User.city, func.count(User.id)).group_by(User.city).all()
```

#### c. **Sum, Average, Min, Max**

```python
# Get the average age of users
average_age = session.query(func.avg(User.age)).scalar()

# Get the minimum age of users
min_age = session.query(func.min(User.age)).scalar()

# Get the maximum age of users
max_age = session.query(func.max(User.age)).scalar()
```

### 6. **Joins**

#### a. **Inner Join**

```python
# Assuming User has a relationship with `Address`
users_with_addresses = session.query(User).join(Address).all()
```

#### b. **Left Join**

```python
# Left join to get all users even if they don't have an address
users_with_addresses = session.query(User).outerjoin(Address).all()
```

#### c. **Join with multiple tables**

```python
# Join User -> Address -> City (assuming relationships are set up correctly)
users_in_city = session.query(User).join(Address).join(City).filter(City.name == 'New York').all()
```

### 7. **Using Subqueries**

```python
# Example of subquery to filter users with a specific condition on addresses
subquery = session.query(Address.user_id).filter(Address.city == 'New York').subquery()
users = session.query(User).filter(User.id.in_(subquery)).all()
```

### 8. **Distinct and Pagination**

#### a. **Distinct**

```python
# Get distinct cities from the User table
distinct_cities = session.query(User.city).distinct().all()
```

#### b. **Pagination (limit + offset)**

```python
# Fetch page 2 of results, 10 users per page
page_number = 2
page_size = 10
users = session.query(User).offset((page_number - 1) * page_size).limit(page_size).all()
```

### 9. **Update Records**

```python
# Update a single record
user = session.query(User).filter(User.id == 1).first()
user.name = 'New Name'
session.commit()
```

#### a. **Bulk Update**

```python
# Update all users in a city
session.query(User).filter(User.city == 'New York').update({"city": "Los Angeles"})
session.commit()
```

### 10. **Delete Records**

#### a. **Delete a single record**

```python
user = session.query(User).filter(User.id == 1).first()
session.delete(user)
session.commit()
```

#### b. **Bulk Delete**

```python
# Delete all users in a specific city
session.query(User).filter(User.city == 'New York').delete()
session.commit()
```

### 11. **Transaction Handling**

```python
from sqlalchemy.exc import SQLAlchemyError

try:
    # Start a transaction
    session.begin()
    user = session.query(User).filter(User.id == 1).first()
    user.name = "Updated Name"
    session.commit()
except SQLAlchemyError:
    session.rollback()
    raise
```

### 12. **Using `with_entities`**

```python
# Using with_entities to select specific columns
users = session.query(User).with_entities(User.name, User.email).all()
```

### 13. **Using Aliases**

```python
from sqlalchemy.orm import aliased

# Create an alias for a table
UserAlias = aliased(User)

# Query using the alias
users = session.query(User, UserAlias).join(UserAlias, User.id == UserAlias.id).all()
```

### 14. **Raw SQL Execution**

If you need to execute raw SQL queries:

```python
# Execute raw SQL
result = session.execute("SELECT * FROM user WHERE city = :city", {"city": "New York"})
```

This guide covers the fundamental and most commonly used query syntaxes in SQLAlchemy ORM. You can extend these queries to match more complex use cases and interactions with your database.
