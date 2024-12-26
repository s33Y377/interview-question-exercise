FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints and the Starlette framework. It offers several key features that make it a great choice for building RESTful APIs.

Here’s a detailed breakdown of FastAPI features with code examples:

---

## 1. **Fast and High Performance**

FastAPI is one of the fastest web frameworks available because it is built on top of Starlette and Pydantic, two highly optimized libraries. It uses asynchronous programming by default, meaning it can handle many requests concurrently.

```python
from fastapi import FastAPI
import time

app = FastAPI()

@app.get("/")
async def read_root():
    time.sleep(1)  # Simulate a delay
    return {"message": "Hello, World!"}
```

### Key Features:
- ASGI-based framework for async capabilities.
- Non-blocking I/O for high concurrency.

---

## 2. **Automatic Interactive API Documentation (Swagger & ReDoc)**

FastAPI automatically generates interactive API documentation using OpenAPI (Swagger UI and ReDoc) without additional configuration.

### Example:
Run the application and visit:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
```

**Visit:**
- `http://127.0.0.1:8000/docs` for Swagger UI.
- `http://127.0.0.1:8000/redoc` for ReDoc.

---

## 3. **Automatic Validation and Serialization**

FastAPI uses Pydantic for automatic data validation and serialization. When you declare a request body, query parameters, or path variables with Pydantic models or native types, FastAPI automatically handles validation and serialization.

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    description: str = None
    price: float
    tax: float = None

@app.post("/items/")
async def create_item(item: Item):
    return {"name": item.name, "price": item.price}
```

FastAPI will automatically validate the `Item` model data and return an error if the input does not meet the validation criteria.

---

## 4. **Dependency Injection System**

FastAPI supports dependency injection, allowing you to define reusable components such as authentication, database connections, and configuration management. This enables cleaner and modular code.

```python
from fastapi import FastAPI, Depends

app = FastAPI()

# Dependency
def get_query_param(query: str = None):
    return query

@app.get("/items/")
async def read_items(query: str = Depends(get_query_param)):
    return {"query": query}
```

You can also use dependencies in more advanced ways, like connecting to a database or creating a reusable authentication system.

---

## 5. **Asynchronous Support (Async/Await)**

FastAPI is fully async/await-compatible, enabling non-blocking operations. This is especially important when dealing with I/O-bound tasks (e.g., database queries, file I/O, web scraping).

```python
from fastapi import FastAPI
import asyncio

app = FastAPI()

@app.get("/long-task")
async def long_task():
    await asyncio.sleep(5)  # Simulate a long-running task
    return {"message": "Task completed!"}
```

Using `async def` makes FastAPI handle these operations efficiently without blocking other requests.

---

## 6. **Path Parameters and Query Parameters**

FastAPI automatically reads path and query parameters, and it can validate their types, which makes the development process fast and safe.

### Path Parameters:
```python
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

### Query Parameters:
```python
@app.get("/items/")
async def read_items(q: str = None):
    return {"q": q}
```

---

## 7. **Security and OAuth2 Support**

FastAPI provides tools for handling authentication and authorization, such as OAuth2 password flow, API keys, etc.

### OAuth2 Password Flow Example:
```python
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
app = FastAPI()

# Fake users database
fake_users_db = {
    "testuser": {
        "username": "testuser",
        "password": "secret"
    }
}

def get_current_user(token: str = Depends(oauth2_scheme)):
    if token == "secret":  # This is just a mock
        return {"username": "testuser"}
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/users/me")
async def read_users_me(current_user: dict = Depends(get_current_user)):
    return current_user
```

In this example, we are using OAuth2PasswordBearer to handle token-based authentication.

---

## 8. **File Uploads**

FastAPI makes it easy to upload files with `File` and `UploadFile`.

### Example:
```python
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    return {"filename": file.filename, "content_length": len(contents)}
```

You can also access metadata about the uploaded file, such as filename and content type.

---

## 9. **CORS (Cross-Origin Resource Sharing)**

FastAPI allows you to configure CORS easily to allow cross-origin requests.

### Example (enabling CORS):
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for all domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)
```

This is useful when your frontend is hosted on a different domain than your FastAPI backend.

---

## 10. **Testing (Built-in Test Client)**

FastAPI supports easy testing with `TestClient`, which is based on `httpx`. You can use it for unit testing your API.

### Example:
```python
from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}

# Testing the API
client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, World!"}
```

---

## 11. **Background Tasks**

FastAPI supports background tasks, which are tasks that run in the background after returning a response to the client.

```python
from fastapi import FastAPI, BackgroundTasks
import time

app = FastAPI()

def write_log(message: str):
    time.sleep(5)  # Simulate a long-running task
    with open("log.txt", mode="a") as log:
        log.write(message)

@app.get("/send-notification/")
async def send_notification(background_tasks: BackgroundTasks):
    background_tasks.add_task(write_log, "Notification sent!")
    return {"message": "Notification is being processed in the background"}
```

---

## 12. **Custom Exception Handling**

FastAPI provides easy ways to handle custom exceptions and create custom responses for specific errors.

```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    if item_id == 42:
        raise HTTPException(status_code=418, detail="This is a special item!")
    return {"item_id": item_id}
```

FastAPI automatically handles HTTP exceptions like `404` or `400`, but you can also create custom ones.

---

## 13. **Request and Response Models**

You can use Pydantic models for request and response bodies. This ensures that data is properly validated and serialized.

```python
from pydantic import BaseModel
from fastapi import FastAPI

class Item(BaseModel):
    name: str
    description: str = None
    price: float

app = FastAPI()

@app.post("/items/")
async def create_item(item: Item):
    return {"item": item}
```

The `Item` model will be automatically validated for incoming requests.

---

## Conclusion


To further delve into FastAPI, let's cover more advanced concepts such as middleware, ORM integration, and models.

### 1. **Middleware**

Middleware in FastAPI is a function that processes requests before they reach the route handler and responses before they are sent to the client. Middleware can be used for tasks like logging, error handling, or adding custom headers.

#### Example: Custom Middleware

```python
from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import time

class CustomMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers['X-Process-Time'] = str(process_time)
        return response

app = FastAPI()

app.add_middleware(CustomMiddleware)

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}
```

Here, the middleware calculates how long the request took and adds the result in the response headers.

---

### 2. **ORM (Object Relational Mapping)**

FastAPI can be used with ORM libraries like SQLAlchemy or Tortoise ORM for handling database interactions. SQLAlchemy is a popular choice in the Python ecosystem, and FastAPI provides seamless integration with it.

#### Example: SQLAlchemy Integration with FastAPI

1. **Install dependencies:**
   ```
   pip install sqlalchemy databases asyncpg
   ```

2. **Create models and database connection:**

```python
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

SQLALCHEMY_DATABASE_URL = "postgresql+asyncpg://user:password@localhost/testdb"

engine = create_engine(SQLALCHEMY_DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Item(Base):
    __tablename__ = 'items'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String, index=True)
    price = Column(Integer)

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI()

@app.post("/items/")
async def create_item(name: str, description: str, price: int, db: Session = Depends(get_db)):
    db_item = Item(name=name, description=description, price=price)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item
```

This example shows how to integrate SQLAlchemy into FastAPI, create a model (`Item`), and perform basic CRUD operations.

---

### 3. **Pydantic Models for Request/Response Validation**

Pydantic models are used in FastAPI to validate incoming requests and serialize outgoing responses. You can use Pydantic for both request body validation and data serialization.

#### Example: Pydantic Model for Request/Response

```python
from pydantic import BaseModel

class ItemRequest(BaseModel):
    name: str
    description: str = None
    price: float

class ItemResponse(BaseModel):
    id: int
    name: str
    description: str = None
    price: float

@app.post("/items/", response_model=ItemResponse)
async def create_item(item: ItemRequest, db: Session = Depends(get_db)):
    db_item = Item(name=item.name, description=item.description, price=item.price)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item
```

Here, `ItemRequest` is used to validate the input data for the request body, and `ItemResponse` is used to format the output data when returning a response.

---

### 4. **Model Pydantic Example (with ORM)**

FastAPI also supports using Pydantic models with ORM objects (e.g., SQLAlchemy models). You can use `orm_mode = True` to tell Pydantic to work with ORM models instead of dictionaries.

```python
from pydantic import BaseModel

class ItemBase(BaseModel):
    name: str
    description: str = None
    price: float

    class Config:
        orm_mode = True

class ItemCreate(ItemBase):
    pass

class Item(ItemBase):
    id: int

@app.post("/items/", response_model=Item)
async def create_item(item: ItemCreate, db: Session = Depends(get_db)):
    db_item = Item(name=item.name, description=item.description, price=item.price)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item
```

The `ItemBase` model is shared for both input and output, and using `orm_mode` ensures that SQLAlchemy objects are properly converted to Pydantic models.

---

### 5. **Database Sessions and Transactions**

FastAPI's dependency injection system can be used to handle database sessions effectively, ensuring proper management of database transactions.

#### Example: Using Dependency for Session Management

```python
from sqlalchemy.orm import Session
from fastapi import Depends

# Same database setup as previous examples...

@app.post("/items/")
async def create_item(item: ItemRequest, db: Session = Depends(get_db)):
    db_item = Item(name=item.name, description=item.description, price=item.price)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

@app.get("/items/{item_id}", response_model=ItemResponse)
async def get_item(item_id: int, db: Session = Depends(get_db)):
    db_item = db.query(Item).filter(Item.id == item_id).first()
    if db_item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return db_item
```

Using `Depends(get_db)` in the route handlers ensures that the database session is properly handled (opened and closed automatically).

---

### 6. **CRUD Operations with FastAPI and SQLAlchemy**

When building an API, you'll often create standard CRUD operations (Create, Read, Update, Delete). Below is an example of how you might structure these operations using SQLAlchemy and FastAPI.

```python
from sqlalchemy.orm import Session
from fastapi import FastAPI, HTTPException, Depends
from typing import List

# Example of CRUD operations

def get_item(db: Session, item_id: int):
    return db.query(Item).filter(Item.id == item_id).first()

def get_items(db: Session, skip: int = 0, limit: int = 10):
    return db.query(Item).offset(skip).limit(limit).all()

@app.get("/items/", response_model=List[ItemResponse])
async def read_items(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    items = get_items(db=db, skip=skip, limit=limit)
    return items

@app.get("/items/{item_id}", response_model=ItemResponse)
async def read_item(item_id: int, db: Session = Depends(get_db)):
    db_item = get_item(db=db, item_id=item_id)
    if db_item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return db_item
```

In this case, `get_item` and `get_items` represent the CRUD operations that interact with the database, and the FastAPI route handlers simply call these functions.

---

### 7. **Database Migration (Alembic)**

To manage database migrations (schema changes) with SQLAlchemy, Alembic is typically used. This allows you to evolve your database schema without losing data.

1. **Install Alembic:**
   ```bash
   pip install alembic
   ```

2. **Initialize Alembic:**
   ```bash
   alembic init alembic
   ```

3. **Create migration files:**
   After configuring Alembic with your database URL, you can create migrations by running:
   ```bash
   alembic revision --autogenerate -m "Create items table"
   alembic upgrade head
   ```

Alembic helps to automatically generate migration scripts and apply them to your database, which is essential in production.

---

### Conclusion

FastAPI provides powerful features like middleware, database integration with ORMs (e.g., SQLAlchemy), and easy-to-use data models with Pydantic. These concepts allow you to scale your application, manage requests efficiently, and integrate with databases in a clean and modular way. With built-in validation, database management, and support for asynchronous programming, FastAPI helps you build highly efficient and maintainable APIs.

---


FastAPI offers a modern, fast, and highly flexible framework for building APIs with Python. By leveraging Python's type hints, async programming, and powerful features like automatic validation, dependency injection, and built-in security mechanisms, FastAPI makes it easier to build high-performance, reliable APIs.

---

### why use pydantic

Pydantic is an essential part of FastAPI because it provides robust data validation, serialization, and parsing for the request and response data in a FastAPI application. Here's why Pydantic is used in FastAPI:

1. **Data Validation and Parsing**: Pydantic models are used to define the structure of request bodies, query parameters, headers, cookies, etc. It automatically validates and parses the data as it comes in, ensuring that the request is well-formed before passing it into the business logic of your application. For instance, if a client sends data that doesn't match the expected types or constraints, Pydantic will raise a validation error.

2. **Automatic Documentation**: FastAPI uses Pydantic models to generate interactive API documentation (using Swagger UI or ReDoc). The Pydantic models define the expected structure of the data, and FastAPI automatically includes this in the generated documentation, making it easier for developers to understand how to interact with the API.

3. **Data Serialization**: When returning data from an endpoint, Pydantic models can be used to serialize Python objects into JSON responses. Pydantic ensures that the data adheres to the expected format and handles any necessary conversions (e.g., converting datetime objects into ISO8601 strings).

4. **Type Safety**: Pydantic provides strict type checking and ensures that data is parsed according to the specified types in the model. This allows FastAPI to detect errors at an early stage and prevents issues caused by unexpected or incompatible data.

5. **Ease of Use**: Pydantic models are Python classes that use standard type annotations, making them simple to define and use. They provide a straightforward way to describe complex data structures (e.g., nested objects, lists, enums, etc.) without needing to manually write a lot of validation code.

6. **Custom Validation**: Pydantic allows you to define custom validation logic through validators, which means you can add additional rules beyond the basic type and value constraints. This makes it easy to implement domain-specific validation rules for your data.

7. **Performance**: Pydantic is designed to be fast and efficient. It uses Python's `dataclasses` and `typing` for better performance compared to other validation libraries. This is crucial for building APIs that need to handle large volumes of requests.

In summary, Pydantic provides powerful features for FastAPI, such as automatic validation, serialization, and type checking. This integration reduces boilerplate code, improves the reliability of your API, and enhances the developer experience with FastAPI.

---

### Code explain

```python
@app.post("/items/", response_model=schemas.Item)
def create_item(item: schemas.ItemCreate, db: Session = Depends(get_db)):
    db_item = db.query(models.Item).filter(models.Item.name == item.name).first()
    if db_item:
        raise HTTPException(status_code=400, detail="Item already exists")
    db_item = models.Item(**item.dict())
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item
```

The code snippet you provided defines a FastAPI route that handles creating an item in a database. Here's a breakdown of the functionality:

### Code Breakdown:

1. **Route Decorator (`@app.post("/items/", response_model=schemas.Item)`)**:
   - This specifies that the route accepts HTTP POST requests at the `/items/` endpoint.
   - The `response_model=schemas.Item` argument indicates that the response should be serialized to match the `Item` schema from your `schemas` module.

2. **Function Definition (`def create_item(item: schemas.ItemCreate, db: Session = Depends(get_db))`)**:
   - `item: schemas.ItemCreate`: The incoming request body is expected to be in the form of a schema `ItemCreate` defined in the `schemas` module. This is the input for the item to be created.
   - `db: Session = Depends(get_db)`: `db` is a database session. The `Depends(get_db)` is a dependency injection that retrieves the database session (`db`), likely from a `get_db` function.

3. **Checking if the Item Exists (`db_item = db.query(models.Item).filter(models.Item.name == item.name).first()`)**:
   - This queries the `models.Item` table to check if an item with the same `name` already exists in the database.
   - `db_item` will hold the first result, or `None` if no item is found.

4. **Raise HTTPException if Item Exists**:
   - If an item with the same name is found (`db_item` is not `None`), an `HTTPException` is raised with a `400` status code and the detail `"Item already exists"`.
   - This ensures that no duplicate items are added.

5. **Create the New Item**:
   - `db_item = models.Item(**item.dict())`: A new `Item` model is created using the data from the `item` schema. `item.dict()` converts the Pydantic schema to a dictionary.
   - This dictionary is passed as keyword arguments to create an instance of the `Item` model.

6. **Add the Item to the Database**:
   - `db.add(db_item)`: The newly created item is added to the session (prepared for committing to the database).
   - `db.commit()`: This commits the transaction, saving the item in the database.
   - `db.refresh(db_item)`: This refreshes the `db_item` object, retrieving the updated state of the item (including its ID or other database-generated fields like timestamps).

7. **Return the Created Item**:
   - The function returns the `db_item`, which will be serialized to match the `schemas.Item` schema due to the `response_model` in the route decorator.

### Improvements:
- **Error handling**: In case of a database error, it might be beneficial to wrap the database operations in a try/except block to catch and handle any database-related exceptions.
- **Data Validation**: You could add additional validation for the `ItemCreate` schema, ensuring fields like `name` are properly validated for uniqueness, length, or other constraints before the database query.
  
### Example `schemas.py` and `models.py`:

- `schemas.py` might define:
  ```python
  from pydantic import BaseModel
  
  class ItemCreate(BaseModel):
      name: str
      description: str
      price: float
  
  class Item(ItemCreate):
      id: int
  
      class Config:
          orm_mode = True
  ```

- `models.py` might define:
  ```python
  from sqlalchemy import Column, Integer, String, Float
  from sqlalchemy.ext.declarative import declarative_base

  Base = declarative_base()

  class Item(Base):
      __tablename__ = 'items'

      id = Column(Integer, primary_key=True, index=True)
      name = Column(String, unique=True, index=True)
      description = Column(String)
      price = Column(Float)
  ```

---
---

### Explanation

```python
from fastapi import FastAPI

app = FastAPI()


@app.get("/items/")
async def read_items(q: str | None = None):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})
    return results
```

The provided code is an example of how to define a simple FastAPI application with an HTTP GET endpoint. Let's go through each part of the code in detail:

### Code Breakdown:

#### 1. **Importing FastAPI (`from fastapi import FastAPI`)**:
   - This line imports the `FastAPI` class from the `fastapi` module. `FastAPI` is the main class that allows you to create a web application with automatic validation, documentation, and other features.

#### 2. **Creating the FastAPI Application Instance (`app = FastAPI()`)**:
   - `app = FastAPI()` creates an instance of the FastAPI class, which represents your application. This is the main entry point to configure and run the app.

#### 3. **Defining the Route (`@app.get("/items/")`)**:
   - The `@app.get("/items/")` decorator defines a GET request handler for the `/items/` endpoint. This means that when a client sends a GET request to `/items/`, this function will be called.
   - The `@app.get` decorator specifies that the request method is `GET`.

#### 4. **Async Function for Handling Requests (`async def read_items(q: str | None = None)`)**:
   - The function `read_items` is defined to handle the `/items/` endpoint. 
   - It's marked as `async`, which means it's an asynchronous function and FastAPI will handle the request asynchronously, which is useful for performance (especially for I/O-bound operations like database queries).
   - The function accepts one optional query parameter, `q`, which is of type `str | None`. This means `q` can either be a string (if provided) or `None` (if not provided). The `str | None` syntax is a feature from Python 3.10+, which represents the possibility of either a `str` or `None` (similar to `Optional[str]` in earlier versions of Python).
   - If the `q` parameter is not provided in the query string, its value will default to `None`.

#### 5. **Processing the Request (`results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}`)**:
   - The `results` dictionary contains a key `"items"` that holds a list of dictionaries, each representing an item. In this case, the items have `"item_id": "Foo"` and `"item_id": "Bar"`.
   - This static list is used as a placeholder. In a real-world application, this could be replaced with dynamic data retrieved from a database.

#### 6. **Handling the Query Parameter (`if q: results.update({"q": q})`)**:
   - This `if` block checks whether the `q` query parameter was provided by the user (i.e., `q` is not `None`).
   - If `q` is provided, it updates the `results` dictionary by adding a new key `"q"` with the value of the query parameter `q`. For example, if the user sends a request like `/items/?q=search_term`, the response will include the search term under the `"q"` key.

#### 7. **Returning the Response (`return results`)**:
   - The function returns the `results` dictionary. FastAPI automatically serializes the dictionary to JSON format and sends it as the HTTP response.
   - The returned response will look something like this in JSON:
     ```json
     {
       "items": [
         {"item_id": "Foo"},
         {"item_id": "Bar"}
       ],
       "q": "search_term"
     }
     ```

   - If the `q` parameter was not provided in the request, the response would look like this:
     ```json
     {
       "items": [
         {"item_id": "Foo"},
         {"item_id": "Bar"}
       ]
     }
     ```

### Example Requests and Responses:

1. **Request without the `q` parameter**:
   - URL: `/items/`
   - Response:
     ```json
     {
       "items": [
         {"item_id": "Foo"},
         {"item_id": "Bar"}
       ]
     }
     ```

2. **Request with the `q` parameter**:
   - URL: `/items/?q=search_term`
   - Response:
     ```json
     {
       "items": [
         {"item_id": "Foo"},
         {"item_id": "Bar"}
       ],
       "q": "search_term"
     }
     ```

### Key Concepts in the Code:

1. **Asynchronous Route Handler (`async def`)**:
   - By marking the function as `async`, you allow FastAPI to handle requests asynchronously. This is especially useful when performing I/O-bound tasks like fetching data from a database or making external HTTP requests, as it allows other tasks to proceed while waiting for the I/O operations to complete.

2. **Query Parameters**:
   - The function accepts an optional query parameter `q`. FastAPI automatically handles the parsing of query parameters and passes them as arguments to the function. The use of `q: str | None` ensures that `q` can either be a string or `None` (if not provided).

3. **Dynamic Response**:
   - The `results` dictionary is dynamically updated if the `q` parameter is provided. This allows the response to change based on the user's query input.

### Summary:
This FastAPI route defines an endpoint `/items/` that returns a list of items. If the client provides a query parameter `q`, it adds that parameter to the response. The route is asynchronous, meaning FastAPI can handle requests concurrently for better performance.
