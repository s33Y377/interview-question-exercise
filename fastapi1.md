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

FastAPI offers a modern, fast, and highly flexible framework for building APIs with Python. By leveraging Python's type hints, async programming, and powerful features like automatic validation, dependency injection, and built-in security mechanisms, FastAPI makes it easier to build high-performance, reliable APIs.

