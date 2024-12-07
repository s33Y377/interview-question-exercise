## Here are some advanced Python interview questions that could be asked to assess deep knowledge of the language

[Source1](https://www.webasha.com/blog/advanced-python-interview-questions)

### 1. What are metaclasses in Python, and how do they work?

   **Answer** : A metaclass in Python is a class that defines how other classes are created. It allows you to modify class creation behavior, such as adding or modifying attributes, methods, or class-level properties. Metaclasses can be defined by inheriting from type, and you can override methods like **new** and **init** to control class instantiation.

---

### 2. Explain the Global Interpreter Lock (GIL) and its impact on multi-threading in Python

   **Answer** : The GIL is a mechanism that prevents multiple native threads from executing Python bytecodes at once in CPython (the default implementation). This means that, in CPython, even though you can have multiple threads, only one thread can execute Python bytecode at a time. This limits concurrency in CPU-bound programs but does not affect I/O-bound operations significantly.

---

### 3. What is the difference between deepcopy and copy in Python?

**Answer**: The copy() function creates a shallow copy of an object, meaning that if the original object contains other objects (like lists inside a list), the references to those inner objects are copied, not the actual inner objects. The deepcopy() function, on the other hand, creates a completely new copy of the object along with all objects nested within it.

---

### 4. What are Python decorators and how do they work?

**Answerer**: Decorators are a way to modify or enhance functions or methods without changing their source code. A decorator is a function that takes another function as an argument and returns a new function. Decorators are commonly used for logging, access control, memoization, etc.

   Example:

```python
def decorator_function(original_function):
    def wrapper_function():
        print("Wrapper executed this before {}".format(original_function.__name__))
        return original_function()
    return wrapper_function
```

### 5. **Explain Python's garbage collection mechanism**

 Answerer**: Python uses reference counting and garbage collection (GC) to manage memory. When an object's reference count drops to zero, it is automatically deallocated. Additionally, Python uses a cyclic garbage collector to handle circular references. The garbage collection process is handled by the gc module.

---

### 6. **What are the differences between @staticmethod and @classmethod in Python?

**Answer**\
    - **@staticmethod**: It is used to define a method that doesn't depend on the instance or the class. It doesn’t take self or cls as the first parameter.\
    - **@classmethod** : It is used to define a method that receives the class itself as the first argument (represented as cls), and it can access class-level attributes or methods.

---

### 7. What are Python generators and how do they work?

**Answer**: A generator is a function that returns an iterator, and it yields values one at a time using the yield keyword. Unlike normal functions that return a value and exit, generators maintain their state between calls and can be resumed. They are memory-efficient as they yield items lazily

Example:

```python
def count_up_to(max):
    count = 1
    while count <= max:
        yield count
        count += 1
```

---

### 8. What is the slots of **slots** in Python

**Answer** - **slots** is used to limit the attributes of a class to a predefined set, which can save memory by preventing the creation of dictult **dict** for each instance. This is especially useful when dealing with a large number of instances and known attributes.

   Example:

```python
class MyClass:
    __slots__ = ['name', 'age']
    def __init__(self, name, age):
        self.name = name
        self.age = age
```

```python
class MyClass:
    __slots__ = ['name', 'age']
    def __init__(self, name, age, a):
        self.name = name
        self.age = age
        self.a = a

m = MyClass("abc", 25, 1)

O/P =>

Traceback (most recent call last):
  File "/home/main.py", line 8, in <module>
    m = MyClass("abc", 25, 1)
  File "/home/main.py", line 6, in __init__
    self.a = a
AttributeError: 'MyClass' object has no attribute 'a'
```

---

### 9. How does Python handle multiple inheritance and method resolution order (MRO)?

- Answer: Python uses the C3 linearization algorithm (also known as C3 superclass linearization) to determine the method resolution order (MRO) when dealing with multiple inheritance. The MRO determines the order in which base classes are searched when a method is called. You can view the MRO of a class using the mro() method or **mro** attribute.

[Link](https://docs.python.org/3/howto/mro.html)

   Example:

```python
class A: pass
class B(A): pass
class C(A): pass
class D(B, C): pass
print(D.mro())
```

```python
[<class '__main__.D'>, <class '__main__.B'>, <class '__main__.C'>, <class '__main__.A'>, <class 'object'>]
```

---

### 10. What is a context manager in Python?

- Answer: A context manager is a Python object that allows you to manage resources, such as files or database connections, using the with statement. It defines methods **enter**() and **exit**() to allocate and release resources.

   Example:

```python
class MyContextManager:
    def __enter__(self):
        print("Entering the context")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("Exiting the context")

with MyContextManager():
    print("Inside the context")
```

---

### 11. Explain the concept of 'duck typing' in Python

- Answer: Duck typing is a concept where the type or class of an object is determined by its behavior (methods and properties) rather than its inheritance or explicit type. If an object behaves like a certain type, it can be treated as that type, regardless of its actual class.

   Example:

```python
class Bird:
    def fly(self):
        print("Flying")

class Airplane:
    def fly(self):
        print("Flying like an airplane")

def take_off(flyable):
    flyable.fly()  # Works if the object has a fly method
```

```python
take_off(Bird())
```

---

### 12. **What is the difference between is and == in Python?**

**Answerer**: The == operator checks if the values of two objects are the same, while the is operator checks if two references point to the same object in memory.

   Example:

```python
a = [1, 2, 3]
b = [1, 2, 3]
print(a == b)  # True (values are equal)
print(a is b)  # False (they are different objects)
```

---

### 13. **How does Python's asyncio module work, and what is the difference between async and await?AnswerAnswer**: The asyncio module is used for writing asynchronous code. It allows you to run I/O-bound operations without blocking the execution of other tasks. The async keyword defines an asynchronous function, while await is used to pause the function's execution until the awaited result is available

   Example:

```python
import asyncio

async def fetch_data():
    await asyncio.sleep(2)  # Simulates an I/O operation
    return "Data fetched"

async def main():
    data = await fetch_data()
    print(data)

asyncio.run(main())
```

---

### 14. **What are f-strings in Python, and why are they preferred over other string formatting methAnswer-

**Answer**: f-strings (formatted string literals) are a concise and efficient way to embed expressions inside string literals, introduced in Python 3.6. They are preferred over older methods (like % formatting or str.format()) because they are more readable and generally faster.

   Example:

```python
name = "John"
age = 30
print(f"My name is {name} and I am {age} years old.")
```

---

### 16. **What is the difference between **del** and **exit** in Python?

**Answerer** : del **del** and **exit** are used for cleanup purposes, but they are used in different contedel

- **del** is a destructor method called when an object is about to be destroyed. It is part of Python’s garbage collection mechanism and is not guaranteed to be called immediately after an object is no longer referen exit

- **exit** is used in the context of a context manager and is part of the with statement. It ensures that the resources are cleaned up when exiting the context, even if an exception occurs.

---

### 17. **What is the purpose of the abc module in Python?

**Answer**: The abc (Abstract Base Classes) module in Python provides a mechanism for defining abstract classes. An abstract class is one that cannot be instantiated directly and is intended to be subclassed. The module allows you to define abstract methods, which must be implemented by any subclass.

   Example:

```python
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def make_sound(self):
        pass

class Dog(Animal):
    def make_sound(self):
        print("Woof")

# animal = Animal()  # Will raise an error, as Animal is abstract
dog = Dog()
dog.make_sound()  # "Woof"
```

---

### 18. **Explain** Python's yield from expression

**Answer**: The yield from expression simplifies delegating part of a generator’s operations to another iterable or generator. It allows a generator to yield all values from another iterable or generator without explicitly looping through it.

   Example:

```python
def generator1():
    yield 1
    yield 2

def generator2():
    yield 3
    yield 4

def combined():
    yield from generator1()
    yield from generator2()

for value in combined():
    print(value)  # 1, 2, 3, 4
```

---

### 19. What is the difference between new and init

**Answer**:
    - **new** is the method responsible for creating a new instance of a class. initialised before **init** and is typically used in metaclasses or when subclassing immutable types like int, str, or tuple.
    - **init** is called after the object new  (i.e., after **new**) and is used to initialize the object's attributes.

   Example:

```python
class MyClass:
    def __new__(cls):
        print("Creating instance")
        return super().__new__(cls)

    def __init__(self):
        print("Initializing instance")

obj = MyClass()

Output:
Creating instance
Initializing instance
```

```python
class LowerCaseTuple(tuple):
    def __new__(cls, iterable):
       lower_iterable = (l.lower() for l in iterable)
       return super().__new__(cls, lower_iterable)

print(LowerCaseTuple(['HELLO', 'MEDIUM']))

```

[Dunder method](https://docs.python.org/3/reference/datamodel.html#basic-customization)

### 20. **What** is a "closure"?

- **Answer**: A closure is a function that "remembers" the environment in which it was created, even after that environment has finished execution. This means that the function has access to variables that were in scope when the function was defined, even if they are no longer in scope when the function is called.

   Example:

```python
def outer(x):
    def inner(y):
        return x + y
    return inner

closure = outer(10)
print(closure(5))  # Output: 15
```

---

### 21. What is the @property decorator, and how is it used?

**Answerer**: The @property decorator is used to define a method as a read-only attribute. It allows you to define a method that can be accessed like an attribute, without explicitly calling the method.

Example:

```python
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        return self._radius

    @property
    def area(self):
        return 3.14159 * self._radius ** 2

c = Circle(5)
print(c.radius)  # 5
print(c.area)    # 78.53975
```

---

### 22. How can you improve the performance of Python code involving large datasets?

**Answerer**: Some strategies to optimize performance include:

- NumPy **Pandas** for large numerical and data-processing tasks.
- built-in functions**and avoid unnecessary loops.
- multiprocessingng **concurrent.futureses** for CPU-bound tasks to take advantage of multiple processors.
- generator expressions** instead of list comprehensions for memory efficiency.
- memoizationon **caching** (via functools.lru_cache or custom caching) to avoid redundant computations.
- Profile code using the cProfile module to identify and optimize bottlenecks.

---

### 23. What are some ways to optimize memory usage in Python?

**Answer**: Techniques for optimizing memory usage include:

- Using generators instead of lists when working with large datasets.
- Using ****slots**** in classes to avoid the overhead of instance dictionaries.
- Avoiding the creation of unnecessary copies of data (e.g., use in-place operations).
- Using **array** or **numpy** arrays for numerical data instead of lists.
- **memoryview** objects to work with large binary data efficientlHow does Python handle namespaces and variable scope?ble scAnswer-

---

### 24. How can you optimize the performance of Python

**Answer**: Some ways to optimize Python code include:

- Using built-in functions and libraries, as they are often optimized.
- Avoiding global variables and using local variables where possible.
- Using list comprehensions or generator expressions instead of loops for better performance.
- Profiling the code using cProfile and focusing on optimizing the bottlenecks.
- Using libraries like NumPy for mathematical operations and multiprocessing for parallelism in CPU-bound tasks.

---

### 25. What are some ways to handle exceptions in Python?

**Answerer**: In Python, exceptions are handled using try, except, else, and finally blocks.

- try: Contains code that may raise an exception.
- except: Catches and handles the exception.
- else: Runs if no exception was raised in the try block.
- finally: Executes code after the try block, regardless of whether an exception was raised or not.

   Example:

```python
   try:
       x = 1 / 0
   except ZeroDivisionError:
       print("Cannot divide by zero.")
   else:
       print("No errors.")
   finally:
       print("This runs no matter what.")
```

### 26. **What** are Python's contextlib and contextmanager?

**Answer**: contextlib is a standard library module that provides utilities for creating and working with context managers. The contextmanager decorator is used to define a simple context manager using a generator function.

   Example:

```python
from contextlib import contextmanager

@contextmanager
def open_file(filename):
    f = open(filename, 'r')
    try:
        yield f
    finally:
        f.close()

with open_file('file.txt') as f:
    print(f.read())
```

### 27. What are Python descriptors, and how do they work?

**Answer**: A descriptor is an object attribute with "binding behavior" that customizes how an attribute is accessed or modified. Descriptors implement any of the methods **get**, **set**, or **delete** to define how attribute access is managed.

   Example:

```python
class Descriptor:
    def __get__(self, instance, owner):
        return 'Attribute accessed'

class MyClass:
    attr = Descriptor()

obj = MyClass()
print(obj.attr)  # 'Attribute accessed'
```

```python
class UppercaseDescriptor:
    def __init__(self, name=''):
        self.name = name

    def __get__(self, instance, owner):
        return instance.__dict__.get(self.name, "").upper()

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value

class MyClass:
    name = UppercaseDescriptor("name")

obj = MyClass()
obj.name = "hello"
print(obj.name)  # Output: "HELLO"
```

---

### 27. What is the purpose of the functools module, and what are some common functions it contain

**Answer**: The functools module provides higher-order functions that operate on other functions or callable objects. Some common functions include:

- lru_cache: Caches function results to improve performance for expensive functions.
- partial: Creates a new function by fixing some arguments of an existing function.
- reduce: Applies a binary function cumulatively to a sequence.

Example:

```python
from functools import partial

def power(base, exponent):
    return base ** exponent

square = partial(power, exponent=2)
print(square(4))  # Output: 16
```

**lru_cache** :

```python
from functools import lru_cache

@lru_cache(maxsize=3)
def expensive_function(x):
    print(f"Calculating {x}...")
    return x * 2

# Example usage
print(expensive_function(1))  # Cache miss
print(expensive_function(2))  # Cache miss
print(expensive_function(3))  # Cache miss
print(expensive_function(1))  # Cache hit (from the cache)
print(expensive_function(4))  # Cache miss (evicts 2)
```

**Custom cache** :

```python
def cache_decorator(func):
    cache = {}
    
    def wrapper(*args):
        if args not in cache:
            print(f"Cache miss for {args}")
            cache[args] = func(*args)
        else:
            print(f"Cache hit for {args}")
        return cache[args]
    
    wrapper.clear_cache = lambda: cache.clear()
    return wrapper

@cache_decorator
def expensive_function(x):
    return x * x

# Test
print(expensive_function(4))  # Cache miss
print(expensive_function(4))  # Cache hit
print(expensive_function(5))  # Cache miss
expensive_function.clear_cache()  # Manually clear cache
print(expensive_function(4))  # Cache miss again
```

---

### 27. Implement a Decorator that Measures the Execution Time of a Function

- Problem: Write a decorator that measures how long a function takes to execute.
- Solution:

```python
import time

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time} seconds")
        return result
    return wrapper

@measure_time
def long_running_task(*args, **kwargs):
    time.sleep(2)

long_running_task()  # Output: Execution time: 2.000xx seconds
```

### 2. **Implement a Custom Iterator Class**

**Problem**: Implement a custom iterator that returns squares of numbers from 1 to n.
 Solution**:

```python
class SquareIterator:
    def __init__(self, n):
        self.n = n
        self.current = 
        
    def __iter__(self):
        return 
        
    def __next__(self):
        if self.current > self.n:
            raise StopIteration
        result = self.current ** 2
        self.current += 1
        return result

squares = SquareIterator(5)
for square in squares:
    print(square)  # Output: 1 4 9 16 25
```

### 2. The OrderedDict is a part of Python's collections module. It is similar to a regular dictionary, but it maintains the order of keys as they are inserted. It was introduced in Python 3.1 to ensure that dictionaries retain their insertion order. Starting from Python 3.7, the built-in dict also maintains insertion order, but OrderedDict provides additional methods specific to its functionality

```python
from collections import OrderedDict

# 1. Creating OrderedDict
ordered_dict = OrderedDict([('apple', 1), ('banana', 2), ('cherry', 3)])
print("Original OrderedDict:", ordered_dict)

# 2. Adding item
ordered_dict['date'] = 4
print("After adding 'date':", ordered_dict)

# 3. Move item to the end
ordered_dict.move_to_end('apple')
print("After moving 'apple' to end:", ordered_dict)

# 4. Move item to the start
ordered_dict.move_to_end('banana', last=False)
print("After moving 'banana' to start:", ordered_dict)

# 5. Deleting an item
del ordered_dict['cherry']
print("After deleting 'cherry':", ordered_dict)

# 6. Pop item
popped_item = ordered_dict.pop('banana')
print("Popped item:", popped_item)
print("After pop:", ordered_dict)

# 7. Reversing the order
reversed_dict = OrderedDict(reversed(ordered_dict.items()))
print("Reversed OrderedDict:", reversed_dict)

# 8. Copying OrderedDict
copied_dict = ordered_dict.copy()
print("Copied OrderedDict:", copied_dict)

# 9. Equality comparison
ordered_dict2 = OrderedDict([('date', 4), ('apple', 1)])
print("Are the two OrderedDicts equal?", ordered_dict == ordered_dict2)

# 10. Using with defaultdict
from collections import defaultdict
ordered_dict = defaultdict(int)  # Default value is 0
ordered_dict['apple'] += 1
ordered_dict['banana'] += 2
ordered_dict['apple'] += 3
print("Defaultdict-like OrderedDict:", ordered_dict)
```

```python
Original OrderedDict: OrderedDict([('apple', 1), ('banana', 2), ('cherry', 3)])
After adding 'date': OrderedDict([('apple', 1), ('banana', 2), ('cherry', 3), ('date', 4)])
After moving 'apple' to end: OrderedDict([('banana', 2), ('cherry', 3), ('date', 4), ('apple', 1)])
After moving 'banana' to start: OrderedDict([('cherry', 3), ('date', 4), ('apple', 1), ('banana', 2)])
After deleting 'cherry': OrderedDict([('date', 4), ('apple', 1), ('banana', 2)])
Popped item: 2
After pop: OrderedDict([('date', 4), ('apple', 1)])
Reversed OrderedDict: OrderedDict([('apple', 1), ('date', 4)])
Copied OrderedDict: OrderedDict([('date', 4), ('apple', 1)])
Are the two OrderedDicts equal? False
Defaultdict-like OrderedDict: defaultdict(<class 'int'>, {'apple': 4, 'banana': 2})

```

---

### 3. Design and Implement a Cache with LRU (Least Recently Used) Evictionon**

 Problemem**: Implement an LRU cache using Python's collections.OrderedDict or your own implementation.
 Solutionon**:

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

# Example usage
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
print(cache.get(1))  # Output: 1
cache.put(3, 3)
print(cache.get(2))  # Output: -1 (evicted)
```

### 4. Write a Python Function to Perform Deep Copying of Nested Dictionaries

**Problem**: Write a function that performs a deep copy of a nested dictionary.

**Solution**:

```python
def deep_copy(d):
    if isinstance(d, dict):
        return {key: deep_copy(value) for key, value in d.items()}
    else:
        return d

original = {'a': 1, 'b': {'x': 2, 'y': 3}}
copy = deep_copy(original)
copy['b']['x'] = 10
print(original)  # Output: {'a': 1, 'b': {'x': 2, 'y': 3}}
print(copy)      # Output: {'a': 1, 'b': {'x': 10, 'y': 3}}
```

### 5. Implement a Function to Flatten a Nested List (of arbitrary depth)**

 Problemem**: Write a function that takes a nested list (of arbitrary depth) and flattens it into a single list.
 Solutionon**:

```python
   def flatten(lst):
       for item in lst:
           if isinstance(item, list):
               yield from flatten(item)
           else:
               yield item

   nested_list = [1, [2, [3, 4], 5], 6]
   flat_list = list(flatten(nested_list))
   print(flat_list)  # Output: [1, 2, 3, 4, 5, 6]
```

### 6. Write a Python Function to Find the Longest Substring Without Repeating Charactersrs**

 Problemem**: Given a string, write a function that finds the length of the longest substring without repeating characters.
 Solutionon**:

```python
def longest_substring(s: str) -> int:
    seen = {}
    left = 0
    max_len = 0
    for right, char in enumerate(s):
        if char in seen and seen[char] >= left:
            left = seen[char] + 1
        seen[char] = right
        max_len = max(max_len, right - left + 1)
    return max_len
    
print(longest_substring("abcabcbb"))  # Output: 3 ("abc")
```  

### 7. **Implement a Python Generator for Fibonacci Sequence**

- **Problem**: Write a generator function that generates the Fibonacci sequence up to `n` numbers.
- **Solution**:

```
python
   def fibonacci(n):
       a, b = 0, 1
       for _ in range(n):
           yield a
           a, b = b, a + b

   fib = fibonacci(10)
   for num in fib:
       print(num)  # Output: 0 1 1 2 3 5 8 13 21 34
```

### 8. **Write a Python Program to Merge Two Sorted Lists into a Single Sorted List**

- **Problem**: Given two sorted lists, merge them into a single sorted list.
- **Solution**:

```
   def merge_sorted_lists(list1, list2):
       merged = []
       i, j = 0, 0
       while i < len(list1) and j < len(list2):
           if list1[i] < list2[j]:
               merged.append(list1[i])
               i += 1
           else:
               merged.append(list2[j])
               j += 1
       merged.extend(list1[i:])
       merged.extend(list2[j:])
       return merged

   list1 = [1, 3, 5, 7]
   list2 = [2, 4, 6, 8]
   print(merge_sorted_lists(list1, list2))  # Output: [1, 2, 3, 4, 5, 6, 7, 8]
```  

### 9. **Implement a Binary Search Algorithm**

- **Problem**: Write a function that implements binary search on a sorted list.
- **Solution**:

```
   def binary_search(arr, target):
       left, right = 0, len(arr) - 1
       while left <= right:
           mid = (left + right) // 2
           if arr[mid] == target:
               return mid
           elif arr[mid] < target:
               left = mid + 1
           else:
               right = mid - 1
       return -1  # Target not found

   arr = [1, 2, 3, 4, 5, 6, 7, 8, 9]
   print(binary_search(arr, 5))  # Output: 4
```  

### 10. **Write a Python Function to Sort a List of Tuples by the Second Element**

- **Problem**: Sort a list of tuples based on the second element in each tuple.
- **Solution**:

```
   def sort_by_second_element(tuples):
       return sorted(tuples, key=lambda x: x[1])

   tuples = [(1, 3), (3, 1), (2, 2)]
   print(sort_by_second_element(tuples))  
   # Output: [(3, 1), (2, 2), (1, 3)]
```
  
### 11. **Implement a Python Function to Check if a String is a Palindrome**

- **Problem**: Write a function that checks if a given string is a palindrome (reads the same forwards and backwards).
- **Solution**:

```
   def is_palindrome(s: str) -> bool:
       s = ''.join(e for e in s if e.isalnum()).lower()  # Remove non-alphanumeric characters and make lowercase
       return s == s[::-1]

   print(is_palindrome("A man, a plan, a canal: Panama"))  # Output: True
```  

### 12. **Write a Python Function to Convert a List of Strings into a Single String (Concatenation)**

- **Problem**: Given a list of strings, write a function that concatenates all strings into a single string.
- **Solution**:

```
   def concat_strings(str_list):
       return ''.join(str_list)

   strings = ["Hello", " ", "World", "!"]
   print(concat_strings(strings))  # Output: "Hello World!"
```

### 13. **Write a Python Function to Find the Most Frequent Element in a List**

- **Problem**: Write a function that finds the most frequent element in a list.
- **Solution**:

```
   from collections import Counter

   def most_frequent(lst):
       count = Counter(lst)
       return count.most_common(1)[0][0]

   print(most_frequent([1, 2, 2, 3, 3, 3, 4]))  # Output: 3
   ```

These practical questions test your ability to write efficient and effective Python code, focusing on problem-solving, algorithm implementation, and understanding Python's advanced features such as decorators, iterators, and context managers.

Here are some advanced Python exercises that cover a range of topics such as decorators, generators, metaprogramming, context managers, and advanced data structures. These exercises are designed to deepen your understanding of Python’s features and help you write more efficient and elegant code.

---

### 1. **Custom Decorator for Timing Function Execution**

Write a decorator that measures the execution time of a function and logs it. The decorator should print the function name, arguments, and execution time. Use the `time` module.

**Requirements:**

- Decorator should handle both positional and keyword arguments.
- It should print out the function name, arguments, and execution time.

```python
import time

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function {func.__name__} executed in {execution_time:.4f} seconds with arguments {args} and keyword arguments {kwargs}")
        return result
    return wrapper

# Example usage:
@timer_decorator
def slow_function(x, y):
    time.sleep(2)
    return x + y

slow_function(3, 4)
```

---

### 2. **Generator for Fibonacci Sequence**

Create a generator function that yields Fibonacci numbers up to a given number `n`. The generator should use the `yield` keyword.

**Requirements:**

- The generator should continue yielding Fibonacci numbers until `n` is reached.
- It should stop when the next Fibonacci number exceeds `n`.

```python
def fibonacci_generator(n):
    a, b = 0, 1
    while a <= n:
        yield a
        a, b = b, a + b

# Example usage:
for num in fibonacci_generator(100):
    print(num)
```

---

### 3. **Context Manager for File Handling**

Create a context manager that automatically closes a file after writing some content into it. You should use the `with` statement to handle file writing and ensure that the file is properly closed after the operation.

```python
class FileWriter:
    def __init__(self, filename):
        self.filename = filename
    
    def __enter__(self):
        self.file = open(self.filename, 'w')
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

# Example usage:
with FileWriter('output.txt') as file:
    file.write('Hello, World!')
```

---

### 4. **Metaclass to Enforce Singleton Pattern**

Write a metaclass that ensures only one instance of a class can exist at any given time. The class should raise an exception if a new instance is created after the first one.

**Requirements:**

- The metaclass should manage the instance and ensure that only one instance of the class is allowed.
- Use the `__call__` method of the metaclass to manage instance creation.

```python
class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class SingletonClass(metaclass=SingletonMeta):
    def __init__(self):
        print("SingletonClass instance created")

# Example usage:
obj1 = SingletonClass()
obj2 = SingletonClass()

# Check if both objects are the same instance
print(obj1 is obj2)  # True
```

---

### 5. **Advanced Sorting with Custom Key Functions**

Create a function that sorts a list of dictionaries by multiple keys, where each key has a different sort order (ascending or descending). You should use the `sorted()` function with a custom key.

**Requirements:**

- The function should accept a list of dictionaries.
- The sort order should be specified by a list of tuples, where each tuple contains the key and the desired order (`True` for ascending, `False` for descending).

```python
def multi_key_sort(data, keys):
    def sort_key(item):
        return tuple((item[k] if order else -item[k]) for k, order in keys)

    return sorted(data, key=sort_key)

# Example usage:
data = [
    {"name": "Alice", "age": 30, "salary": 50000},
    {"name": "Bob", "age": 25, "salary": 70000},
    {"name": "Charlie", "age": 35, "salary": 60000},
]

keys = [("age", True), ("salary", False)]  # Sort by age ascending, salary descending
sorted_data = multi_key_sort(data, keys)
print(sorted_data)
```

---

### 6. **Custom Iterator for Prime Numbers**

Write a custom iterator that generates prime numbers. The iterator should yield the next prime number each time it’s called.

**Requirements:**

- The iterator should use the `__next__` method to yield the next prime number.
- The iterator should be able to be reset to start from the first prime number again.

```python
class PrimeIterator:
    def __init__(self):
        self.num = 2

    def __iter__(self):
        return self
    
    def __next__(self):
        while True:
            if self._is_prime(self.num):
                prime = self.num
                self.num += 1
                return prime
            self.num += 1

    def _is_prime(self, n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

# Example usage:
primes = PrimeIterator()
for _ in range(10):
    print(next(primes))
```

---

### 7. **Dynamic Class Creation Using `type()`**

Use the `type()` function to dynamically create a class at runtime. The class should inherit from a base class and have a method that prints a message.

**Requirements:**

- The dynamic class should inherit from a given base class.
- The class should have a method that prints a custom message.

```python
def create_class(name, base_class):
    def custom_method(self):
        print(f"{self.__class__.__name__} instance created!")
    
    return type(name, (base_class,), {"custom_method": custom_method})

class Base:
    pass

# Example usage:
DynamicClass = create_class("DynamicClass", Base)
obj = DynamicClass()
obj.custom_method()  # Output: DynamicClass instance created!
```

---

### 8. **Implementing a Linked List**

Implement a simple singly linked list with methods to insert nodes, delete nodes, and print the list. Define a `Node` class and a `LinkedList` class.

**Requirements:**

- The `Node` class should represent a node in the list.
- The `LinkedList` class should have methods like `insert`, `delete`, and `print_list`.

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def insert(self, data):
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node

    def delete(self, data):
        current = self.head
        if current and current.data == data:
            self.head = current.next
            return
        prev = None
        while current:
            if current.data == data:
                prev.next = current.next
                return
            prev = current
            current = current.next

    def print_list(self):
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

# Example usage:
ll = LinkedList()
ll.insert(10)
ll.insert(20)
ll.insert(30)
ll.print_list()  # Output: 30 -> 20 -> 10 -> None
ll.delete(20)
ll.print_list()  # Output: 30 -> 10 -> None
```

---

These exercises touch on several advanced Python topics and will challenge you to apply your knowledge of Python's features like decorators, generators, context managers, metaclasses, and more. Try to implement them step-by-step, and experiment with different variations to fully grasp these concepts!

Sure! Here are a few advanced exercises related to dunder methods (special methods) in Python. These exercises will help you get more comfortable working with the Python data model.

### Exercise 1: Custom Complex Number Class

Create a custom class `MyComplex` that represents a complex number, and implement the following dunder methods:

- `__init__(self, real, imag)` – for initializing a complex number.
- `__repr__(self)` – for string representation of the complex number.
- `__add__(self, other)` – to add two complex numbers.
- `__sub__(self, other)` – to subtract two complex numbers.
- `__mul__(self, other)` – to multiply two complex numbers.
- `__truediv__(self, other)` – to divide two complex numbers.
- `__eq__(self, other)` – to check equality of two complex numbers.

```python
class MyComplex:
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

    def __repr__(self):
        return f'{self.real} + {self.imag}i'

    def __add__(self, other):
        if isinstance(other, MyComplex):
            return MyComplex(self.real + other.real, self.imag + other.imag)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, MyComplex):
            return MyComplex(self.real - other.real, self.imag - other.imag)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, MyComplex):
            real_part = self.real * other.real - self.imag * other.imag
            imag_part = self.real * other.imag + self.imag * other.real
            return MyComplex(real_part, imag_part)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, MyComplex):
            denominator = other.real**2 + other.imag**2
            real_part = (self.real * other.real + self.imag * other.imag) / denominator
            imag_part = (self.imag * other.real - self.real * other.imag) / denominator
            return MyComplex(real_part, imag_part)
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, MyComplex):
            return self.real == other.real and self.imag == other.imag
        return False

# Test your class
c1 = MyComplex(3, 2)
c2 = MyComplex(1, 7)
print(c1 + c2)  # Should print the sum of c1 and c2
print(c1 - c2)  # Should print the difference
print(c1 * c2)  # Should print the product
print(c1 / c2)  # Should print the division result
print(c1 == c2) # Should print False
```

---

### Exercise 2: Custom String Formatter

Create a class `MyString` that behaves like a string but adds additional functionality. Implement the following dunder methods:

- `__init__(self, value)` – for initializing the string value.
- `__str__(self)` – for string representation.
- `__add__(self, other)` – to concatenate two strings.
- `__len__(self)` – to return the length of the string.
- `__contains__(self, item)` – to check if a substring is present in the string.
- `__getitem__(self, index)` – to get a character at a specific index.
- `__call__(self)` – to return the string in uppercase.

```python
class MyString:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"MyString('{self.value}')"

    def __add__(self, other):
        if isinstance(other, MyString):
            return MyString(self.value + other.value)
        return NotImplemented

    def __len__(self):
        return len(self.value)

    def __contains__(self, item):
        return item in self.value

    def __getitem__(self, index):
        return self.value[index]

    def __call__(self):
        return self.value.upper()

# Test your class
s1 = MyString("Hello")
s2 = MyString(" World")
print(s1 + s2)    # Should concatenate the strings
print(len(s1))    # Should print the length of s1
print("lo" in s1)  # Should check if "lo" is a substring
print(s1[1])       # Should print the second character of s1
print(s1())        # Should return the string in uppercase
```

---

### Exercise 3: Implementing Iterable Class

Create a custom iterable class `MyRange` that behaves like `range()` but with custom behavior. Implement the following dunder methods:

- `__init__(self, start, stop, step)` – for initializing the range.
- `__iter__(self)` – to return the iterator object.
- `__next__(self)` – to return the next value in the range.
- `__contains__(self, item)` – to check if an item is in the range.

```python
class MyRange:
    def __init__(self, start, stop, step=1):
        self.start = start
        self.stop = stop
        self.step = step
        self.current = start

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.stop:
            raise StopIteration
        value = self.current
        self.current += self.step
        return value

    def __contains__(self, item):
        return self.start <= item < self.stop and (item - self.start) % self.step == 0

# Test your class
r = MyRange(0, 10, 2)
for num in r:
    print(num)  # Should print: 0, 2, 4, 6, 8

print(4 in r)  # Should print: True
print(5 in r)  # Should print: False
```

---

### Exercise 4: Custom Context Manager

Implement a custom context manager using `__enter__` and `__exit__`. The context manager will log when the code enters and exits a block, and handle exceptions.

```python
class MyContextManager:
    def __enter__(self):
        print("Entering the context")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("Exiting the context")
        if exc_type:
            print(f"An exception occurred: {exc_value}")
        return True  # Suppress exception

# Test your context manager
with MyContextManager():
    print("Inside the context")
    # Uncomment the next line to test exception handling
    # raise ValueError("Something went wrong")
```

**Expected output without an exception:**

```
Entering the context
Inside the context
Exiting the context
```

**Expected output with an exception:**

```
Entering the context
Inside the context
Exiting the context
An exception occurred: Something went wrong
```

---

### Exercise 5: Custom Descriptor

Create a class `Age` that acts as a descriptor to enforce age constraints in another class `Person`. Implement the following:

- `__get__(self, instance, owner)` – returns the value of age.
- `__set__(self, instance, value)` – sets the value of age, but ensures it's between 0 and 120.
- `__delete__(self, instance)` – deletes the age attribute.

```python
class Age:
    def __get__(self, instance, owner):
        return instance._age

    def __set__(self, instance, value):
        if 0 <= value <= 120:
            instance._age = value
        else:
            raise ValueError("Age must be between 0 and 120.")

    def __delete__(self, instance):
        del instance._age

class Person:
    age = Age()

    def __init__(self, name, age):
        self.name = name
        self.age = age

# Test your class
p = Person("John", 25)
print(p.age)  # Should print: 25
p.age = 30     # Should set age to 30
print(p.age)
# p.age = 130  # Uncommenting this should raise a ValueError
del p.age     # Should delete the age attribute
```

---

### Exercise 6: Implementing `__call__`

Create a class `Counter` that increments a counter value each time it is called. Implement:

- `__init__(self)` – for initializing the counter value.
- `__call__(self)` – for incrementing and returning the counter.

```python
class Counter:
    def __init__(self):
        self.count = 0

    def __call__(self):
        self.count += 1
        return self.count

# Test your class
counter = Counter()
print(counter())  # Should print: 1
print(counter())  # Should print: 2
print(counter())  # Should print: 3
```

In Python, Object-Oriented Programming (OOP) allows you to structure your code in a way that mimics real-world objects and their interactions. The advanced concepts in OOP include things like multiple inheritance, method resolution order (MRO), mixins, abstract base classes (ABCs), class methods, static methods, and decorators.

Here's an explanation of these concepts with implementation examples:

### 1. **Multiple Inheritance & Method Resolution Order (MRO)**

Multiple inheritance occurs when a class inherits from more than one base class. Python uses the C3 linearization algorithm to resolve method calls in a way that respects the inheritance hierarchy.

#### Example of Multiple Inheritance

```python
class Animal:
    def sound(self):
        return "Some generic animal sound"

class Bird:
    def sound(self):
        return "Chirp"

class Dog(Animal, Bird):
    def sound(self):
        return "Bark"

# Instantiating Dog class
dog = Dog()
print(dog.sound())  # Output: Bark

# Check method resolution order
print(Dog.__mro__)
```

In the code above, `Dog` inherits from both `Animal` and `Bird`. The method resolution order (`__mro__`) will help us understand the order in which methods are looked up when invoked.

### 2. **Mixins**

A mixin is a class designed to be inherited by other classes to provide functionality, but it’s not meant to be instantiated on its own. Mixins usually provide reusable functionality for multiple classes.

#### Example of a Mixin

```python
class CanFly:
    def fly(self):
        return "Flying!"

class CanSwim:
    def swim(self):
        return "Swimming!"

class Duck(CanFly, CanSwim):
    def quack(self):
        return "Quack!"

# Instantiating Duck
duck = Duck()
print(duck.fly())  # Output: Flying!
print(duck.swim())  # Output: Swimming!
print(duck.quack())  # Output: Quack!
```

In this example, `Duck` inherits from two mixins, `CanFly` and `CanSwim`, which provide it with flying and swimming capabilities.

### 3. **Abstract Base Classes (ABCs)**

Abstract base classes define methods that must be implemented in subclasses. This ensures that subclasses follow a certain interface.

#### Example of ABC

```python
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def make_sound(self):
        pass

class Dog(Animal):
    def make_sound(self):
        return "Bark"

class Cat(Animal):
    def make_sound(self):
        return "Meow"

# Instantiating the classes
dog = Dog()
print(dog.make_sound())  # Output: Bark

cat = Cat()
print(cat.make_sound())  # Output: Meow

# Uncommenting the following line will raise an error
# animal = Animal()  # TypeError: Can't instantiate abstract class Animal with abstract methods make_sound
```

Here, `Animal` is an abstract base class with an abstract method `make_sound`. Both `Dog` and `Cat` implement this method, but you cannot instantiate `Animal` directly.

### 4. **Class Methods & Static Methods**

Class methods and static methods allow for defining methods that are not bound to instances but are still associated with the class.

- **Class Methods**: Operate on the class itself, not an instance, and have access to the class state.
- **Static Methods**: Do not operate on the class or instance at all and are independent methods.

#### Example of Class and Static Methods

```python
class Calculator:
    base_value = 10

    def __init__(self, value):
        self.value = value

    @classmethod
    def set_base_value(cls, new_base_value):
        cls.base_value = new_base_value

    @staticmethod
    def add(x, y):
        return x + y

    def calculate(self):
        return self.value + self.base_value

# Using Class Method to change class-level value
Calculator.set_base_value(20)

# Using Static Method
print(Calculator.add(5, 3))  # Output: 8

# Instantiating Calculator
calc = Calculator(5)
print(calc.calculate())  # Output: 25 (5 + 20)
```

In this example, `set_base_value` is a class method that modifies the class-level `base_value`, while `add` is a static method that performs a simple addition.

### 5. **Property Decorators**

The `property` decorator allows you to define getter, setter, and deleter methods for attributes, making them more manageable and controlled.

#### Example of Property Decorators

```python
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("Radius cannot be negative.")
        self._radius = value

    @property
    def area(self):
        return 3.1415 * self._radius ** 2

# Instantiating Circle
circle = Circle(5)
print(circle.radius)  # Output: 5
print(circle.area)    # Output: 78.5375

circle.radius = 10
print(circle.area)    # Output: 314.15

# Uncommenting the following will raise an error
# circle.radius = -5  # ValueError: Radius cannot be negative
```

Here, `radius` is a property, and `area` is another property that calculates the area based on the radius.

### 6. **Dynamic Class Creation with `type`**

You can create classes dynamically at runtime using the built-in `type()` function.

#### Example of Dynamic Class Creation

```python
# Dynamically creating a class
Person = type('Person', (object,), {'greet': lambda self: "Hello!"})

# Instantiating the dynamically created class
p = Person()
print(p.greet())  # Output: Hello!
```

In this case, `Person` is created dynamically using `type()`. The class has a `greet` method, and you can instantiate and use it just like any other class.

### 7. **Operator Overloading**

Python allows you to overload operators (such as `+`, `-`, `*`, etc.) by defining special methods in the class.

#### Example of Operator Overloading

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

# Instantiating Vectors
v1 = Vector(2, 3)
v2 = Vector(4, 5)

# Using overloaded '+' operator
v3 = v1 + v2
print(v3)  # Output: Vector(6, 8)
```

In this example, the `+` operator is overloaded to add two `Vector` objects.

---

These are some advanced OOP concepts in Python that help create more flexible, reusable, and maintainable code. If you want more detail on any specific topic or additional examples, feel free to ask
---

### Bonus Exercise: Metaclass Example

Create a metaclass `UppercaseMeta` that automatically converts all class variable names to uppercase when a class is created.

```python
class UppercaseMeta(type):
    def __new__(cls, name, bases, dct):
        uppercase_attributes = {
            key.upper(): value for key, value in dct.items()
        }
        return super().__new__(cls, name, bases, uppercase_attributes)

class MyClass(metaclass=UppercaseMeta):
    lowercase = "I am a lowercase attribute"

# Test your class
print(hasattr(MyClass, "lowercase"))  # Should print False
print(hasattr(MyClass, "LOWERCASE"))  # Should print True
```

---

These exercises will provide you with a deeper understanding of Python’s special methods and how they fit into the language’s data model. Have fun implementing and experimenting!

Here's an advanced Python interview exercise that tests deep knowledge of Python's features, including object-oriented programming (OOP), generators, decorators, context managers, and functional programming concepts.

---

### Problem 1: Implementing a Custom Context Manager

Write a context manager that ensures the following conditions:

1. Upon entering the context, it will lock a file to prevent other processes from modifying it. You can simulate this by printing a message about acquiring a lock (you can assume no actual file locking is necessary for the test).
2. Upon exiting the context, the lock should be released (simulated by a message).
3. If any exception occurs within the context, it should log the error message and re-raise the exception after releasing the lock.

**Requirements**:

- Implement this using Python's `with` statement and context manager protocol (`__enter__` and `__exit__`).
- Simulate file locking with print statements.

```python
class FileLock:
    def __init__(self, filename: str):
        self.filename = filename

    def __enter__(self):
        # Acquire lock
        print(f"Lock acquired for {self.filename}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Release lock, and handle any exception raised within the context
        print(f"Lock released for {self.filename}")
        if exc_type:
            print(f"An error occurred: {exc_value}")
        # Returning False will allow the exception to propagate, True would suppress it
        return False
```

---

### Problem 2: Generator-based Lazy Evaluation

Create a generator `lazy_square` that lazily evaluates and yields the square of integers starting from 0 up to a given number `n` (exclusive). The generator should use the `yield` keyword, and the caller should be able to consume the squares one at a time, without computing all of them at once.

```python
def lazy_square(n: int):
    for i in range(n):
        yield i ** 2
```

**Test Case**:

```python
squares = lazy_square(5)
for square in squares:
    print(square)
```

**Expected Output**:

```
0
1
4
9
16
```

---

### Problem 3: Decorators and Memoization

Create a decorator `memoize` that caches the results of expensive function calls so that subsequent calls with the same arguments can be returned instantly.

```python
def memoize(func):
    cache = {}
    
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    
    return wrapper
```

Test the decorator with a function that computes Fibonacci numbers.

```python
@memoize
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Test the memoization
print(fibonacci(10))  # Should compute Fibonacci number 55 without recalculating intermediate values
```

---

### Problem 4: Class Method and Static Method

Given the following class, implement the `from_string` class method to create an instance of `Rectangle` from a string in the format `"width,height"`, and the `area` static method to calculate the area of the rectangle.

```python
class Rectangle:
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    @classmethod
    def from_string(cls, s: str):
        width, height = map(float, s.split(','))
        return cls(width, height)
    
    @staticmethod
    def area(width: float, height: float):
        return width * height
```

Test case:

```python
rect = Rectangle.from_string("4.5,3.2")
print(rect.width, rect.height)  # Output: 4.5 3.2
print(Rectangle.area(rect.width, rect.height))  # Output: 14.4
```

---

### Problem 5: Advanced Iterable Handling with `__iter__` and `__next__`

Create a class `PrimeGenerator` that generates prime numbers up to a given limit. The class should implement the iterator protocol (`__iter__` and `__next__`).

```python
class PrimeGenerator:
    def __init__(self, limit: int):
        self.limit = limit
        self.current = 2

    def __iter__(self):
        return self

    def __next__(self):
        while self.current <= self.limit:
            if self.is_prime(self.current):
                prime = self.current
                self.current += 1
                return prime
            self.current += 1
        raise StopIteration

    def is_prime(self, n: int):
        if n <= 1:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True
```

Test case:

```python
primes = PrimeGenerator(30)
for prime in primes:
    print(prime)
```

**Expected Output**:

```
2
3
5
7
11
13
17
19
23
29
```

---

### Problem 6: Custom Deserialization

Create a custom `JSONDecoder` that deserializes JSON-like strings to Python dictionaries, handling both the basic types (`int`, `float`, `str`, `bool`, `None`) and lists or dictionaries.

You may assume that the input string is well-formed and use Python's built-in `json` library for comparison.

```python
import json

class CustomJSONDecoder:
    def decode(self, s: str):
        # This method should parse the string and return the corresponding Python object.
        # Implement it without using json.loads directly.
        pass
```

Test case:

```python
decoder = CustomJSONDecoder()
data = '{"name": "Alice", "age": 30, "is_employee": true, "skills": ["Python", "Java"]}'
print(decoder.decode(data))
```

**Expected Output**:

```python
{'name': 'Alice', 'age': 30, 'is_employee': True, 'skills': ['Python', 'Java']}
```

---

These exercises test a candidate's knowledge of advanced Python features such as context managers, generators, decorators, static/class methods, iterators, and custom deserialization logic. Be sure to implement and test each solution independently!

Here are some advanced Python exercises that many people may not know, but which will challenge your understanding of Python concepts such as metaprogramming, decorators, context managers, and advanced data structures.

### 1. **Metaclasses and Dynamic Class Creation**

Metaclasses allow you to control the creation of classes in Python. This exercise involves creating a metaclass that automatically adds a `__str__` method to any class that doesn't have it.

**Exercise:**
Create a metaclass that ensures every class has a `__str__` method. If a class does not define it, the metaclass should automatically generate a default `__str__` method that prints out the name of the class.

```python
class AutoStrMeta(type):
    def __new__(cls, name, bases, dct):
        if '__str__' not in dct:
            dct['__str__'] = lambda self: f'{self.__class__.__name__} instance'
        return super().__new__(cls, name, bases, dct)

class MyClass(metaclass=AutoStrMeta):
    pass

# Test
obj = MyClass()
print(obj)  # Should print: "MyClass instance"
```

### 2. **Custom Context Manager with `__enter__` and `__exit__`**

Context managers are useful for managing resources like file handles, network connections, or database sessions. You can create a custom context manager to handle operations automatically.

**Exercise:**
Create a custom context manager that measures the time taken to execute a block of code using `time` module.

```python
import time

class TimerContextManager:
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        print(f'Elapsed time: {self.end_time - self.start_time:.4f} seconds')

# Test
with TimerContextManager() as timer:
    sum(range(1_000_000))  # Some operation
```

### 3. **Function Caching with a Custom Decorator**

Python's built-in `functools.lru_cache` decorator caches results of functions. But you can write your own caching decorator to implement caching based on custom logic.

**Exercise:**
Create a decorator that caches function results using a dictionary and only invalidates the cache when explicitly told to do so.

```python
def cache_decorator(func):
    cache = {}
    
    def wrapper(*args):
        if args not in cache:
            print(f"Cache miss for {args}")
            cache[args] = func(*args)
        else:
            print(f"Cache hit for {args}")
        return cache[args]
    
    wrapper.clear_cache = lambda: cache.clear()
    return wrapper

@cache_decorator
def expensive_function(x):
    return x * x

# Test
print(expensive_function(4))  # Cache miss
print(expensive_function(4))  # Cache hit
print(expensive_function(5))  # Cache miss
expensive_function.clear_cache()  # Manually clear cache
print(expensive_function(4))  # Cache miss again
```

### 4. **Creating a Singleton Class**

A singleton is a design pattern where a class ensures only one instance is created. You can implement a singleton pattern using a metaclass.

**Exercise:**
Implement the Singleton design pattern using a metaclass.

```python
class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class SingletonClass(metaclass=SingletonMeta):
    def __init__(self, value):
        self.value = value

# Test
obj1 = SingletonClass(1)
obj2 = SingletonClass(2)
print(obj1 is obj2)  # Should be True
```

### 5. **Descriptors and Property Management**

Descriptors are a low-level mechanism that allows customization of attribute access, which can be used for validation, lazy loading, etc.

**Exercise:**
Create a descriptor that validates whether the assigned value is a positive integer.

```python
class PositiveInteger:
    def __get__(self, instance, owner):
        return instance._value

    def __set__(self, instance, value):
        if value <= 0:
            raise ValueError("Value must be positive.")
        instance._value = value

class MyClass:
    positive_value = PositiveInteger()

# Test
obj = MyClass()
obj.positive_value = 10  # Valid
print(obj.positive_value)  # 10

obj.positive_value = -5  # Raises ValueError
```

### 6. **Lazy Evaluation with `__iter__`**

Lazy evaluation is a programming technique where you delay computation until the result is actually needed. This is often used in situations like large datasets or infinite sequences.

**Exercise:**
Create a generator that lazily computes Fibonacci numbers.

```python
class Fibonacci:
    def __init__(self):
        self.a, self.b = 0, 1
    
    def __iter__(self):
        return self
    
    def __next__(self):
        self.a, self.b = self.b, self.a + self.b
        return self.a

# Test
fib = Fibonacci()
for i, num in zip(range(10), fib):
    print(num)
```

### 7. **Advanced Comprehensions and Generator Expressions**

You can use advanced comprehension techniques to handle complex data transformations and filtering.

**Exercise:**
Create a nested list comprehension to find all pairs of numbers in a list that add up to a specific target sum.

```python
numbers = [1, 2, 3, 4, 5, 6]
target = 7

pairs = [(x, y) for i, x in enumerate(numbers) for y in numbers[i+1:] if x + y == target]
print(pairs)  # Should print [(1, 6), (2, 5), (3, 4)]
```

### 8. **Handling Circular Imports**

Circular imports are a common issue in larger Python projects. You can design your program to handle circular imports gracefully.

**Exercise:**
Write a small program with two modules that import each other and resolve the circular import using `importlib`.

- `module_a.py`: Contains a reference to `module_b`.
- `module_b.py`: Contains a reference to `module_a`.

```python
# module_a.py
import importlib

def call_b():
    module_b = importlib.import_module('module_b')
    module_b.call_a()

def call_a():
    print("Function call_a from module_a")

# module_b.py
import importlib

def call_a():
    module_a = importlib.import_module('module_a')
    module_a.call_b()

def call_b():
    print("Function call_b from module_b")
```

### 9. **Context-Sensitive Logger**

Build a logger that can dynamically adjust its logging level based on the context.

**Exercise:**
Create a logger class that changes its logging level within certain scopes, using a context manager.

```python
import logging

class ContextSensitiveLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def debug(self, msg):
        self.logger.debug(msg)
    
    def info(self, msg):
        self.logger.info(msg)
    
    def set_level(self, level):
        self.logger.setLevel(level)

    def log_with_context(self, context, msg):
        with self.contextualize(context):
            self.logger.info(msg)
    
    class contextualize:
        def __init__(self, context):
            self.context = context
        def __enter__(self):
            logging.getLogger().setLevel(self.context)
        def __exit__(self, exc_type, exc_value, traceback):
            logging.getLogger().setLevel(logging.DEBUG)
```

These exercises will help you explore some advanced Python techniques that are not as commonly used. Mastering them can significantly improve your ability to write efficient, elegant, and highly Pythonic code.

Here are some advanced Python string exercises often encountered in interviews, along with their examples and solutions. These exercises test your understanding of Python string manipulation techniques, such as regular expressions, slicing, formatting, and working with more complex scenarios.

### 1. **Reverse Words in a String**

**Problem**: Given a string, reverse the order of words while maintaining the order of characters within each word.

#### Example

```python
Input: "The quick brown fox"
Output: "fox brown quick The"
```

#### Solution

```python
def reverse_words(s: str) -> str:
    # Split the string by spaces and reverse the list of words
    words = s.split()
    return ' '.join(reversed(words))

# Test the function
input_str = "The quick brown fox"
print(reverse_words(input_str))  # Output: "fox brown quick The"
```

### 2. **Check if a String is a Palindrome**

**Problem**: Check whether a given string is a palindrome (ignoring spaces, punctuation, and case sensitivity).

#### Example

```python
Input: "A man, a plan, a canal, Panama"
Output: True
```

#### Solution

```python
import re

def is_palindrome(s: str) -> bool:
    # Remove non-alphanumeric characters and convert to lowercase
    cleaned_str = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
    return cleaned_str == cleaned_str[::-1]

# Test the function
input_str = "A man, a plan, a canal, Panama"
print(is_palindrome(input_str))  # Output: True
```

### 3. **Count Occurrences of a Substring in a String**

**Problem**: Given a string and a substring, count how many times the substring appears in the string without overlapping.

#### Example

```python
Input: "ababcabcab"
Substring: "ab"
Output: 3
```

#### Solution

```python
def count_substring(s: str, sub: str) -> int:
    return s.count(sub)

# Test the function
input_str = "ababcabcab"
substring = "ab"
print(count_substring(input_str, substring))  # Output: 3
```

### 4. **Longest Palindromic Substring**

**Problem**: Given a string, find the longest substring which is a palindrome.

#### Example

```python
Input: "babad"
Output: "bab" or "aba" (both are correct)
```

#### Solution

```python
def longest_palindromic_substring(s: str) -> str:
    def expand_around_center(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left+1:right]

    longest = ""
    for i in range(len(s)):
        # Odd length palindrome
        odd_palindrome = expand_around_center(i, i)
        if len(odd_palindrome) > len(longest):
            longest = odd_palindrome
        # Even length palindrome
        even_palindrome = expand_around_center(i, i+1)
        if len(even_palindrome) > len(longest):
            longest = even_palindrome
    return longest

# Test the function
input_str = "babad"
print(longest_palindromic_substring(input_str))  # Output: "bab" or "aba"
```

### 5. **Remove Duplicate Characters**

**Problem**: Given a string, remove all duplicate characters while keeping the first occurrence of each character.

#### Example

```python
Input: "aabbccabc"
Output: "abc"
```

#### Solution

```python
def remove_duplicates(s: str) -> str:
    return ''.join(sorted(set(s), key=s.index))

# Test the function
input_str = "aabbccabc"
print(remove_duplicates(input_str))  # Output: "abc"
```

### 6. **Find the First Non-Repeating Character**

**Problem**: Given a string, find the first character that does not repeat in the string.

#### Example

```python
Input: "geeksforgeeks"
Output: "f"
```

#### Solution

```python
from collections import Counter

def first_non_repeating(s: str) -> str:
    # Count frequency of each character
    freq = Counter(s)
    for char in s:
        if freq[char] == 1:
            return char
    return None

# Test the function
input_str = "geeksforgeeks"
print(first_non_repeating(input_str))  # Output: "f"
```

### 7. **String Compression**

**Problem**: Given a string, compress it using the counts of repeated characters. For example, "aabcccccaaa" becomes "a2b1c5a3". If the compressed string is not smaller than the original string, return the original string.

#### Example

```python
Input: "aabcccccaaa"
Output: "a2b1c5a3"
```

#### Solution

```python
def compress_string(s: str) -> str:
    compressed = []
    count = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            count += 1
        else:
            compressed.append(s[i - 1] + str(count))
            count = 1
    compressed.append(s[-1] + str(count))  # for the last character

    compressed_str = ''.join(compressed)
    return compressed_str if len(compressed_str) < len(s) else s

# Test the function
input_str = "aabcccccaaa"
print(compress_string(input_str))  # Output: "a2b1c5a3"
```

### 8. **Find All Anagrams of a String in Another String**

**Problem**: Given two strings `s` and `p`, return all the start indices of `p`'s anagrams in `s`. Strings consists of lowercase English letters.

#### Example

```python
Input: s = "cbaebabacd", p = "abc"
Output: [0, 6]
```

#### Solution

```python
from collections import Counter

def find_anagrams(s: str, p: str):
    result = []
    p_count = Counter(p)
    s_count = Counter()

    for i in range(len(s)):
        s_count[s[i]] += 1
        if i >= len(p):
            if s_count[s[i - len(p)]] == 1:
                del s_count[s[i - len(p)]]
            else:
                s_count[s[i - len(p)]] -= 1

        if s_count == p_count:
            result.append(i - len(p) + 1)

    return result

# Test the function
s = "cbaebabacd"
p = "abc"
print(find_anagrams(s, p))  # Output: [0, 6]
```

These exercises are commonly used in coding interviews to assess problem-solving skills, familiarity with string operations, and the ability to write efficient code. They also help to test knowledge of Python libraries like `re` for regular expressions and `collections.Counter` for counting occurrences in strings.

In Python, the term "oops" refers to **Object-Oriented Programming (OOP)**, a programming paradigm that uses objects and classes to organize code. The advanced concepts in OOP typically include the following:

1. **Inheritance**
2. **Polymorphism**
3. **Encapsulation**
4. **Abstraction**
5. **Method Overriding**
6. **Multiple Inheritance**
7. **Super() and super method**
8. **Static Methods and Class Methods**
9. **Magic Methods (Dunder Methods)**

Let's go over these concepts with code examples:

### 1. Inheritance

Inheritance allows one class to inherit the attributes and methods of another class.

```python
class Animal:
    def speak(self):
        return "Animal sound"

class Dog(Animal):  # Dog inherits from Animal
    def speak(self):
        return "Bark"

dog = Dog()
print(dog.speak())  # Output: Bark
```

### 2. Polymorphism

Polymorphism allows methods to have different implementations based on the object type.

```python
class Cat(Animal):
    def speak(self):
        return "Meow"

animals = [Dog(), Cat()]
for animal in animals:
    print(animal.speak())  # Output: Bark, Meow
```

### 3. Encapsulation

Encapsulation is the bundling of data and methods that operate on the data within a class, and restricting direct access to some of the object's components.

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.__age = age  # Private variable

    def get_age(self):  # Public method to access private variable
        return self.__age

    def set_age(self, age):  # Public method to set private variable
        if age >= 0:
            self.__age = age
        else:
            print("Age cannot be negative")

p = Person("John", 30)
print(p.get_age())  # Output: 30
p.set_age(35)
print(p.get_age())  # Output: 35
```

### 4. Abstraction

Abstraction allows hiding complex implementation details and exposing only necessary parts.

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14 * (self.radius ** 2)

circle = Circle(5)
print(circle.area())  # Output: 78.5
```

### 5. Method Overriding

Method overriding occurs when a subclass provides its own implementation of a method that is already defined in the parent class.

```python
class Animal:
    def speak(self):
        return "Some generic sound"

class Dog(Animal):
    def speak(self):
        return "Bark"

dog = Dog()
print(dog.speak())  # Output: Bark (Overridden method)
```

### 6. Multiple Inheritance

Multiple inheritance allows a class to inherit from more than one parent class.

```python
class Animal:
    def speak(self):
        return "Animal sound"

class Canine:
    def walk(self):
        return "Walking on 4 legs"

class Dog(Animal, Canine):
    pass

dog = Dog()
print(dog.speak())  # Output: Animal sound
print(dog.walk())   # Output: Walking on 4 legs
```

### 7. Using `super()`

`super()` allows calling methods from a parent class in the child class.

```python
class Animal:
    def speak(self):
        return "Animal sound"

class Dog(Animal):
    def speak(self):
        return super().speak() + " and Bark"

dog = Dog()
print(dog.speak())  # Output: Animal sound and Bark
```

### 8. Static Methods and Class Methods

- **Static methods**: Do not operate on an instance of the class.
- **Class methods**: Operate on the class itself.

```python
class MyClass:
    @staticmethod
    def static_method():
        return "This is a static method"

    @classmethod
    def class_method(cls):
        return f"This is a class method of {cls.__name__}"

print(MyClass.static_method())  # Output: This is a static method
print(MyClass.class_method())   # Output: This is a class method of MyClass
```

### 9. Magic Methods (Dunder Methods)

Magic methods are special methods that have double underscores before and after their name. They are used to define how instances of the class behave in certain operations.

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

point1 = Point(2, 3)
point2 = Point(4, 5)

print(point1 + point2)  # Output: Point(6, 8)
```

### Conclusion

These advanced OOP concepts in Python help structure and manage complex programs. Understanding inheritance, polymorphism, encapsulation, abstraction, and method overriding can greatly enhance your ability to design and maintain object-oriented systems.

Here is a comprehensive guide to **Python sorting algorithms**, including explanations, example exercises, and their solutions.

### 1. **Bubble Sort**

Bubble sort is a simple comparison-based sorting algorithm. It repeatedly steps through the list, compares adjacent elements, and swaps them if they are in the wrong order. This process is repeated until the list is sorted.

#### Example Code

```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr
```

#### Exercise

Sort the following list using Bubble Sort: `[64, 34, 25, 12, 22, 11, 90]`

#### Solution

```python
arr = [64, 34, 25, 12, 22, 11, 90]
sorted_arr = bubble_sort(arr)
print(sorted_arr)  # Output: [11, 12, 22, 25, 34, 64, 90]
```

---

### 2. **Selection Sort**

Selection sort is another comparison-based sorting algorithm. It repeatedly selects the smallest (or largest) element from the unsorted portion of the list and swaps it with the first unsorted element.

#### Example Code

```python
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr
```

#### Exercise

Sort the following list using Selection Sort: `[29, 10, 14, 37, 13]`

#### Solution

```python
arr = [29, 10, 14, 37, 13]
sorted_arr = selection_sort(arr)
print(sorted_arr)  # Output: [10, 13, 14, 29, 37]
```

---

### 3. **Insertion Sort**

Insertion sort works by taking one element at a time from the unsorted portion and inserting it into the correct position in the sorted portion.

#### Example Code

```python
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr
```

#### Exercise

Sort the following list using Insertion Sort: `[12, 11, 13, 5, 6]`

#### Solution

```python
arr = [12, 11, 13, 5, 6]
sorted_arr = insertion_sort(arr)
print(sorted_arr)  # Output: [5, 6, 11, 12, 13]
```

---

### 4. **Merge Sort**

Merge sort is a divide-and-conquer algorithm. It divides the list into two halves, recursively sorts them, and then merges the sorted halves.

#### Example Code

```python
def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        merge_sort(left_half)
        merge_sort(right_half)

        i = j = k = 0

        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1

    return arr
```

#### Exercise

Sort the following list using Merge Sort: `[38, 27, 43, 3, 9, 82, 10]`

#### Solution

```python
arr = [38, 27, 43, 3, 9, 82, 10]
sorted_arr = merge_sort(arr)
print(sorted_arr)  # Output: [3, 9, 10, 27, 38, 43, 82]
```

---

### 5. **Quick Sort**

Quick sort is a divide-and-conquer algorithm that works by selecting a 'pivot' element and partitioning the array around it.

#### Example Code

```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
```

#### Exercise

Sort the following list using Quick Sort: `[3, 6, 8, 10, 1, 2, 1]`

#### Solution

```python
arr = [3, 6, 8, 10, 1, 2, 1]
sorted_arr = quick_sort(arr)
print(sorted_arr)  # Output: [1, 1, 2, 3, 6, 8, 10]
```

---

### 6. **Heap Sort**

Heap sort is a comparison-based sorting algorithm that uses a binary heap data structure. It first builds a max heap and then repeatedly extracts the maximum element.

#### Example Code

```python
import heapq

def heap_sort(arr):
    heapq.heapify(arr)  # Turn the list into a heap in-place
    return [heapq.heappop(arr) for _ in range(len(arr))]
```

#### Exercise

Sort the following list using Heap Sort: `[4, 10, 3, 5, 1]`

#### Solution

```python
arr = [4, 10, 3, 5, 1]
sorted_arr = heap_sort(arr)
print(sorted_arr)  # Output: [1, 3, 4, 5, 10]
```

---

### 7. **Tim Sort**

Tim Sort is a hybrid sorting algorithm derived from merge sort and insertion sort. It is used in Python’s built-in sorting functions (`sorted()` and `.sort()`).

#### Example Code (Using Python's Built-In Sorting)

```python
arr = [7, 1, 3, 9, 5]
arr.sort()
print(arr)  # Output: [1, 3, 5, 7, 9]
```

#### Exercise

Sort the following list using Tim Sort: `[3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]`

#### Solution

```python
arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
arr.sort()
print(arr)  # Output: [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]
```

---

### 8. **Radix Sort**

Radix sort is a non-comparative integer sorting algorithm. It processes each digit of the number starting from the least significant digit to the most significant digit.

#### Example Code

```python
def counting_sort(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10

    for i in range(n):
        index = arr[i] // exp
        count[index % 10] += 1

    for i in range(1, 10):
        count[i] += count[i - 1]

    i = n - 1
    while i >= 0:
        index = arr[i] // exp
        output[count[index % 10] - 1] = arr[i]
        count[index % 10] -= 1
        i -= 1

    for i in range(n):
        arr[i] = output[i]

def radix_sort(arr):
    max_val = max(arr)
    exp = 1
    while max_val // exp > 0:
        counting_sort(arr, exp)
        exp *= 10
    return arr
```

#### Exercise

Sort the following list using Radix Sort: `[170, 45, 75, 90, 802, 24, 2, 66]`

#### Solution

```python
arr = [170, 45, 75, 90, 802, 24, 2, 66]
sorted_arr = radix_sort(arr)
print(sorted_arr)  # Output: [2, 24, 45, 66, 75, 90, 170, 802]
```

---

### Conclusion

These are common Python sorting algorithms, each with its own strengths and weaknesses. You can practice sorting different arrays using the given algorithms to understand their behavior and performance.

Here is a list of common Python searching algorithms with explanations, exercises, and solutions:

### 1. **Linear Search**

#### Explanation

Linear search is the simplest search algorithm. It checks each element in the list one by one until the desired element is found or the end of the list is reached.

#### Exercise

- Implement a function `linear_search(arr, target)` that returns the index of the target element if it exists in the list, otherwise returns `-1`.

#### Solution

```python
def linear_search(arr, target):
    for index, element in enumerate(arr):
        if element == target:
            return index
    return -1

# Example usage:
arr = [10, 20, 30, 40, 50]
target = 30
print(linear_search(arr, target))  # Output: 2
```

---

### 2. **Binary Search**

#### Explanation

Binary search works only on sorted arrays. It divides the array into halves and eliminates half of the search space after each comparison.

#### Exercise

- Implement a function `binary_search(arr, target)` that returns the index of the target element if it exists, otherwise returns `-1`.

#### Solution

```python
def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

# Example usage:
arr = [10, 20, 30, 40, 50]
target = 30
print(binary_search(arr, target))  # Output: 2
```

---

### 3. **Jump Search**

#### Explanation

Jump search is an algorithm for searching a sorted array. It works by jumping ahead by a fixed number of steps (called `block size`), then performing a linear search within that block.

#### Exercise

- Implement a function `jump_search(arr, target)` that searches for the target element in a sorted array.

#### Solution

```python
import math

def jump_search(arr, target):
    n = len(arr)
    step = int(math.sqrt(n))  # Block size (step size)
    prev = 0
    
    while arr[min(step, n) - 1] < target:
        prev = step
        step += int(math.sqrt(n))
        if prev >= n:
            return -1
    
    for i in range(prev, min(step, n)):
        if arr[i] == target:
            return i
    return -1

# Example usage:
arr = [10, 20, 30, 40, 50]
target = 30
print(jump_search(arr, target))  # Output: 2
```

---

### 4. **Exponential Search**

#### Explanation

Exponential search is an algorithm for searching a sorted array. It starts by checking the first element, then exponentially increases the search range.

#### Exercise

- Implement a function `exponential_search(arr, target)` that searches for a target in a sorted array using exponential search.

#### Solution

```python
def exponential_search(arr, target):
    if arr[0] == target:
        return 0
    i = 1
    while i < len(arr) and arr[i] <= target:
        i *= 2
    
    # Perform binary search in the found range
    return binary_search(arr[i//2: min(i, len(arr))], target)

def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

# Example usage:
arr = [10, 20, 30, 40, 50]
target = 30
print(exponential_search(arr, target))  # Output: 2
```

---

### 5. **Interpolation Search**

#### Explanation

Interpolation search is similar to binary search but instead of dividing the array into two halves, it tries to estimate where the value might be based on the distribution of the numbers.

#### Exercise

- Implement a function `interpolation_search(arr, target)` for searching the target in a sorted array using interpolation search.

#### Solution

```python
def interpolation_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high and arr[low] <= target <= arr[high]:
        pos = low + ((target - arr[low]) * (high - low)) // (arr[high] - arr[low])
        if arr[pos] == target:
            return pos
        elif arr[pos] < target:
            low = pos + 1
        else:
            high = pos - 1
    return -1

# Example usage:
arr = [10, 20, 30, 40, 50]
target = 30
print(interpolation_search(arr, target))  # Output: 2
```

---

### 6. **Ternary Search**

#### Explanation

Ternary search is similar to binary search but divides the array into three parts instead of two.

#### Exercise

- Implement a function `ternary_search(arr, target)` for searching the target element in a sorted array using ternary search.

#### Solution

```python
def ternary_search(arr, target):
    low, high = 0, len(arr) - 1
    while high >= low:
        mid1 = low + (high - low) // 3
        mid2 = high - (high - low) // 3
        if arr[mid1] == target:
            return mid1
        elif arr[mid2] == target:
            return mid2
        elif target < arr[mid1]:
            high = mid1 - 1
        elif target > arr[mid2]:
            low = mid2 + 1
        else:
            low = mid1 + 1
            high = mid2 - 1
    return -1

# Example usage:
arr = [10, 20, 30, 40, 50]
target = 30
print(ternary_search(arr, target))  # Output: 2
```

---

### 7. **Fibonacci Search**

#### Explanation

Fibonacci search is a divide-and-conquer algorithm that uses Fibonacci numbers to calculate the mid-point of the array. It is similar to binary search but instead of halving the search range, it reduces the range based on Fibonacci numbers.

#### Exercise

- Implement a function `fibonacci_search(arr, target)` to find the target element using Fibonacci search.

#### Solution

```python
def fibonacci_search(arr, target):
    n = len(arr)
    fib_m_minus_2 = 0
    fib_m_minus_1 = 1
    fib_m = fib_m_minus_1 + fib_m_minus_2

    while fib_m < n:
        fib_m_minus_2 = fib_m_minus_1
        fib_m_minus_1 = fib_m
        fib_m = fib_m_minus_1 + fib_m_minus_2

    offset = -1
    while fib_m > 1:
        i = min(offset + fib_m_minus_2, n - 1)
        if arr[i] == target:
            return i
        elif arr[i] < target:
            fib_m = fib_m_minus_1
            fib_m_minus_1 = fib_m_minus_2
            fib_m_minus_2 = fib_m - fib_m_minus_1
            offset = i
        else:
            fib_m = fib_m_minus_2
            fib_m_minus_1 -= fib_m_minus_2
            fib_m_minus_2 = fib_m - fib_m_minus_1

    if fib_m_minus_1 and arr[offset + 1] == target:
        return offset + 1
    return -1

# Example usage:
arr = [10, 20, 30, 40, 50]
target = 30
print(fibonacci_search(arr, target))  # Output: 2
```

---

These are some of the most commonly used searching algorithms in Python. Each algorithm has its own use case based on the input data and its properties (e.g., sorted or unsorted arrays).

Here's an advanced Python interview exercise focusing on **tuples** and their various features. This exercise tests understanding of how to manipulate tuples, their immutability, and how they can be used in various scenarios.

---

### Problem: Tuple Manipulation Exercise

You are given a list of tuples, where each tuple contains two elements: a string and an integer. Your task is to:

1. **Sort** the list of tuples by the integer value in each tuple.
2. **Filter** out tuples where the integer is less than or equal to 5.
3. Convert the resulting tuples into a dictionary where the string in each tuple is the key and the integer is the value.
4. Extract the first two values from the dictionary and store them as a tuple of tuples. (Make sure the result is still in tuple format).
5. Return the final result as a tuple of tuples containing the first two elements of the filtered dictionary.

**Example:**

```python
data = [("apple", 7), ("banana", 3), ("orange", 9), ("pear", 2), ("grape", 8)]
```

1. **Sort** the list by the integer values.
2. **Filter** out elements with integers less than or equal to 5.
3. Convert it to a dictionary.
4. Extract the first two elements as tuples and return the result.

---

### Solution

```python
def advanced_tuple_manipulation(data):
    # Step 1: Sort the list of tuples by the integer values (second element in each tuple)
    sorted_data = sorted(data, key=lambda x: x[1])
    
    # Step 2: Filter out tuples where the integer is less than or equal to 5
    filtered_data = [item for item in sorted_data if item[1] > 5]
    
    # Step 3: Convert the filtered list of tuples into a dictionary
    data_dict = dict(filtered_data)
    
    # Step 4: Extract the first two elements from the dictionary as a tuple of tuples
    # Convert the dictionary items to a list of tuples and get the first two elements
    dict_items = list(data_dict.items())
    first_two_items = tuple(dict_items[:2])
    
    return first_two_items

# Test the function
data = [("apple", 7), ("banana", 3), ("orange", 9), ("pear", 2), ("grape", 8)]
result = advanced_tuple_manipulation(data)
print(result)
```

### Explanation

1. **Sorting**: We sort the list of tuples based on the second item (integer) using the `sorted()` function with a custom sorting key (`lambda x: x[1]`).

2. **Filtering**: We use a list comprehension to filter out tuples where the second element (integer) is less than or equal to 5.

3. **Dictionary Conversion**: After filtering, we convert the list of tuples into a dictionary using the `dict()` function. The first element of each tuple becomes the key, and the second element becomes the value.

4. **Extracting Tuples**: We then convert the dictionary back to a list of items and select the first two elements. These are returned as a tuple of tuples.

---

### Expected Output

For the input:

```python
data = [("apple", 7), ("banana", 3), ("orange", 9), ("pear", 2), ("grape", 8)]
```

The steps result in:

1. Sorted: `[("banana", 3), ("pear", 2), ("apple", 7), ("grape", 8), ("orange", 9)]`
2. Filtered (integer > 5): `[("apple", 7), ("grape", 8), ("orange", 9)]`
3. Converted to dictionary: `{"apple": 7, "grape": 8, "orange": 9}`
4. First two items: `(("apple", 7), ("grape", 8))`

The final output will be:

```python
(('apple', 7), ('grape', 8))
```

### Complexity Analysis

- **Time Complexity**:
  - Sorting takes \(O(n \log n)\), where \(n\) is the number of tuples.
  - Filtering takes \(O(n)\).
  - Converting the list to a dictionary takes \(O(n)\).
  - Extracting the first two elements from the dictionary takes constant time, \(O(1)\).
  
Thus, the overall time complexity is \(O(n \log n)\), dominated by the sorting step.

- **Space Complexity**:
  - The space complexity is \(O(n)\) due to the space required for the filtered list, sorted list, and dictionary.

In Python, **set operations** allow you to perform various mathematical operations on sets, such as union, intersection, difference, and symmetric difference. Here's a breakdown of all the common set operations with examples:

### 1. **Union (`|` or `union()`)**

The union of two sets returns a new set containing all the unique elements from both sets.

#### Example

```python
set1 = {1, 2, 3}
set2 = {3, 4, 5}

union_result = set1 | set2  # Using the `|` operator
# or
union_result_method = set1.union(set2)  # Using the `union()` method

print(union_result)          # Output: {1, 2, 3, 4, 5}
print(union_result_method)   # Output: {1, 2, 3, 4, 5}
```

### 2. **Intersection (`&` or `intersection()`)**

The intersection of two sets returns a new set with elements that are common to both sets.

#### Example

```python
set1 = {1, 2, 3}
set2 = {3, 4, 5}

intersection_result = set1 & set2  # Using the `&` operator
# or
intersection_result_method = set1.intersection(set2)  # Using the `intersection()` method

print(intersection_result)          # Output: {3}
print(intersection_result_method)   # Output: {3}
```

### 3. **Difference (`-` or `difference()`)**

The difference between two sets returns a new set containing elements that are in the first set but not in the second set.

#### Example

```python
set1 = {1, 2, 3}
set2 = {3, 4, 5}

difference_result = set1 - set2  # Using the `-` operator
# or
difference_result_method = set1.difference(set2)  # Using the `difference()` method

print(difference_result)          # Output: {1, 2}
print(difference_result_method)   # Output: {1, 2}
```

### 4. **Symmetric Difference (`^` or `symmetric_difference()`)**

The symmetric difference of two sets returns a new set containing elements that are in either of the sets but not in both.

#### Example

```python
set1 = {1, 2, 3}
set2 = {3, 4, 5}

symmetric_difference_result = set1 ^ set2  # Using the `^` operator
# or
symmetric_difference_result_method = set1.symmetric_difference(set2)  # Using the `symmetric_difference()` method

print(symmetric_difference_result)          # Output: {1, 2, 4, 5}
print(symmetric_difference_result_method)   # Output: {1, 2, 4, 5}
```

### 5. **Subset (`<=` or `issubset()`)**

A set is a subset of another if all elements of the first set are contained within the second set.

#### Example

```python
set1 = {1, 2}
set2 = {1, 2, 3, 4}

is_subset = set1 <= set2  # Using the `<=` operator
# or
is_subset_method = set1.issubset(set2)  # Using the `issubset()` method

print(is_subset)          # Output: True
print(is_subset_method)   # Output: True
```

### 6. **Superset (`>=` or `issuperset()`)**

A set is a superset of another if it contains all elements of the second set.

#### Example

```python
set1 = {1, 2, 3, 4}
set2 = {2, 3}

is_superset = set1 >= set2  # Using the `>=` operator
# or
is_superset_method = set1.issuperset(set2)  # Using the `issuperset()` method

print(is_superset)          # Output: True
print(is_superset_method)   # Output: True
```

### 7. **Disjoint Sets (`isdisjoint()`)**

Two sets are disjoint if they have no common elements.

#### Example

```python
set1 = {1, 2, 3}
set2 = {4, 5, 6}

are_disjoint = set1.isdisjoint(set2)  # Using the `isdisjoint()` method

print(are_disjoint)  # Output: True
```

### 8. **Adding and Removing Elements**

You can add or remove elements from a set using methods like `add()`, `remove()`, `discard()`, and `pop()`.

#### Example

```python
set1 = {1, 2, 3}

# Adding an element
set1.add(4)  # Adds 4 to the set
print(set1)  # Output: {1, 2, 3, 4}

# Removing an element (raises KeyError if not found)
set1.remove(2)
print(set1)  # Output: {1, 3, 4}

# Removing an element (doesn't raise error if not found)
set1.discard(5)  # No error even though 5 is not in the set
print(set1)  # Output: {1, 3, 4}

# Popping an element (removes and returns an arbitrary element)
popped_element = set1.pop()
print(popped_element)  # Output: arbitrary element (e.g., 1)
print(set1)  # Output: remaining elements
```

### 9. **Clear All Elements (`clear()`)**

This method removes all elements from a set.

#### Example

```python
set1 = {1, 2, 3}
set1.clear()  # Clears the set
print(set1)  # Output: set()
```

### Summary

- **Union**: Combines elements of both sets.
- **Intersection**: Finds common elements between two sets.
- **Difference**: Finds elements that are in the first set but not in the second.
- **Symmetric Difference**: Finds elements that are in either set, but not in both.
- **Subset**: Checks if one set is a subset of another.
- **Superset**: Checks if one set is a superset of another.
- **Disjoint**: Checks if two sets have no common elements.
- **Modification methods**: Add, remove, or clear elements in a set.

These operations are powerful tools for manipulating sets in Python!

In Python, class attributes and instance attributes are two types of attributes that can be associated with a class. Let's break down the difference with an example.

### Class Attribute

- A class attribute is shared by all instances of the class.
- It is defined inside the class but outside of any instance methods.
- Class attributes are accessed by ClassName.attribute or by instances.

### Instance Attribute

- An instance attribute is specific to an individual instance of the class.
- It is defined inside the constructor (**init**) or in any method that modifies it for a particular instance.
- Instance attributes are accessed using the instance of the class, like instance.attribute.

Here's an example demonstrating the difference:

```python
class Dog:
    # Class attribute (shared by all instances of Dog)
    species = "Canis familiaris"

    def __init__(self, name, age):
        # Instance attributes (specific to each instance)
        self.name = name
        self.age = age

    def bark(self):
        print(f"{self.name} says Woof!")

# Create two instances of Dog
dog1 = Dog("Buddy", 5)
dog2 = Dog("Lucy", 3)

# Access class attribute
print(dog1.species)  # "Canis familiaris" (same for all dogs)
print(dog2.species)  # "Canis familiaris" (same for all dogs)

# Access instance attributes
print(dog1.name)  # "Buddy"
print(dog2.name)  # "Lucy"
print(dog1.age)   # 5
print(dog2.age)   # 3
```

# Modify class attribute via class name

```python
Dog.species = "Canis lupus familiaris"
```

# After modifying, class attribute is updated for all instances

```python
print(dog1.species)  # "Canis lupus familiaris"
print(dog2.species)  # "Canis lupus familiaris"
```

# Modify instance attribute via instance

```python
dog1.age = 6
print(dog1.age)  # 6 (only changed for dog1)
print(dog2.age)  # 3 (remains unchanged for dog2)
```

### Explanation

1. Class Attribute (species):
   - The species attribute is defined at the class level, outside any methods. This means that it's shared by all instances of Dog.
   - You can access it through the class (Dog.species) or through an instance (dog1.species), but it remains the same for all instances unless explicitly modified at the class level.

2. Instance Attributes (name and age):
   - The name and age attributes are defined as **init** method. These are specific to each instance and can be different for each Dog.
   - When you modify dog1.age, it only affects dog1, not dog2, because each instance has its own copy of the instance attributes.

### Key Takeaways

- Class attributes are shared across all instances of the class.
- Instance attributes are unique to each instance and are typically initializedinit__init__ method.

```python
class MethodTypes:

    name = "Ragnar"

    def instanceMethod(self):
        # Creates an instance atribute through keyword self
        self.lastname = "Lothbrock"
        print(self.name)
        print(self.lastname)

    @classmethod
    def classMethod(cls):
        # Access a class atribute through keyword cls
        print(MethodTypes.name)
        cls.name = "Lagertha"
        print(cls.name)
        

    @staticmethod
    def staticMethod():
        print("This is a static method")

# Creates an instance of the class
m = MethodTypes()
# Calls instance method
m.instanceMethod()

print()

MethodTypes.classMethod()
MethodTypes.staticMethod()
```

Monkey patching in Python refers to the practice of modifying or extending existing classes or modules at runtime. This is done by dynamically changing or adding behavior to classes, methods, or functions, typically without modifying the original source code. While monkey patching can be a powerful technique, it should be used cautiously as it can lead to unexpected side effects, maintenance challenges, and harder-to-debug code.

Here are some examples to understand monkey patching in Python:

### 1. **Monkey Patching a Method**

You can modify the behavior of an existing method in a class by reassigning it to a new function.

```python
class MyClass:
    def greet(self):
        print("Hello!")

# Instance of MyClass
obj = MyClass()
obj.greet()  # Output: Hello!

# Monkey patching the greet method
def new_greet(self):
    print("Hi there!")

MyClass.greet = new_greet  # Patch the method

obj.greet()  # Output: Hi there!
```

In this example, the method `greet` of `MyClass` is replaced at runtime with `new_greet`. Now, any instance of `MyClass` will use the patched `greet` method.

### 2. **Monkey Patching a Function**

You can also monkey patch functions. This is often done with functions from external libraries.

```python
import math

# Original function
print(math.sqrt(16))  # Output: 4.0

# Monkey patching math.sqrt
def custom_sqrt(x):
    print("Calculating square root...")
    return x ** 0.5

math.sqrt = custom_sqrt  # Patch the function

print(math.sqrt(16))  # Output: Calculating square root... 4.0
```

Here, we've replaced the `sqrt` function from the `math` module with our own `custom_sqrt` function. Now, whenever `math.sqrt` is called, it will invoke the custom function.

### 3. **Monkey Patching to Add New Functionality**

Monkey patching can be used to add new functionality to a class or module.

```python
class MyClass:
    def method_one(self):
        print("Original method")

# Instance of MyClass
obj = MyClass()
obj.method_one()  # Output: Original method

# Adding a new method using monkey patching
def method_two(self):
    print("New method added")

MyClass.method_two = method_two  # Add new method

obj.method_two()  # Output: New method added
```

In this case, we've added a new method `method_two` to `MyClass` via monkey patching.

### Use Cases for Monkey Patching

- **Testing**: Monkey patching is often used in unit tests to mock or stub methods and functions, especially when dealing with external services, APIs, or third-party libraries.
  
- **Bug Fixes**: In some rare cases, monkey patching can be used to fix bugs or issues in libraries without having to wait for an official update or patch.

- **Changing Behavior**: You can use monkey patching to modify or extend the behavior of third-party libraries or built-in classes without modifying the original code.

### Drawbacks of Monkey Patching

1. **Risk of Breaking Code**: If not carefully managed, it can lead to unexpected behavior or break other parts of the codebase.
2. **Difficult Debugging**: Since monkey patching changes the behavior of existing code, debugging becomes harder as the modifications are done at runtime.
3. **Unintended Side Effects**: Other parts of the program (or other libraries) might depend on the original behavior, and monkey patching could cause conflicts or bugs.

### Best Practices

- **Use with Caution**: Only use monkey patching when necessary and if no other solution is viable.
- **Limit Scope**: Apply monkey patches as locally as possible, ideally only within the scope of tests or limited areas of the application.
- **Document Changes**: Clearly document any monkey patches to ensure that other developers understand why the patch was applied and what it does.

In summary, while monkey patching can be useful in certain scenarios, it's important to weigh the pros and cons and consider whether it's the best solution for your problem.
