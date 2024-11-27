Here are some advanced Python interview questions that could be asked to assess deep knowledge of the language:

### 1. What are metaclasses in Python, and how do they work?
   - Answer: A metaclass in Python is a class that defines how other classes are created. It allows you to modify class creation behavior, such as adding or modifying attributes, methods, or class-level properties. Metaclasses can be defined by inheriting from type, and you can override methods like __new__ and __init__ to control class instantiation.

### 2. Explain the Global Interpreter Lock (GIL) and its impact on multi-threading in Python.
   - Answer: The GIL is a mechanism that prevents multiple native threads from executing Python bytecodes at once in CPython (the default implementation). This means that, in CPython, even though you can have multiple threads, only one thread can execute Python bytecode at a time. This limits concurrency in CPU-bound programs but does not affect I/O-bound operations significantly.

### 3. **What is the difference between deepcopy and copy in Python?**
 Answerer**: The copy() function creates a shallow copy of an object, meaning that if the original object contains other objects (like lists inside a list), the references to those inner objects are copied, not the actual inner objects. The deepcopy() function, on the other hand, creates a completely new copy of the object along with all objects nested within it.

### 4. What are Python decorators and how do they work?k?**
 Answerer**: Decorators are a way to modify or enhance functions or methods without changing their source code. A decorator is a function that takes another function as an argument and returns a new function. Decorators are commonly used for logging, access control, memoization, etc.

   Example:
```  
   def decorator_function(original_function):
       def wrapper_function():
           print("Wrapper executed this before {}".format(original_function.__name__))
           return original_function()
       return wrapper_function
```
   
### 5. **Explain Python's garbage collection mechanism.m.**
 Answerer**: Python uses reference counting and garbage collection (GC) to manage memory. When an object's reference count drops to zero, it is automatically deallocated. Additionally, Python uses a cyclic garbage collector to handle circular references. The garbage collection process is handled by the gc module.

### 6. **What are the differences between @staticmethod and @classmethod in Python?AnswerAnswer**:
     - @staticmethod: It is used to define a method that doesn't depend on the instance or the class. It doesn’t take self or cls as the first parameter.
     - @classmethod: It is used to define a method that receives the class itself as the first argument (represented as cls), and it can access class-level attributes or methods.
### 7. What are Python generators and how do they work? work?AnswerAnswer**: A generator is a function that returns an iterator, and it yields values one at a time using the yield keyword. Unlike normal functions that return a value and exit, generators maintain their state between calls and can be resumed. They are memory-efficient as they yield items lazily.

   Example:
```  
   def count_up_to(max):
       count = 1
       while count <= max:
           yield count
           count += 1
```
   
### 8. **What is theslotsof __slots__ in PytAnswer- slots*: __slots__ is used to limit the attributes of a class to a predefined set, which can save memory by preventing the creation of dictult __dict__ for each instance. This is especially useful when dealing with a large number of instances and known attributes.

   Example:
```  
   class MyClass:
       __slots__ = ['name', 'age']
       def __init__(self, name, age):
           self.name = name
           self.age = age
```

### 9. How does Python handle multiple inheritance and method resolution order (MRO)?
   - Answer: Python uses the C3 linearization algorithm (also known as C3 superclass linearization) to determine the method resolution order (MRO) when dealing with multiple inheritance. The MRO determines the order in which base classes are searched when a method is called. You can view the MRO of a class using the mro() method or __mro__ attribute.

   Example:
```  
   class A: pass
   class B(A): pass
   class C(A): pass
   class D(B, C): pass
   print(D.mro())
```
   
### 10. What is a context manager in Python?
   - Answer: A context manager is a Python object that allows you to manage resources, such as files or database connections, using the with statement. It defines methods __enter__() and __exit__() to allocate and release resources.

   Example:
```  
   class MyContextManager:
       def __enter__(self):
           print("Entering the context")
           return self
       def __exit__(self, exc_type, exc_value, traceback):
           print("Exiting the context")
   with MyContextManager():
       print("Inside the context")
```
   
### 11. Explain the concept of 'duck typing' in Python.
   - Answer: Duck typing is a concept where the type or class of an object is determined by its behavior (methods and properties) rather than its inheritance or explicit type. If an object behaves like a certain type, it can be treated as that type, regardless of its actual class.

   Example:
```  
   class Bird:
       def fly(self):
           print("Flying")

   class Airplane:
       def fly(self):
           print("Flying like an airplane")

   def take_off(flyable):
       flyable.fly()  # Works if the object has a fly method
```
   
### 12. **What is the difference between is and == in Python?**
 Answerer**: The == operator checks if the values of two objects are the same, while the is operator checks if two references point to the same object in memory. 

   Example:
```  
   a = [1, 2, 3]
   b = [1, 2, 3]
   print(a == b)  # True (values are equal)
   print(a is b)  # False (they are different objects)
```
   
### 13. **How does Python's asyncio module work, and what is the difference between async and await?AnswerAnswer**: The asyncio module is used for writing asynchronous code. It allows you to run I/O-bound operations without blocking the execution of other tasks. The async keyword defines an asynchronous function, while await is used to pause the function's execution until the awaited result is available.

   Example:
```  
   import asyncio

   async def fetch_data():
       await asyncio.sleep(2)  # Simulates an I/O operation
       return "Data fetched"

   async def main():
       data = await fetch_data()
       print(data)

   asyncio.run(main())
```
   
### 14. **What are f-strings in Python, and why are they preferred over other string formatting methAnswer- **Answer**: f-strings (formatted string literals) are a concise and efficient way to embed expressions inside string literals, introduced in Python 3.6. They are preferred over older methods (like % formatting or str.format()) because they are more readable and generally faster.

   Example:
```  
   name = "John"
   age = 30
   print(f"My name is {name} and I am {age} years old.")
```
   
### 15. **How can you optimize the performance of PythAnswer
   - **Answer**: Some ways to optimize Python code include:
     - Using built-in functions and libraries, as they are often optimized.
     - Avoiding global variables and using local variables where possible.
     - Using list comprehensions or generator expressions instead of loops for better performance.
     - Profiling the code using cProfile and focusing on optimizing the bottlenecks.
     - Using libraries like NumPy for mathematical operations and multiprocessing for parallelism in CPU-bound tasks.


Here are more advanced Python interview questions, focusing on deeper concepts, design patterns, and performance considerations:

### 1. **What is the difference between __del__ and __exit__ in Python?**
 Answerer*del __del__ and __exit__ are used for cleanup purposes, but they are used in different contedel   - __del__ is a destructor method called when an object is about to be destroyed. It is part of Python’s garbage collection mechanism and is not guaranteed to be called immediately after an object is no longer referenexit  - __exit__ is used in the context of a context manager and is part of the with statement. It ensures that the resources are cleaned up when exiting the context, even if an exception occurs.

### 2. What is method resolution order (MRO) and how does it work in multiple inheritance?e?**
 Answerer**: Method Resolution Order (MRO) is the order in which Python searches for methods and attributes in the class hierarchy when using multiple inheritance. Python uses the C3 Linearization algorithm to determine the order, which is the order in which classes are considered for method lookup.
   
   You can view a class's MRO using the mro() method or __mro__ attribute:
```  
   class A: pass
   class B(A): pass
   class C(A): pass
   class D(B, C): pass
   print(D.mro())  # Shows the MRO for class D
```
   
### 3. **What is the purpose of the abc module in Python?AnswerAnswer**: The abc (Abstract Base Classes) module in Python provides a mechanism for defining abstract classes. An abstract class is one that cannot be instantiated directly and is intended to be subclassed. The module allows you to define abstract methods, which must be implemented by any subclass.
   
   Example:
```  
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
   
### 4. **Explain Python's yield from expressAnswer- **Answer**: The yield from expression simplifies delegating part of a generator’s operations to another iterable or generator. It allows a generator to yield all values from another iterable or generator without explicitly looping through it.
   
   Example:
```  
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
   
### 5. **What is the difnew between __new__ and __init__ inAnswer
   -newer**:
     - __new__ is the method responsible for creating a new instance of a class. initlled before __init__ and is typically used in metaclasses or when subclassing immutable types like int, str, or tuple.
     - __init__ is called after the object newted (i.e., after __new__) and is used to initialize the object's attributes.

   Example:
```  
   class MyClass:
       def __new__(cls):
           print("Creating instance")
           return super().__new__(cls)

       def __init__(self):
           print("Initializing instance")

   obj = MyClass()  # Output: Creating instance
                    #         Initializing instance
```
   
### 6. **What is a "closureAnswern?**
   - **Answer**: A closure is a function that "remembers" the environment in which it was created, even after that environment has finished execution. This means that the function has access to variables that were in scope when the function was defined, even if they are no longer in scope when the function is called.

   Example:
```  
   def outer(x):
       def inner(y):
           return x + y
       return inner

   closure = outer(10)
   print(closure(5))  # Output: 15
```

### 7. **What is the @property decorator, and how is it used?**
 Answerer**: The @property decorator is used to define a method as a read-only attribute. It allows you to define a method that can be accessed like an attribute, without explicitly calling the method.
   
   Example:
```  
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
   
###How can you improve the performance of Python code involving large datasets?s?**
 Answerer**: Some strategies to optimize performance include:
     - NumPyPy**Pandasas** for large numerical and data-processing tasks.
     - built-in functionsns** and avoid unnecessary loops.
     - multiprocessingng**concurrent.futureses** for CPU-bound tasks to take advantage of multiple processors.
     - generator expressionsns** instead of list comprehensions for memory efficiency.
     - memoizationon**cachingng** (via functools.lru_cache or custom caching) to avoid redundant computations.
     - Profile code using the cProfile module to identify and optimize bottlenecks.

###What are some ways to handle exceptions in Python?n?**
 Answerer**: In Python, exceptions are handled using try, except, else, and finally blocks.
     - try: Contains code that may raise an exception.
     - except: Catches and handles the exception.
     - else: Runs if no exception was raised in the try block.
     - finally: Executes code after the try block, regardless of whether an exception was raised or not.

   Example:
```  
   try:
       x = 1 / 0
   except ZeroDivisionError:
       print("Cannot divide by zero.")
   else:
       print("No errors.")
   finally:
       print("This runs no matter what.")
```
   
### 10. **What are Python's contextlib and contextmanager?AnswerAnswer**: contextlib is a standard library module that provides utilities for creating and working with context managers. The contextmanager decorator is used to define a simple context manager using a generator function.

   Example:
```  
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
   
What are Python descriptors, and how do they work? work?AnswerAnswer**: A descriptor is an object attribute with "binding behavior" that customizes how an attribute is accessed or modified. Descriptors implement any of the methods __get__, __set__, or __delete__ to define how attribute access is managed.

   Example:
```  
   class Descriptor:
       def __get__(self, instance, owner):
           return 'Attribute accessed'

   class MyClass:
       attr = Descriptor()

   obj = MyClass()
   print(obj.attr)  # 'Attribute accessed'
```
   
### 12. **What is the purpose of the functools module, and what are some common functions it contaAnswer- **Answer**: The functools module provides higher-order functions that operate on other functions or callable objects. Some common functions include:
     - lru_cache: Caches function results to improve performance for expensive functions.
     - partial: Creates a new function by fixing some arguments of an existing function.
     - reduce: Applies a binary function cumulatively to a sequence.

   Example:
```  
   from functools import partial

   def power(base, exponent):
       return base ** exponent

   square = partial(power, exponent=2)
   print(square(4))  # Output: 16
```

### 13. What are some ways to optimize memory usage in Python?
   - Answer: Techniques for optimizing memory usage include:
     - Using generators instead of lists when working with large datasets.
     - Using **__slots__** in classes to avoid the overhead of instance dictionaries.
     - Avoiding the creation of unnecessary copies of data (e.g., use in-place operations).
     - Using **array** or **numpy** arrays for numerical data instead of lists.
     -memoryviewmemoryview** objects to work with large binary data efficientlHow does Python handle namespaces and variable scope?ble scAnswer- **Answer**: Python uses namespaces to manage variable scope. Each function, module, and class has its own

Here are some advanced practical Python interview questions that can assess both your understanding of complex Python concepts and your ability to solve real-world problems using Python:

### 1. Implement a Decorator that Measures the Execution Time of a Function
   - Problem: Write a decorator that measures how long a function takes to execute.
   - Solution:
```  
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
   def long_running_task():
       time.sleep(2)

   long_running_task()  # Output: Execution time: 2.000xx seconds
```
   
### 2. **Implement a Custom Iterator Class**
 Problemem**: Implement a custom iterator that returns squares of numbers from 1 to n.
 Solutionon**:
```  
   class SquareIterator:
       def __init__(self, n):
           self.n = n
           self.current = 1

       def __iter__(self):
           return self

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
   
###Design and Implement a Cache with LRU (Least Recently Used) Evictionon**
 Problemem**: Implement an LRU cache using Python's collections.OrderedDict or your own implementation.
 Solutionon**:
```  
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
###Write a Python Function to Perform Deep Copying of Nested Dictionarieses**
 Problemem**: Write a function that performs a deep copy of a nested dictionary.
 Solutionon**:
 
```  
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
   
###Implement a Function to Flatten a Nested List (of arbitrary depth)h)**
 Problemem**: Write a function that takes a nested list (of arbitrary depth) and flattens it into a single list.
 Solutionon**:
```  
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
   
###Write a Python Function to Find the Longest Substring Without Repeating Charactersrs**
 Problemem**: Given a string, write a function that finds the length of the longest substring without repeating characters.
 Solutionon**:
```
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
