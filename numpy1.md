### NumPy Basics to Advanced Concepts

NumPy is a powerful library for numerical computing in Python. It provides support for arrays, matrices, and a wide range of mathematical functions to manipulate them. Below is a detailed guide to NumPy concepts from basics to advanced with examples and outputs.

---

## Table of Contents:

1. **Installation and Import**
2. **Arrays in NumPy**
   - Creating Arrays
   - Array Indexing and Slicing
   - Array Attributes
   - Reshaping Arrays
   - Array Concatenation and Splitting
3. **Mathematical Operations**
   - Element-wise Operations
   - Broadcasting
   - Universal Functions (ufuncs)
4. **Linear Algebra**
   - Dot Product
   - Matrix Multiplication
   - Eigenvalues and Eigenvectors
   - Solving Linear Systems
5. **Random Module**
   - Random Numbers Generation
   - Random Sampling
   - Random Distributions
6. **Advanced NumPy Concepts**
   - Fancy Indexing
   - Masking
   - Structured Arrays

---

## 1. Installation and Import

To install NumPy, use pip:

```bash
pip install numpy
```

Then, import NumPy in your script:

```python
import numpy as np
```

---

## 2. Arrays in NumPy

### a. Creating Arrays

**1.1. Creating a 1D array**

```python
arr1 = np.array([1, 2, 3, 4])
print(arr1)
```
**Output:**
```
[1 2 3 4]
```

**1.2. Creating a 2D array**

```python
arr2 = np.array([[1, 2], [3, 4]])
print(arr2)
```
**Output:**
```
[[1 2]
 [3 4]]
```

**1.3. Creating arrays with specific values**

- **Zeros array:**

```python
arr_zeros = np.zeros((2, 3))
print(arr_zeros)
```
**Output:**
```
[[0. 0. 0.]
 [0. 0. 0.]]
```

- **Ones array:**

```python
arr_ones = np.ones((2, 2))
print(arr_ones)
```
**Output:**
```
[[1. 1.]
 [1. 1.]]
```

- **Identity matrix:**

```python
arr_identity = np.eye(3)
print(arr_identity)
```
**Output:**
```
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
```

### b. Array Indexing and Slicing

**2.1. Indexing a 1D array**

```python
arr = np.array([10, 20, 30, 40])
print(arr[2])  # Accessing the third element
```
**Output:**
```
30
```

**2.2. Indexing a 2D array**

```python
arr_2d = np.array([[1, 2], [3, 4]])
print(arr_2d[1, 1])  # Accessing element at row 1, column 1
```
**Output:**
```
4
```

**2.3. Slicing a 1D array**

```python
arr = np.array([10, 20, 30, 40, 50])
print(arr[1:4])  # Slicing from index 1 to 3
```
**Output:**
```
[20 30 40]
```

**2.4. Slicing a 2D array**

```python
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr_2d[1:, :2])  # Slicing all rows from 1 onward and first two columns
```
**Output:**
```
[[4 5]
 [7 8]]
```

### c. Array Attributes

**3.1. Shape, Size, and Dimension**

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.shape)  # Shape of the array
print(arr.size)   # Total number of elements
print(arr.ndim)   # Number of dimensions
```
**Output:**
```
(2, 3)
6
2
```

### d. Reshaping Arrays

**4.1. Reshaping a 1D array to 2D**

```python
arr = np.array([1, 2, 3, 4, 5, 6])
reshaped = arr.reshape(2, 3)  # Reshape to 2x3 matrix
print(reshaped)
```
**Output:**
```
[[1 2 3]
 [4 5 6]]
```

### e. Array Concatenation and Splitting

**5.1. Concatenating arrays**

```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
concat = np.concatenate((arr1, arr2))
print(concat)
```
**Output:**
```
[1 2 3 4 5 6]
```

**5.2. Splitting an array**

```python
arr = np.array([1, 2, 3, 4, 5, 6])
split = np.split(arr, 3)  # Split into 3 parts
print(split)
```
**Output:**
```
[array([1, 2]), array([3, 4]), array([5, 6])]
```

---

## 3. Mathematical Operations

### a. Element-wise Operations

**1.1. Addition, subtraction, multiplication**

```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
print(arr1 + arr2)  # Element-wise addition
print(arr1 * arr2)  # Element-wise multiplication
```
**Output:**
```
[5 7 9]
[ 4 10 18]
```

### b. Broadcasting

Broadcasting allows operations between arrays of different shapes.

**2.1. Broadcasting a scalar to an array**

```python
arr = np.array([1, 2, 3])
result = arr + 5
print(result)
```
**Output:**
```
[6 7 8]
```

**2.2. Broadcasting with arrays of different shapes**

```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([[1], [2], [3]])
result = arr1 + arr2
print(result)
```
**Output:**
```
[[2 3 4]
 [3 4 5]
 [4 5 6]]
```

### c. Universal Functions (ufuncs)

NumPy provides many mathematical functions that can be applied element-wise on arrays.

**3.1. Square root and exponential functions**

```python
arr = np.array([1, 4, 9, 16])
print(np.sqrt(arr))  # Element-wise square root
print(np.exp(arr))   # Element-wise exponential
```
**Output:**
```
[1. 2. 3. 4.]
[2.71828183e+00 5.45981500e+01 8.10308393e+03 8.88611052e+06]
```

---

## 4. Linear Algebra

### a. Dot Product

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
dot_product = np.dot(a, b)
print(dot_product)
```
**Output:**
```
32
```

### b. Matrix Multiplication

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
result = np.matmul(A, B)
print(result)
```
**Output:**
```
[[19 22]
 [43 50]]
```

### c. Eigenvalues and Eigenvectors

```python
matrix = np.array([[4, -2], [1,  1]])
eigenvalues, eigenvectors = np.linalg.eig(matrix)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:", eigenvectors)
```
**Output:**
```
Eigenvalues: [3. 2.]
Eigenvectors: [[ 0.89442719 -0.70710678]
 [ 0.4472136   0.70710678]]
```

### d. Solving Linear Systems

```python
A = np.array([[3, 2], [1, 4]])
b = np.array([7, 10])
x = np.linalg.solve(A, b)
print(x)
```
**Output:**
```
[1. 2.]
```

---

## 5. Random Module

### a. Random Numbers Generation

```python
random_arr = np.random.random((2, 2))  # 2x2 array with random floats between 0 and 1
print(random_arr)
```
**Output:**
```
[[0.67744969 0.93116765]
 [0.58255092 0.41431771]]


```

### b. Random Sampling

```python
sample = np.random.choice([10, 20, 30, 40], size=3)  # Randomly sample 3 elements
print(sample)
```
**Output:**
```
[20 40 10]
```

---

## 6. Advanced NumPy Concepts

### a. Fancy Indexing

```python
arr = np.array([0, 1, 2, 3, 4, 5])
print(arr[[1, 3, 4]])  # Accessing elements at indices 1, 3, and 4
```
**Output:**
```
[1 3 4]
```

### b. Masking

```python
arr = np.array([10, 20, 30, 40, 50])
mask = arr > 30
print(arr[mask])  # Elements greater than 30
```
**Output:**
```
[40 50]
```

### c. Structured Arrays

Structured arrays allow you to define compound data types.

```python
dtype = [('name', 'U10'), ('age', 'i4')]
data = np.array([('Alice', 25), ('Bob', 30)], dtype=dtype)
print(data)
```
**Output:**
```
[('Alice', 25) ('Bob', 30)]
```

---

### Summary

This guide covered the fundamental to advanced concepts in NumPy with examples and outputs, including array creation, manipulation, linear algebra, random number generation, and advanced indexing techniques. NumPy is incredibly powerful for numerical and scientific computations and learning it opens the door to many machine learning, data science, and deep learning tasks.



---



