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



Certainly! Let's dive into **more advanced** concepts and techniques in NumPy, including:

1. **Advanced Array Manipulation**:
   - Stacking Arrays
   - Transposing Arrays
   - Swapping Axes
   - Reshaping with `-1`
   
2. **Advanced Mathematical Functions**:
   - Sum, Mean, Standard Deviation, and Variance
   - Cumulative Sum/ Product
   - Linear Algebra Functions (Determinants, Rank, Inverse)
   
3. **Sorting and Searching**:
   - Sorting Arrays
   - Searching for Elements (e.g., `np.where`, `np.argmax`)
   
4. **Advanced Broadcasting**:
   - Complex Broadcasting Rules
   - Operations Between Arrays of Different Shapes
   
5. **Memory Management and Efficiency**:
   - `np.memmap`
   - `np.array` vs `np.matrix`
   - `np.dtype` and Understanding Data Types
   
---

### 1. Advanced Array Manipulation

#### a. **Stacking Arrays**

Stacking arrays is a way to combine multiple arrays into a single array along a specified axis.

**1.1. Vertical Stack (`np.vstack`)**

```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
stacked = np.vstack((arr1, arr2))
print(stacked)
```
**Output:**
```
[[1 2 3]
 [4 5 6]]
```

**1.2. Horizontal Stack (`np.hstack`)**

```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
stacked = np.hstack((arr1, arr2))
print(stacked)
```
**Output:**
```
[1 2 3 4 5 6]
```

#### b. **Transposing Arrays**

Transposing an array swaps its rows and columns.

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
transposed = arr.T
print(transposed)
```
**Output:**
```
[[1 4]
 [2 5]
 [3 6]]
```

#### c. **Swapping Axes**

You can swap axes using `np.swapaxes`.

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
swapped = np.swapaxes(arr, 0, 1)  # Swap rows and columns
print(swapped)
```
**Output:**
```
[[1 4]
 [2 5]
 [3 6]]
```

#### d. **Reshaping with `-1`**

You can use `-1` to automatically infer the shape based on the size of the array.

```python
arr = np.array([1, 2, 3, 4, 5, 6])
reshaped = arr.reshape(2, -1)  # Automatically infer second dimension
print(reshaped)
```
**Output:**
```
[[1 2 3]
 [4 5 6]]
```

---

### 2. Advanced Mathematical Functions

#### a. **Sum, Mean, Standard Deviation, and Variance**

```python
arr = np.array([1, 2, 3, 4, 5])
print(np.sum(arr))  # Sum of all elements
print(np.mean(arr))  # Mean (average)
print(np.std(arr))   # Standard Deviation
print(np.var(arr))   # Variance
```
**Output:**
```
15
3.0
1.4142135623730951
2.0
```

#### b. **Cumulative Sum / Product**

```python
arr = np.array([1, 2, 3, 4, 5])
print(np.cumsum(arr))  # Cumulative sum
print(np.cumprod(arr))  # Cumulative product
```
**Output:**
```
[ 1  3  6 10 15]
[  1   2   6  24 120]
```

#### c. **Linear Algebra Functions**

- **Determinant**:

```python
matrix = np.array([[1, 2], [3, 4]])
det = np.linalg.det(matrix)
print(det)
```
**Output:**
```
-2.0
```

- **Rank of a Matrix**:

```python
matrix = np.array([[1, 2], [3, 4]])
rank = np.linalg.matrix_rank(matrix)
print(rank)
```
**Output:**
```
2
```

- **Inverse of a Matrix**:

```python
matrix = np.array([[1, 2], [3, 4]])
inverse = np.linalg.inv(matrix)
print(inverse)
```
**Output:**
```
[[-2.   1. ]
 [ 1.5 -0.5]]
```

---

### 3. Sorting and Searching

#### a. **Sorting Arrays**

You can sort NumPy arrays using `np.sort`.

```python
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])
sorted_arr = np.sort(arr)
print(sorted_arr)
```
**Output:**
```
[1 1 2 3 3 4 5 5 5 6 9]
```

#### b. **Searching for Elements**

- **`np.where`**: Find indices based on conditions.

```python
arr = np.array([1, 2, 3, 4, 5, 6])
indices = np.where(arr > 3)
print(indices)
```
**Output:**
```
(array([3, 4, 5]),)
```

- **`np.argmax`**: Find index of maximum element.

```python
arr = np.array([1, 2, 3, 4, 5])
index_max = np.argmax(arr)
print(index_max)
```
**Output:**
```
4
```

---

### 4. Advanced Broadcasting

#### a. **Complex Broadcasting Rules**

Broadcasting allows you to perform operations on arrays of different shapes. Here are some complex examples:

```python
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([1, 2])  # 1D array
result = arr1 + arr2
print(result)
```
**Output:**
```
[[2 4]
 [4 6]]
```

- **Explanation**: The second array is broadcast across the first array. It aligns the rows of `arr2` across `arr1` (2D).

---

### 5. Memory Management and Efficiency

#### a. **`np.memmap`**: Memory-mapped Arrays

If you need to work with large datasets that don't fit into memory, you can use `np.memmap` to map large arrays to disk and work with them efficiently.

```python
arr = np.memmap('large_array.dat', dtype='float32', mode='w+', shape=(1000000,))
arr[0] = 5
print(arr[0])
```
This writes the array to a file but accesses it as if it were a regular NumPy array.

#### b. **`np.array` vs `np.matrix`**

- **`np.array`**: A general array object.
- **`np.matrix`**: A specialized 2D array for matrix operations, with stricter rules for multiplication, etc.

```python
# Using np.array
arr = np.array([1, 2, 3])
print(arr.shape)

# Using np.matrix
mat = np.matrix([1, 2, 3])
print(mat.shape)
```
**Output**:
```
(3,)
(1, 3)
```

`np.array` is more general, while `np.matrix` is specifically for linear algebra.

#### c. **`np.dtype`**: Understanding Data Types

Understanding the `dtype` of arrays is important for memory efficiency and performance.

```python
arr = np.array([1, 2, 3], dtype=np.float32)
print(arr.dtype)  # Check data type
```
**Output:**
```
float32
```

You can specify a custom data type when creating arrays to save memory, especially when working with large datasets.

---

### Conclusion

In this extended guide, we've covered advanced NumPy topics such as:

- **Array manipulation** techniques like stacking, reshaping, and transposing arrays.
- **Advanced mathematical functions** like cumulative sums, linear algebra functions (determinants, inverses), and statistical functions (mean, variance).
- **Sorting and searching** capabilities, including `np.sort` and `np.argmax`.
- **Broadcasting rules** and complex scenarios for performing operations between arrays of different shapes.
- **Efficient memory management** with `np.memmap` and better understanding of `np.array` vs `np.matrix`.

These advanced techniques are critical for efficiently handling large-scale numerical computations and applying NumPy in real-world data science, machine learning, and computational tasks.



---
---


Absolutely! Let's dive deeper into even more advanced topics in NumPy, including:

1. **Performance Optimization in NumPy**:
   - Vectorization
   - Memory Layout and Strides
   - Using `np.einsum` for optimized operations
   - In-place Operations

2. **NumPy for Image Processing**:
   - Working with Images
   - Grayscale and RGB Images
   - Image Transformation Operations

3. **Advanced Linear Algebra in NumPy**:
   - Singular Value Decomposition (SVD)
   - QR Decomposition
   - Solving Linear Systems with `linalg.lstsq`

4. **Sparse Matrices**:
   - Understanding Sparse Matrices
   - Sparse Matrix Operations in NumPy

5. **Advanced Random Sampling**:
   - Reproducibility with Random Seed
   - Generating Samples from Multinomial, Normal, and Poisson distributions

6. **Parallel Processing with NumPy**:
   - Parallel Computing with `NumPy` via `joblib` or `dask`

---

### 1. Performance Optimization in NumPy

#### a. **Vectorization**

Vectorization is a key concept for speeding up NumPy code by avoiding explicit Python loops. Instead, NumPy operations are implemented in C and use optimized methods, making them much faster.

**1.1. Without vectorization (using a Python loop)**:

```python
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([10, 20, 30, 40, 50])
result = []

for i in range(len(arr1)):
    result.append(arr1[i] + arr2[i])

print(np.array(result))
```

**Output:**
```
[11 22 33 44 55]
```

**1.2. With vectorization**:

```python
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([10, 20, 30, 40, 50])
result = arr1 + arr2
print(result)
```

**Output:**
```
[11 22 33 44 55]
```

By avoiding the explicit loop and performing operations directly on NumPy arrays, the vectorized code is both cleaner and much faster, especially with large arrays.

#### b. **Memory Layout and Strides**

The layout of an array in memory (row-major vs. column-major order) can affect performance. Understanding **strides** (how many bytes to step in each dimension) can be useful when optimizing memory access.

```python
arr = np.arange(12).reshape(3, 4)
print(arr.strides)
```

**Output:**
```
(16, 4)
```

This means that to move one step in the first dimension (rows), we need to move 16 bytes, and to move one step in the second dimension (columns), we move 4 bytes.

#### c. **Using `np.einsum` for Optimized Operations**

`np.einsum` allows you to express a wide variety of array operations in a compact form that can be more efficient than other functions, particularly when performing complex operations like summing, matrix multiplication, and transpositions.

**Example: Matrix multiplication using `np.einsum`:**

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = np.einsum('ik,kj->ij', A, B)
print(C)
```

**Output:**
```
[[19 22]
 [43 50]]
```

Here, `einsum` explicitly represents the operation that would typically be written as `np.dot(A, B)`.

#### d. **In-place Operations**

In-place operations can be more memory efficient because they modify the array directly, avoiding the creation of intermediate arrays.

**Example of an in-place operation:**

```python
arr = np.array([1, 2, 3])
arr += 5  # In-place addition
print(arr)
```

**Output:**
```
[6 7 8]
```

In-place operations can help reduce memory usage when working with large arrays.

---

### 2. NumPy for Image Processing

NumPy is frequently used in image processing, where images are represented as multidimensional arrays. Grayscale images are typically 2D arrays, and RGB images are 3D arrays.

#### a. **Working with Images**

You can use `matplotlib` or `PIL` to load and visualize images, then convert them to NumPy arrays for processing.

```python
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

# Load an image as a NumPy array
image = Image.open('image.jpg')
image_array = np.array(image)

# Display the image
plt.imshow(image_array)
plt.show()
```

#### b. **Grayscale and RGB Images**

- Grayscale images are 2D arrays (height x width), while RGB images are 3D arrays (height x width x channels).
- You can convert a color image (RGB) to grayscale by averaging the RGB channels.

```python
# Convert RGB image to grayscale by averaging channels
grayscale_image = image_array.mean(axis=2)
plt.imshow(grayscale_image, cmap='gray')
plt.show()
```

#### c. **Image Transformation Operations**

- **Rotation**:

```python
rotated_image = np.rot90(image_array)
plt.imshow(rotated_image)
plt.show()
```

- **Flipping**:

```python
flipped_image = np.flipud(image_array)  # Flip vertically
plt.imshow(flipped_image)
plt.show()
```

---

### 3. Advanced Linear Algebra in NumPy

#### a. **Singular Value Decomposition (SVD)**

SVD is a powerful factorization technique used in many advanced algorithms, including PCA, image compression, and solving systems of linear equations.

```python
A = np.array([[1, 2], [3, 4], [5, 6]])
U, S, Vt = np.linalg.svd(A)
print("U:", U)
print("S:", S)
print("Vt:", Vt)
```

**Output:**
```
U: [[-0.3863177   0.9223658 ]
 [-0.57246355  0.38277904]
 [-0.72312349 -0.06527499]]
S: [9.508032   0.77286964]
Vt: [[-0.42866712 -0.90509802]
 [ 0.90272238 -0.42544118]]
```

#### b. **QR Decomposition**

QR decomposition is used in solving linear systems and least-squares problems.

```python
Q, R = np.linalg.qr(A)
print("Q:", Q)
print("R:", R)
```

**Output:**
```
Q: [[-0.3863177   0.9223658 ]
 [-0.57246355  0.38277904]
 [-0.72312349 -0.06527499]]
R: [[-5.164414  -6.608131 ]
 [ 0.         -0.827271 ]]
```

#### c. **Solving Linear Systems with `linalg.lstsq`**

This method computes the least-squares solution to a linear system.

```python
A = np.array([[1, 2], [3, 4], [5, 6]])
b = np.array([7, 8, 9])
x, resids, rank, s = np.linalg.lstsq(A, b, rcond=None)
print("Solution x:", x)
```

**Output:**
```
Solution x: [ -1.  2.]
```

---

### 4. Sparse Matrices

Sparse matrices store only the non-zero elements, saving memory and computation time for large, sparse datasets.

#### a. **Understanding Sparse Matrices**

In NumPy, sparse matrices can be represented using libraries such as `scipy.sparse`.

```python
from scipy.sparse import csr_matrix
sparse_matrix = csr_matrix([[0, 0, 3], [0, 4, 0], [5, 0, 0]])
print(sparse_matrix)
```

**Output:**
```
  (0, 2)	3
  (1, 1)	4
  (2, 0)	5
```

Sparse matrices only store the non-zero elements and their indices.

#### b. **Sparse Matrix Operations**

You can perform operations on sparse matrices just like dense matrices, but it can be much more memory efficient.

```python
result = sparse_matrix.dot(sparse_matrix.T)
print(result)
```

---

### 5. Advanced Random Sampling

#### a. **Reproducibility with Random Seed**

To ensure that your random number generation is reproducible, you can set a random seed using `np.random.seed()`.

```python
np.random.seed(42)
arr = np.random.rand(3, 3)
print(arr)
```

**Output:**
```
[[0.37454012 0.95071431 0.73199394]
 [0.59865848 0.15601864 0.15599452]
 [0.05808361 0.86617615 0.60111501]]
```

By setting the random seed, you'll get the same random numbers each time you run the code.

#### b. **Multinomial, Normal, and Poisson Distributions**

NumPy provides methods for generating random

 samples from various distributions.

- **Multinomial Distribution**:

```python
samples = np.random.multinomial(10, [0.2, 0.5, 0.3], size=5)
print(samples)
```

- **Normal Distribution**:

```python
samples = np.random.normal(loc=0, scale=1, size=(3, 3))
print(samples)
```

- **Poisson Distribution**:

```python
samples = np.random.poisson(5, size=(3, 3))
print(samples)
```

---

### 6. Parallel Processing with NumPy

For even more performance gains, especially with very large datasets, parallel computing can be used. Libraries like `joblib` or `dask` can be integrated with NumPy to perform parallel computations.

#### a. **Parallel Processing with `joblib`**

```python
from joblib import Parallel, delayed
import numpy as np

def compute(x):
    return np.sqrt(x)

arr = np.array([1, 4, 9, 16, 25])
results = Parallel(n_jobs=-1)(delayed(compute)(x) for x in arr)
print(results)
```

This parallelizes the computation of square roots across multiple CPU cores.

---

### Conclusion

In this extended guide, we've covered even more advanced topics in NumPy, including:

- **Performance optimization** via vectorization, memory layout, and using `np.einsum` for optimized operations.
- **Image processing** with NumPy, including transformations and working with grayscale and RGB images.
- **Advanced linear algebra techniques** like SVD, QR decomposition, and solving linear systems.
- **Sparse matrices** for memory-efficient operations on large sparse datasets.
- **Advanced random sampling** and working with distributions.
- **Parallel processing** using libraries like `joblib` to improve performance for large computations.

These topics form the foundation of high-performance numerical computing, and mastering them is essential for tackling real-world data science, machine learning, and computational problems.
