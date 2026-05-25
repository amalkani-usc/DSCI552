import numpy as np

# (a) 2D Python list -> Numpy array
py_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
arr = np.array(py_list)
print("Array:\n", arr)

# (b) Array attributes
print("\nndim:", arr.ndim)        # number of dimensions
print("shape:", arr.shape)        # (rows, cols)
print("size:", arr.size)          # total number of elements
print("dtype:", arr.dtype)        # data type of elements
print("itemsize:", arr.itemsize)  # bytes per element
print("data:", arr.data)          # memory buffer address

# (c) reshape and flatten
reshaped = arr.reshape(1, 9)      # 3x3 -> 1x9
print("\nReshaped (1x9):\n", reshaped)

reshaped2 = arr.reshape(9, 1)     # 3x3 -> 9x1
print("Reshaped (9x1):\n", reshaped2)

flattened = arr.flatten()         # always returns a copy as 1D
print("Flattened:", flattened)

# (d) Slicing
arr1d = np.array([10, 20, 30, 40, 50])
print("\n1D slicing:")
print("arr1d[1:4]:", arr1d[1:4])  # [20, 30, 40]
print("arr1d[:3]:", arr1d[:3])    # [10, 20, 30]
print("arr1d[::2]:", arr1d[::2])  # [10, 30, 50] (every other)

arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
print("\n2D slicing:")
print("arr2d[0, :]:", arr2d[0, :])      # first row
print("arr2d[:, 1]:", arr2d[:, 1])      # second column
print("arr2d[0:2, 1:3]:\n", arr2d[0:2, 1:3])  # submatrix

# (e) Array operations
a = np.array([-4, 1, -2, 3, 5])
b = np.array([2, 3, 1, 4, 2])

print("\nOperations on a =", a)
print("argmin:", a.argmin())       # index of min
print("argmax:", a.argmax())       # index of max
print("min:", a.min())
print("max:", a.max())
print("mean:", a.mean())
print("sum:", a.sum())
print("std:", a.std())
print("dot(a,b):", np.dot(a, b))   # dot product
print("square:", np.square(a))
print("sqrt(abs):", np.sqrt(np.abs(a)))
print("abs:", np.abs(a))
print("exp:", np.exp(a))
print("sign:", np.sign(a))         # -1, 0, or 1
print("mod(a,3):", np.mod(a, 3))   # remainder

# (f) Array creation methods
print("\narange(0,10,2):", np.arange(0, 10, 2))       # [0,2,4,6,8]
print("ones(3,3):\n", np.ones((3, 3)))
print("zeros(2,4):\n", np.zeros((2, 4)))
print("eye(3):\n", np.eye(3))                          # identity matrix
print("linspace(0,1,5):", np.linspace(0, 1, 5))        # 5 evenly spaced from 0 to 1

a1 = np.array([[1, 2], [3, 4]])
a2 = np.array([[5, 6]])
print("concatenate (axis=0):\n", np.concatenate((a1, a2), axis=0))  # stack rows
print("concatenate (axis=1):\n", np.concatenate((a1, a2.T), axis=1))  # stack cols