import numpy as np


# Generate a random array between 10 and 50
print(np.random.randint(10, 50, 5))

print(
    np.random.rand(2, 3)
)  # creates a 2x3 matrix with random float numbers between 0 and 1

# Multiply 2 matrices
a = np.array([[1, 2, 5], [3, 4, 7]])  # 2X3 matrix
b = np.array([[5, 6], [7, 8], [9, 10]])  # 3X2 matrix
print(np.dot(a, b))

# Element wise multiplication of two matrices
a = [[1, 2], [3, 4]]
b = [[5, 6], [7, 8]]
print(np.multiply(a, b))

# Transpose a Matrix
a = np.array([[1, 2, 3], [4, 5, 6]])
print(np.transpose(a))

# Inverse of a matrix
a = np.array([[1, 2], [3, 4]])
print(np.linalg.inv(a))

# Sample from a normal distribution
print(np.random.normal(0, 1, 5))

# Sample from a uniform distribution
print(np.random.uniform(0, 1, 5))

# Sample from a distribution with given probabilities
# p(2) = 0.5, p(3) = 0.3, p(4) = 0.2
print(np.random.choice([2, 3, 4], 12, p=[0.5, 0.3, 0.2]))  # 12 samples
