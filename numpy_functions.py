import numpy as np

a = [2,2,3,3,3]
print( np.unique(a) ) # Returns an araay containing only the unique values
print(np.unique(a, return_counts=True)) # Returns 2 arrays, 1st containing unique values and 2nd containing the counts of those values
# this function would be very useful in classification problems, where you need to calculate fraction of all classes

#Let's create a 2-D array
# a 2D array is n 1-D arrays stacked together
_2d_array1 = np.array([ 
    [2,5],
    [4,6]
]) 
_2d_array2 = np.array([
    [7,8, 3],
    [5,6,9]
])
print(_2d_array1.shape) #- (2,2)
print(_2d_array2.shape) #- (2,3)
#let's see indexing
#Select 1 row
print(_2d_array2[1, :]) # selects all elements in 1st row - returns 1D array
#Select column 0 to 1
print(_2d_array2[:, 0:2]) # returns a 2D array

#if we do subsetting
print(_2d_array1 < 5) # returns a 2-D array with True/False for each position

#Select all elements from column2 that are greater than 5
column_values = _2d_array2[:, 2]
filter = column_values > 5
print(column_values[ filter ]) # this returns a 1-D array, all values selected from column 2

#Select all elements from column0 and column2 that are greater than 5
column_values = _2d_array2[:, [0,2]]
filter = column_values > 3
print(column_values[ filter ]) #this still returns a 1-D array , by first selecting 2 values from column 0 and 1 value from column 2

#Creating blank matrices of given shape
print(np.ones([2,3,2]))
print(np.zeros([2,3,1]))

#concatenating 2 matrices
#vertically
print( np.vstack(([1,2], [3,4]) ))
#horizontally
print(np.hstack(([1,2], [3,4]) ))

#Reshape a 1D array to 2D array, by adding 1 column
print(np.array([1,2,3]).reshape(-1, 1))

#Reshape a 1D array to 2D array, by adding 1 row
print(np.array([1,2,3]).reshape(1, -1))

#Reshape a 2D array to 3D array, by adding a batch size dimension of 1
print(_2d_array1.reshape(1, _2d_array1.shape[0], _2d_array1.shape[1]))

#Element wise multiplication of 2 matrices, both matrices should be of same shape
A = np.array([[1,3],[3,4]]) 
B = np.array([[1,3],[3,4]])
C = A * B
print(C)

#Multiplication of 2 matrices of different shapes
A = np.array([[1,3],[3,4]]) 
B = np.array([[1,3,5],[3,4,2]])
C = np.matmul(A, B)
print(C)

#np.dot is another function for mat multiplication, but its better to use matmul

#Sampling from a distribution using numpy
# case 1: sampling from a discrete distribution with pmf given
X = [2,3,5]
prob = [0.2, 0.3, 0.5]
samples = np.random.choice(X, size=100, p=prob)
print(np.unique(samples, return_counts=True))

# case 2: sample from a binomial distribution
# for binomial distibution, no_trials = 10, p_success = 0.6
samples = np.random.binomial(n=10, p=0.6, size=100)
print(np.unique(samples, return_counts=True))

# case 3: sample from a normal distribution
# mean = 5, std = 2
samples = np.random.normal(loc=5, scale=2, size=100)
# print(np.unique(samples, return_counts=True))

# case 4: sample from a standard normal distribution
samples = np.random.standard_normal(size=100)
# print(np.unique(samples, return_counts=True))

"""
Numpy Axis
obj.sum(axis = 0) --> SUMS the COLUMNS (in 2-D Matrix)
So OUTPUT = 1 ROW * NO OF COLUMNS

obj.sum(axis = 1) --> SUMS THE ROWS (in 2-D Matrix)
So OUTPUT = NO OF ROWS * 1 COLUMN

Please note that this might be different from concat, vstack, hstack operations
"""



