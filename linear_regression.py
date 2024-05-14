from generate_reg_data import generate_x_y
import numpy as np

#Parameters
NO_FEATURES = 10
NO_SAMPLES = 5000
BATCH_SIZE = 128
NO_EPOCHS = 5
LEARNING_RATE = 0.05

#Training Data
X, Y = generate_x_y(no_features=NO_FEATURES, sample_size=NO_SAMPLES)

print(X.shape)
print(Y.shape)

#We need to initalize Weight Matrix and Bias Matrix
# No of weights should be equal to no of features
# No of biases should be equal to 1, for 1 linear regression model
# If X = (Batch Size, No features), then after multiplying with weight matrix, 
# we should be left with an array of size (Batch Size * 1)
# To achieve this, W = (No features, 1) --> Columnar Matrix
# Then the bias should be added, which is (1,1) matrix

#Intialize weights and biases
rng = np.random.default_rng(seed=0)
W = rng.standard_normal(size=(NO_FEATURES, 1))
B = rng.standard_normal(size=(1,1))

#pred function
def get_predicted_y(weight, bias, x_matrix):
    return np.matmul(x_matrix,weight) + bias

#cost function
def get_cost(y_pred, y_actual):
    return np.mean( (y_pred - y_actual)**2 ) / 2

#gradient function
#The gradient formula for linear regression is given by (X.T * error)
# error is given by y_pred - y_actual
# If X is of shape (NO_SAMPLES , NO_FEATURES), Then X.T = (NO_FEATURES, NO_SAMPLES)
# Error is of the shape (NO_SAMPLES, 1)
# On multiplying these 2 matrices, we'll get a matrix = (NO_FEATURES, 1)
# This is the same size as Weight Matrix of Parameter Matrix
# Then gradient will be of the same shape as no of Weight Parameters
# The gradient is multiplied by a factor (2/m) - This comes out from the differentiation of loss function
def get_gradient_weights(x_matrix, y_pred, y_actual):
    error = y_pred - y_actual
    gradient = np.matmul(x_matrix.T, error) * (2/ len(y_pred))

    return gradient

#For bias, terms the gradient is equal to the error term (y_pred, y_actual)
# Since bias is of dim 1, we need to sum the error terms of all predictions to get final gradient
def get_gradient_bias(y_pred, y_actual):
    return np.sum((y_pred - y_actual)) * (2/ len(y_pred))

def update_bias(bias, bias_gradient, learning_rate =LEARNING_RATE):
    return bias - (learning_rate * bias_gradient)

def update_weights(weight, weight_gradient, learning_rate=LEARNING_RATE):
    return weight - (learning_rate * weight_gradient)

if __name__ == '__main__':
    NO_BATCHES = NO_SAMPLES // BATCH_SIZE
    for epoch in range(NO_EPOCHS):
        for i in range(NO_BATCHES):
            x = X[i : i + BATCH_SIZE]
            y_actual = Y[i : i + BATCH_SIZE]

            y_pred = get_predicted_y(W, B, x)
            cost = get_cost(y_pred, y_actual)
            gradient_weights = get_gradient_weights(x, y_pred, y_actual)
            gradient_bias = get_gradient_bias(y_pred, y_actual)
            print(cost)
            W = update_weights(W, gradient_weights)
            B = update_bias(B, gradient_bias)
            

