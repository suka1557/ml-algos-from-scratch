from generate_reg_data import generate_x_y
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

#Parameters
NO_FEATURES = 2
NO_SAMPLES = 1000
BATCH_SIZE = 128
NO_EPOCHS = 10
LEARNING_RATE = 0.05



#Training Data
# X, Y = generate_x_y(no_features=NO_FEATURES, sample_size=NO_SAMPLES)
X, Y = make_regression(n_features=NO_FEATURES, n_samples=NO_SAMPLES, random_state=100)

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
    predictions = np.matmul(x_matrix,weight) + bias
    return predictions

#cost function
def get_cost(y_pred, y_actual):
    return np.sum( (y_actual - y_pred)**2 ) / len(y_actual)

#gradient function
#The gradient formula for linear regression is given by (X.T * error)
# error is given by y_pred - y_actual
# If X is of shape (NO_SAMPLES , NO_FEATURES), Then X.T = (NO_FEATURES, NO_SAMPLES)
# Error is of the shape (NO_SAMPLES, 1)
# On multiplying these 2 matrices, we'll get a matrix = (NO_FEATURES, 1)
# This is the same size as Weight Matrix of Parameter Matrix
# Then gradient will be of the same shape as no of Weight Parameters
# The gradient is multiplied by a factor (1/m) - This comes out from the differentiation of loss function
def get_gradient_weights(x_matrix, y_pred, y_actual):
    error = y_actual - y_pred
    gradient = np.matmul(np.transpose(x_matrix), error) 
    gradient = gradient * (-2/len(y_actual))
    return gradient

#For bias, terms the gradient is equal to the error term (y_pred, y_actual)
# Since bias is of dim 1, we need to sum the error terms of all predictions to get final gradient
def get_gradient_bias(y_pred, y_actual):
    bias_gradient =  np.sum((y_actual - y_pred)) * (-2/len(y_actual)) 
    return bias_gradient

def update_bias(bias, bias_gradient, learning_rate =LEARNING_RATE):
    return bias - (learning_rate * bias_gradient)

def update_weights(weight, weight_gradient, learning_rate=LEARNING_RATE):
    return weight - (learning_rate * weight_gradient)

if __name__ == '__main__':
    print()
    print(W.shape)
    print(B.shape)
    NO_BATCHES = NO_SAMPLES // BATCH_SIZE
    for epoch in range(NO_EPOCHS):
        for i in range(NO_BATCHES):
            x = X[i : i + BATCH_SIZE, :]
            y_actual = Y[i : i + BATCH_SIZE].reshape(-1, 1)
            #This reshaping is needed, as without it while calculating error, numpy erongly broadcasts it
            # making the error matrix a square matrix
            y_pred = get_predicted_y(W, B, x)
            cost = get_cost(y_pred, y_actual)
            gradient_weights = get_gradient_weights(x, y_pred, y_actual)
            gradient_bias = get_gradient_bias(y_pred, y_actual)
            W = update_weights(W, gradient_weights)
            B = update_bias(B, gradient_bias)
        
    
    print(cost)
    print(W.shape)
    print(B.shape)
    #results for sklearn model
    print()
    model = LinearRegression()
    model.fit(X,Y)
    y_pred = model.predict(X)
    print(get_cost(y_pred, Y))

    #There is some mistake in Linear Reg Model which needs to be fixed
            

