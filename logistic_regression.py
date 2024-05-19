# The following link explains well the loss function, cost and gradients
# for logistic regression
# Link - https://www.baeldung.com/cs/gradient-descent-logistic-regression
# Basically it turns out that the gradient update rule is the same for 
# Linear Regression and Logistic Regression
# The only thing that changes is the sigmoid function applied on top of linear transformation of features

import numpy as np
from generate_class_data import generate_x, generate_y
from sklearn.datasets import make_classification


#Parameters
NO_FEATURES = 10
NO_SAMPLES = 5000
BATCH_SIZE = 128
NO_EPOCHS = 100
LEARNING_RATE = 0.1
P_SUCCESS = 0.8

#Training Data
# X = generate_x(no_features=NO_FEATURES, sample_size=NO_SAMPLES)

# binary_class_configs = {
#     'sample_size': NO_SAMPLES,
#     'classification': 'binary',
#     'probability_of_success': P_SUCCESS,
# }

# Y = generate_y(class_dict=binary_class_configs)
X,Y = make_classification(n_classes=2, n_features=NO_FEATURES, n_samples=NO_SAMPLES)

print(X.shape)
print(Y.shape)

#Intialize weights and biases
rng = np.random.default_rng(seed=0)
W = rng.standard_normal(size=(NO_FEATURES, 1))
B = rng.standard_normal(size=(1,1))

#Define Sigmoid Function
def sigmoid(z):
    return (1 + np.exp(-z))**(-1)

#pred function
def get_predicted_y(weight, bias, x_matrix):
    return sigmoid( np.matmul(x_matrix,weight) + bias )

#cost function
def get_cost(y_pred, y_actual):
    #Cost Part 1 
    cost1 = np.sum( y_actual * np.log(y_pred) )  # Elementwise multiplication
    #Cost Part 2
    cost2 = np.sum( (1-y_actual) * np.log(1 - y_pred) )  #Elementwise multiplication

    total_cost = (cost1 + cost2) * (-1/ len(y_pred))

    return total_cost

#get gradient functions
#using the property that gradients are same as linear regression
def get_gradient_weights(x_matrix, y_pred, y_actual):
    error = y_actual - y_pred
    gradient = np.matmul(x_matrix.T, error) * (-1/ len(y_pred))

    return gradient

def get_gradient_bias(y_pred, y_actual):
    return np.sum((y_actual - y_pred)) * (-1/ len(y_pred))

#Update Weights and biases
def update_bias(bias, bias_gradient, learning_rate =LEARNING_RATE):
    return bias - (learning_rate * bias_gradient)

def update_weights(weight, weight_gradient, learning_rate=LEARNING_RATE):
    return weight - (learning_rate * weight_gradient)


if __name__ == '__main__':
    NO_BATCHES = NO_SAMPLES // BATCH_SIZE
    for epoch in range(NO_EPOCHS):
        for i in range(NO_BATCHES):
            x = X[i : i + BATCH_SIZE]
            y_actual = Y[i : i + BATCH_SIZE].reshape(-1, 1)

            y_pred = get_predicted_y(W, B, x)
            cost = get_cost(y_pred, y_actual)
            gradient_weights = get_gradient_weights(x, y_pred, y_actual)
            gradient_bias = get_gradient_bias(y_pred, y_actual)
            print(cost)
            W = update_weights(W, gradient_weights)
            B = update_bias(B, gradient_bias)
