import numpy as np

def generate_x_y(distribution='normal', no_features=1, sample_size=1000, low=0, high=1):
    # Create a random generator object
    rng = np.random.default_rng(seed=0)

    if distribution == 'normal':
        feature_data = rng.standard_normal(size=(sample_size, no_features))
        target = rng.standard_normal(sample_size)

    else:
        #Assuming uniform distribution is wanted
        feature_data = rng.uniform(low=low, high=high, size=(sample_size, no_features))
        target = rng.uniform(low, high, sample_size)
    
    return np.array(feature_data), target

if __name__ == '__main__':
    X,Y = generate_x_y(no_features=5,sample_size=10000)
    print(X.shape)
    print(type(X))
    print(Y.shape)
    print(type(Y))