import numpy as np
from sklearn.datasets import make_regression
import pandas as pd

# Dataset dimensions
# 7 Numerical Features
# 3 categorical features
# 1 target variable

X_num, Y = make_regression(n_samples=1000, n_features=7)
print(X_num.shape)  # 1000, 7
print(Y.shape)  # 1000

# Build Categorical Features using numoy choice
cat1 = np.random.choice(["Mumbai", "Delhi", "Hyd", "Bang"], 1000).reshape(
    -1, 1
)  # convert it into a 2-d array
cat2 = np.random.choice(["Yes", "No"], 1000).reshape(-1, 1)
cat3 = np.random.choice(["1000", "1000-2900", "5000+"], 1000).reshape(-1, 1)

# Combine with X_num
X = np.concatenate([X_num, cat1, cat2, cat3], axis=1)
print(X.shape)  # 1000 X 10

# Convert to dataframe
train = pd.DataFrame(X)
print(train.shape)  # 1000 X 10

train["target"] = Y
print(train.shape)  # 1000 X 11
