import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# Dataset specification
# 7 niumerical
# 3 categorical
# 1 target variable with 5 different classes

X_num, Y = make_classification(
    n_samples=1000, n_features=7, n_informative=4, n_classes=5
)
print(pd.Series(Y).value_counts())  # roughly 200 samples for each class

# add categorical variables
cat1 = np.random.choice(["Hindi", "English", "Bengali"], 1000).reshape(-1, 1)
cat2 = np.random.choice(["Yes", "No"], 1000).reshape(-1, 1)
cat3 = np.random.choice(["0-10", "10-25", "25-50", "50+"], 1000).reshape(-1, 1)

# Combine into same array
X = np.concatenate([X_num, cat1, cat2, cat3], axis=1)

# convert to pandas dataframe
df = pd.DataFrame(X)
df["target"] = Y

print(df.shape)  # 1000 X 11
