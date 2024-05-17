import numpy as np
from generate_class_data import generate_x, generate_y
from sklearn.datasets import make_classification
from scipy import stats


#Define Parameters
NO_FEATURES = 5
NO_SAMPLES = 1000
MAX_DEPTH = 10
MIN_SAMPLES_SPLIT = 50
MAX_LEAF_NODES = 32

#Define a Node class
class Node:

    def __init__(self, feature_index, feature_threshold, left_child=None, 
                 right_child=None, predicted_value=None, gain=None, depth=None):
        
        self.feature_index = feature_index
        self.feature_threshold = feature_threshold
        self.left_child = left_child
        self.right_child = right_child
        self.gain = gain
        self.predicted_value = predicted_value
        self.depth = depth


# Define helper functions
def get_gini_impurity(y_labels):
    classes, counts = np.unique(y_labels, return_counts=True)
    n = len(y_labels)

    gini_val = 1
    for i in range(len(classes)):
        gini_val = gini_val - ((counts[i]/n)**2)

    return gini_val

#get training data
# X = generate_x(no_features=NO_FEATURES, sample_size=NO_SAMPLES)
# class_config_dict = {
#     'classification': 'binary',
#     'sample_size': NO_SAMPLES,
#     'probability_of_success': 0.7,
# }
# Y = generate_y(class_config_dict)
X, Y = make_classification(n_samples=NO_SAMPLES, n_features=NO_FEATURES, n_classes=2, random_state=100)

print(X.shape)
print(Y.shape)

def get_split(data_x, data_y):
    n_samples = len(data_y)
    n_features = data_x.shape[1]
    parent_gini = get_gini_impurity(data_y)

    max_gain = 0
    feature_index = None
    feature_threshold = None

    for j in range(n_features):
        feature_x = data_x[: , j]

        thresholds = np.unique(feature_x)
        for threshold in thresholds:
            left_indices, right_indices = feature_x < threshold, feature_x >= threshold

            if any(left_indices) and any(right_indices):
                gini_left, gini_right = get_gini_impurity(data_y[left_indices]), get_gini_impurity(data_y[left_indices])
                gini_children = ( len(left_indices) * gini_left + len(right_indices) * gini_right ) / n_samples

                gain = parent_gini - gini_children
                if gain > max_gain:
                    feature_index = j
                    feature_threshold = threshold
                    max_gain = gain

    return feature_index, feature_threshold, max_gain


def build_recursive_tree(data_X, data_Y, depth):
    print(f"At Recursive Depth of {depth}")

    #Assuming that this is a leaf node
    #Make node
    current_node = Node(feature_index=None,
                        feature_threshold=None,
                        left_child=None,
                        right_child=None,
                        predicted_value=stats.mode(data_Y)[0],
                        gain=None,
                        depth=depth)

    #Return if any hyperparameters satisfied or pure node
    if len(np.unique(data_Y)) == 1 or depth==MAX_DEPTH or len(data_X) <= MIN_SAMPLES_SPLIT:   
        print("Leaf Node Found")     
        return current_node
    
    #find split
    feature_index, feature_threshold, gain = get_split(data_x=data_X, data_y=data_Y)

    if feature_index is None:
        return current_node
    
    else: #Split found out
        #make the preidiction to None and add other features
        current_node.predicted_value = None
        current_node.gain = gain
        current_node.feature_index = feature_index
        current_node.feature_threshold = feature_threshold

        #find left and right datasets
        left_indices, right_indices = data_X[: , feature_index] < feature_threshold, \
                            data_X[: , feature_index] >= feature_threshold
        
        left_x, left_y = data_X[left_indices], data_Y[left_indices]
        right_x, right_y = data_X[right_indices], data_Y[right_indices]

        #make left and right subtrees
        current_node.left = build_recursive_tree(left_x, left_y, depth=depth+1)
        current_node.right = build_recursive_tree(right_x, right_y, depth=depth+1)

        return current_node
    
tree = build_recursive_tree(X, Y, depth=0)

def predict_for_new_sample(x_data):
    current_node = tree

    while current_node.predicted_value is None:
        feature_index, feature_threshold = current_node.feature_index,  current_node.feature_threshold

        if x_data[feature_index] < feature_threshold:
            #go left
            current_node = current_node.left

        else:
            #go right
            current_node = current_node.right

    return current_node.predicted_value

y_pred = []
for i in range(NO_SAMPLES):
    x_sample = X[i, :]
    y_pred.append(predict_for_new_sample(x_sample))

#Actual classes
print(np.unique(Y, return_counts=True))
print()
print(np.unique(y_pred, return_counts=True))
        



    




