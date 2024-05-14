import numpy as np
rng = np.random.default_rng(seed=0)

def generate_x(distribution='normal', no_features=1, sample_size=1000, low=0, high=1):
    if distribution == 'normal':
        feature_data = rng.standard_normal(size=(sample_size, no_features))
    else:
        feature_data = rng.uniform(low=low, high=high, size=(sample_size, no_features))

    return feature_data

def generate_y(class_dict):
    sample_size = class_dict['sample_size']
    classification_type = class_dict['classification']
    if classification_type == 'binary':
        p = class_dict['probability_of_success']
        target = rng.binomial(n=1, p=p, size=(sample_size, 1))

    else:
        p_classes = class_dict['probability_classwise']
        target = rng.multinomial(n=1, pvals=p_classes, size=sample_size)
        target = np.array(np.argmax(target, axis=1))

    return target


if __name__ == '__main__':
    X = generate_x(no_features=5, sample_size=5)
    config_dict = {
        'sample_size': 5,
        'classification': 'multinomial',
        'probability_classwise': [0.2, 0.3, 0.15, 0.15, 0.1, 0.05],
    }
    Y = generate_y(class_dict=config_dict)
    print(X.shape)
    print(type(X))
    print(Y.shape)
    print(Y)
    print(type(Y))
  