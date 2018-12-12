import numpy as np


def get_class_exemplar(draws, class_index, n_exemplars=1, best=True, labels=None):
    """
    Get representative examples of a class for a BNN classifier

    :param class_index: int
    :param n_exemplars: int
    :param best: True if you want exemplars. False if you want the least representative examples
    :param labels: If provided and best is False, return the worst exemplars from the actual class
    :return:
    """
    if labels is not None:
        # We only want the "predictions" where the ground truth is the label in question
        draws = draws[:, labels[:, class_index] == 1]
    draws = np.argsort(draws[class_index, :])
    if best:
        draws = draws[::-1]
    examples = draws[:n_exemplars]
    return examples


def get_dense_network_shapes(n_layers, hidden_size, n_features, n_outputs):
    """
    Helper function to generate the input/output shapes for the layers of a densely connected network
    :param n_layers: Number of hidden layers in the network
    :param hidden_size: How many hidden neurons to use
    :param n_features: Number of features in the original input
    :param n_outputs: Output size/number of target variables
    :return:
    """
    shapes = {'input': (n_features, hidden_size),
              'hidden': [],
              'output': (hidden_size * (n_layers+1) + n_features, n_outputs)}
    for i in range(n_layers):
        shapes['hidden'].append((hidden_size * (i + 1) + n_features, hidden_size))
    return shapes


def flat_to_hierarchical(x, y, groups):
    Xs, Ys = [], []
    min_len = np.inf
    for i in np.unique(groups):
        X = np.array(x[groups == i]).astype(float)
        Y = np.array(y[groups == i]).astype(float)
        if len(X) < min_len:
            min_len = len(X)
        Xs.append(X)
        Ys.append(Y)

    Xss, Yss = [], []
    for _x, _y in zip(Xs, Ys):
        _x = _x[:min_len, :]
        _y = _y[:min_len]

        Xss.append(_x)
        Yss.append(_y)

    x = np.stack(Xss)
    y = np.stack(Yss)
    return x, y
