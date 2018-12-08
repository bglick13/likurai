def get_dense_network_shapes(n_layers, hidden_size, n_features, n_outputs):
    """
    Helper function to generate the input/output shapes for the layers of a densely connected network
    :param n_layers: Number of hidden layers in the network
    :param hidden_size: How many hidden neurons to use
    :param n_features: Number of features in the original input
    :param n_outputs: Output size/number of target variables
    :return:
    """
    shapes = []
    for i in range(n_layers):
        shapes.append((hidden_size * (i + 1) + n_features, hidden_size))
    shapes.append((hidden_size * (n_layers + 1), n_outputs))
    return shapes