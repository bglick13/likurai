"""
Defines the interface for a Model. Ideally, everything in this API will follow this signature, which will be similar
to scikit-learn's
"""
from ..layer import Layer


class Model:
    def __init__(self):
        pass

    def add_layer(self, layer: Layer):
        """
        Helper function to build simple models. Makes the assumption that the input to a layer is only the output of
        the previous layer.
        The output layer requires the following additional kwargs...
        - sd (model variance)
        - observed (shared variable)
        - total_size (necessary for minibatch training)
        - mu = model.activations[-1]
        :param layer:
        :return:
        """
        with self.model:
            if len(self.layers) == 0:
                self.activations.append(layer(self.x))
            else:
                self.activations.append(layer(self.activations[-1]))
            self.layers.append(layer)

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def save_model(self, filepath):
        pass

    def load_model(self, filepath):
        pass

