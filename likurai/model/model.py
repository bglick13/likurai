"""
Defines the interface for a Model. Ideally, everything in this API will follow this signature, which will be similar
to scikit-learn's
"""


class Model:
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass
