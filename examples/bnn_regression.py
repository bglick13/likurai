from likurai.model import BayesianNeuralNetwork
from likurai.layer import BayesianDenseLayer
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score, mean_squared_error
import pymc3 as pm
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Define some constants for the model architecture
    HIDDEN_SIZE = 8

    # Load the dataset
    boston = load_boston()
    X, y = boston['data'], boston['target']
    # We'll arbitrarily leave out the last 100 data points as our test set
    X_train, y_train = X[:-100, :], y[:-100]
    X_test, y_test = X[-100:, :], y[-100:]
    # Get some information about the dataset that we'll need for the model
    n_features = X.shape[1]
    n_samples  = X.shape[0]

    # Initialize a model object
    bnn = BayesianNeuralNetwork()
    bnn.x.set_value(X_train)
    bnn.y.set_value(y_train)

    # Create our first layer (Input -> Hidden1). We specify the priors for the weights/bias in the kwargs
    with bnn.model:
        input_layer = BayesianDenseLayer('input', type='Normal', input_size=n_features, output_size=HIDDEN_SIZE, activation='relu',
                                         **{'weight_kwargs': {'mu': 0., 'sd': 1.},
                                            'bias_kwargs': {'mu': 0., 'sd': 1.}})

        # Create a hidden layer. We can also specify the shapes for the weights/bias in the kwargs
        hidden_layer = BayesianDenseLayer('hidden', type='Normal', activation='relu',
                                          **{'weight_kwargs': {'mu': 0., 'sd': 1., 'shape': (HIDDEN_SIZE, HIDDEN_SIZE)},
                                             'bias_kwargs': {'mu': 0., 'sd': 1., 'shape': HIDDEN_SIZE}})

        # Create our output layer. We tell it not to use a bias.
        output_layer = BayesianDenseLayer('output', type='Normal', use_bias=False, activation='linear',
                                          **{'weight_kwargs': {'mu': 0., 'sd': 1., 'shape': (HIDDEN_SIZE, )}})

    bnn.add_layer(input_layer)
    bnn.add_layer(hidden_layer)
    bnn.add_layer(output_layer)
    bnn.compile()
    # The model itself follows the scikit-learn interface for training/predicting
    bnn.fit(X_train, y, epochs=1000000, method='advi')
    pred = bnn.predict(X_test, n_samples=1000)
    print(pred)
    # However, for simplicity's sake, we can also tell the model to just give us point-estimate predictions
    point_pred = bnn.predict(X_test, n_samples=1000, point_estimate=True)
    print(point_pred.shape)
    fig, ax = plt.subplots()
    ax.scatter(point_pred, y_test)
    print('R2 Score: {}'.format(r2_score(y_test, point_pred)))
    print('MSE Score: {}'.format(mean_squared_error(y_test, point_pred)))
    plt.show()