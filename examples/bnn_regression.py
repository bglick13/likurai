from likurai.model import BayesianNeuralNetwork
from likurai.layer import BayesianDenseLayer
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import ensemble
import pymc3 as pm
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Define some constants for the model architecture
    HIDDEN_SIZE = 16

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
        hidden_layer_1 = BayesianDenseLayer('hidden1', type='Normal', activation='relu',
                                          **{'weight_kwargs': {'mu': 0., 'sd': 1., 'shape': (HIDDEN_SIZE, HIDDEN_SIZE)},
                                             'bias_kwargs': {'mu': 0., 'sd': 1., 'shape': HIDDEN_SIZE}})
        #
        # # Create a hidden layer. We can also specify the shapes for the weights/bias in the kwargs
        # hidden_layer_2 = BayesianDenseLayer('hidden2', type='Normal', activation='relu',
        #                                     **{'weight_kwargs': {'mu': 0., 'sd': 1.,
        #                                                          'shape': (HIDDEN_SIZE, HIDDEN_SIZE)},
        #                                        'bias_kwargs': {'mu': 0., 'sd': 1., 'shape': HIDDEN_SIZE}})

        # Create our output layer. We tell it not to use a bias.
        output_layer = BayesianDenseLayer('output', type='Normal', use_bias=False, activation='linear',
                                          **{'weight_kwargs': {'mu': 0., 'sd': 1., 'shape': (HIDDEN_SIZE, )}})

    bnn.add_layer(input_layer)
    bnn.add_layer(hidden_layer_1)
    # bnn.add_layer(hidden_layer_2)
    bnn.add_layer(output_layer)
    # Before we can use our model, we have to compile it. This adds the likelihood distribution that the model params
    # are conditioned on
    with bnn.model:
        likelihood = pm.Normal('likelihood', mu=bnn.activations[-1], sd=pm.HalfCauchy('sigma', 2.),
                                       observed=bnn.y, total_size=len(bnn.x.get_value()))

    # The model itself follows the scikit-learn interface for training/predicting
    bnn.fit(X_train, y, epochs=1000, method='nuts', **{'tune': 5000, 'chains': 2})
    pred = bnn.predict(X_test, n_samples=1000)
    print(pred)
    # However, for simplicity's sake, we can also tell the model to just give us point-estimate predictions
    point_pred = bnn.predict(X_test, n_samples=1000, point_estimate=True)

    # Let's just make a simple baseline using a scikit model. Eventually I'll use a comparable NN
    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
              'learning_rate': 0.01, 'loss': 'ls'}
    clf = ensemble.GradientBoostingRegressor(**params)

    clf.fit(X_train, y_train)
    mse = mean_squared_error(y_test, clf.predict(X_test))
    r2 = r2_score(y_test, clf.predict(X_test))

    print(point_pred.shape)
    fig, ax = plt.subplots()
    ax.scatter(point_pred, y_test)
    ax.scatter(clf.predict(X_test), y_test, c='g')
    plt.legend(['BNN', 'Baseline'])
    print('R2 Score: {}, Baseline R2: {}'.format(r2_score(y_test, point_pred), r2))
    print('MSE Score: {}, Baseline MSE: {}'.format(mean_squared_error(y_test, point_pred), mse))
    plt.show()