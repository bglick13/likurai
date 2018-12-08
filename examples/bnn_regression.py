from likurai.model import BayesianNeuralNetwork
from likurai.layer import BayesianDenseLayer
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import QuantileTransformer
from sklearn import ensemble
import pymc3 as pm
import matplotlib.pyplot as plt
import theano
floatX = theano.config.floatX

if __name__ == '__main__':
    # Define some constants for the model architecture
    HIDDEN_SIZE = 6

    # Load the dataset
    boston = load_boston()
    X, y = boston['data'], boston['target']
    X = X.astype(floatX)
    y = y.astype(floatX)

    # We'll arbitrarily leave out the last 100 data points as our test set
    X_train, y_train = X[:-100, :], y[:-100]
    X_test, y_test = X[-100:, :], y[-100:]

    # Get some information about the dataset that we'll need for the model
    n_features = X.shape[1]
    n_samples  = X.shape[0]

    # Initialize a model object
    bnn = BayesianNeuralNetwork()
    bnn.x.set_value(X_train.astype(floatX))
    bnn.y.set_value(y_train.astype(floatX))

    # Create our first layer (Input -> Hidden1). We specify the priors for the weights/bias in the kwargs
    with bnn.model:
        input_layer = BayesianDenseLayer('input', weight_dist='Normal',
                                         input_size=n_features, output_size=HIDDEN_SIZE, activation='relu',
                                         **{'weight_kwargs': {'mu': 0., 'sd': .5},
                                            'bias_kwargs': {'mu': 0., 'sd': .5}})

        # # Create a hidden layer. We can also specify the shapes for the weights/bias in the kwargs
        hidden_layer_1 = BayesianDenseLayer('hidden1', weight_dist='Normal', activation='relu',
                                            **{'weight_kwargs': {'mu': 0., 'sd': .5, 'shape': (HIDDEN_SIZE, HIDDEN_SIZE)},
                                             'bias_kwargs': {'mu': 0., 'sd': .5, 'shape': HIDDEN_SIZE}})
        # #
        # Create a hidden layer. We can also specify the shapes for the weights/bias in the kwargs
        # hidden_layer_2 = BayesianDenseLayer('hidden2', weight_dist='Normal', activation='relu',
        #                                     **{'weight_kwargs': {'mu': 0., 'sd': 1.,
        #                                                          'shape': (HIDDEN_SIZE, HIDDEN_SIZE)},
        #                                        'bias_kwargs': {'mu': 0., 'sd': 1., 'shape': HIDDEN_SIZE}})
        #
        # hidden_layer_3 = BayesianDenseLayer('hidden3', type='Normal', activation='relu',
        #                                     **{'weight_kwargs': {'mu': 0., 'sd': 1.,
        #                                                          'shape': (HIDDEN_SIZE, HIDDEN_SIZE)},
        #                                        'bias_kwargs': {'mu': 0., 'sd': 1., 'shape': HIDDEN_SIZE}})

        # Create our output layer. We tell it not to use a bias.
        output_layer = BayesianDenseLayer('output', weight_dist='Normal', activation='relu',
                                          **{'weight_kwargs': {'mu': 0., 'sd': .5, 'shape': (HIDDEN_SIZE, )},
                                             'bias_kwargs': {'mu': 0., 'sd': .5, 'shape': 1}})

    bnn.add_layer(input_layer)
    bnn.add_layer(hidden_layer_1)
    # bnn.add_layer(hidden_layer_2)
    # bnn.add_layer(hidden_layer_3)
    bnn.add_layer(output_layer)
    # Before we can use our model, we have to compile it. This adds the likelihood distribution that the model params
    # are conditioned on
    bnn.compile('Gamma', 'alpha', **{'jitter': 1e-7, 'beta': {'dist': 'HalfCauchy', 'name': 'beta', 'beta': 3.}})
    # with bnn.model:
    #     likelihood = pm.Gamma('likelihood', alpha=bnn.activations[-1] + 1.e-7, beta=pm.HalfCauchy('beta', 3.),
    #                            observed=bnn.y, total_size=len(bnn.x.get_value()))

    # The model itself follows the scikit-learn interface for training/predicting
    # bnn.fit(X_train, y_train, epochs=1000, method='nuts', **{'tune': 2000, 'njobs': 1, 'chains': 1})
    bnn.fit(X_train, y_train, epochs=100000, method='advi', batch_size=32)

    # Generate predictions
    pred = bnn.predict(X_test, n_samples=1000)
    print(pred)
    # However, for simplicity's sake, we can also tell the model to just give us point-estimate predictions
    point_pred = bnn.predict(X_test, n_samples=1000, point_estimate=True)

    # Let's just make a simple baseline using a scikit model. Eventually I'll use a comparable NN
    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
              'learning_rate': 0.01, 'loss': 'ls'}
    clf = ensemble.GradientBoostingRegressor(**params)

    clf.fit(X_train, y_train)
    clf_pred = clf.predict(X_test)

    mse = mean_squared_error(y_test, clf_pred)
    r2 = r2_score(y_test, clf_pred)

    print(point_pred.shape)
    fig, ax = plt.subplots()
    ax.scatter(point_pred, y_test)
    ax.scatter(clf_pred, y_test, c='g')
    plt.legend(['BNN', 'Baseline'])
    print('R2 Score: {}, Baseline R2: {}'.format(r2_score(y_test, point_pred), r2))
    print('MSE Score: {}, Baseline MSE: {}'.format(mean_squared_error(y_test, point_pred), mse))
    plt.show()