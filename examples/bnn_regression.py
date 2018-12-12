from likurai.model import BayesianNeuralNetwork
from likurai.layer import BayesianDense, Likelihood
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import ensemble
import matplotlib.pyplot as plt
from likurai import floatX, shared

if __name__ == '__main__':
    # Define some constants for the model architecture
    HIDDEN_SIZE = 6

    # Load the dataset
    boston = load_boston()
    X, y = boston['data'], boston['target']
    X = X.astype(floatX)
    y = y.astype(floatX).reshape(-1, 1)

    # We'll arbitrarily leave out the last 100 data points as our test set
    X_train, y_train = X[:-100, :], y[:-100]
    X_test, y_test = X[-100:, :], y[-100:]

    # Get some information about the dataset that we'll need for the model
    n_features = X.shape[1]
    n_samples  = X.shape[0]

    # Initialize a model object
    bnn = BayesianNeuralNetwork()
    bnn.x.set_value(X_train.astype(floatX))
    bnn.y = shared(y_train.astype(floatX))

    # Create our first layer (Input -> Hidden1). We specify the priors for the weights/bias in the kwargs
    with bnn.model:
        input_layer = BayesianDense('input', input_size=n_features, neurons=HIDDEN_SIZE, activation='relu')(bnn.x)

        # # Create a hidden layer. We can also specify the shapes for the weights/bias in the kwargs
        hidden_layer_1 = BayesianDense('hidden1', input_size=HIDDEN_SIZE, neurons=HIDDEN_SIZE, activation='relu')(input_layer)

        # Create our output layer
        output_layer = BayesianDense('output', input_size=HIDDEN_SIZE, neurons=1, activation='relu')(hidden_layer_1)

        likelihood = Likelihood('Gamma', 'alpha')(output_layer,
                                                  **{'jitter': 1e-7, 'observed': bnn.y,
                                                          'beta': {'dist': 'HalfCauchy', 'name': 'beta', 'beta': 3.}})

    # The model itself follows the scikit-learn interface for training/predicting
    # bnn.fit(X_train, y_train, epochs=1000, method='nuts', **{'tune': 2000, 'njobs': 1, 'chains': 1})
    bnn.fit(X_train, y_train, epochs=100000, method='advi', batch_size=32, n_models=1)

    # Generate predictions
    pred = bnn.predict(X_test, n_samples=1000)
    print(pred.shape)
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