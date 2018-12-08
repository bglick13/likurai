from likurai.model import BayesianNeuralNetwork
from likurai.layer import BayesianDenseLayer
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, log_loss
from theano import shared
from sklearn import ensemble
from sklearn.preprocessing import OneHotEncoder
import pymc3 as pm
from sklearn.utils import shuffle
import theano
import numpy as np
floatX = theano.config.floatX

if __name__ == '__main__':
    # Define some constants for the model architecture
    HIDDEN_SIZE = 6

    # Load the dataset
    iris = load_iris()
    X, y = iris['data'], iris['target']
    X, y = shuffle(X, y)
    y_one_hot = OneHotEncoder(sparse=False, dtype=floatX).fit_transform(y.reshape(-1, 1))
    print(y.shape)
    # We'll arbitrarily leave out the last 50 data points as our test set
    X_train, y_train, y_train_one_hot = X[:-20, :], y[:-20], y_one_hot[:-20]
    X_test, y_test, y_test_one_hot = X[-20:, :], y[-20:], y_one_hot[-20:]

    # Get some information about the dataset that we'll need for the model
    n_features = X.shape[1]
    n_samples  = X.shape[0]
    n_classes  = y_one_hot.shape[1]

    # Initialize a model object
    bnn = BayesianNeuralNetwork()
    bnn.x.set_value(X_train.astype(floatX))
    bnn.y = shared(y_train_one_hot)

    # Create our first layer (Input -> Hidden1). We specify the priors for the weights/bias in the kwargs
    with bnn.model:
        input_layer = BayesianDenseLayer('input', weight_dist='Normal', shape=(n_features, HIDDEN_SIZE),
                                         activation='relu')

        # # Create a hidden layer. We can also specify the shapes for the weights/bias in the kwargs
        hidden_layer_1 = BayesianDenseLayer('hidden1', weight_dist='Normal', activation='relu', shape=(HIDDEN_SIZE, HIDDEN_SIZE))

        # Create our output layer. We tell it not to use a bias.
        output_layer = BayesianDenseLayer('output', weight_dist='Normal', activation='softmax', shape=(HIDDEN_SIZE, n_classes))

    bnn.add_layer(input_layer)
    bnn.add_layer(hidden_layer_1)
    bnn.add_layer(output_layer)

    # Before we can use our model, we have to compile it. This adds the likelihood distribution that the model params
    # are conditioned on
    bnn.compile('Multinomial', 'p', **{'n': 1})

    # The model itself follows the scikit-learn interface for training/predicting
    bnn.fit(X_train, y, epochs=1000, method='nuts', **{'tune': 2000, 'njobs': 1, 'chains': 1})
    # bnn.fit(X_train, y_train_one_hot, epochs=100000, method='advi', batch_size=32)

    # Generate predictions
    pred = bnn.predict(X_test, n_samples=1000)
    print(pred.shape)
    # However, for simplicity's sake, we can also tell the model to just give us point-estimate predictions
    point_pred = bnn.predict(X_test, n_samples=1000, point_estimate=True)
    point_pred = point_pred
    print(point_pred)
    print(point_pred.shape)
    # Let's just make a simple baseline using a scikit model. Eventually I'll use a comparable NN
    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2, 'learning_rate': 0.01}
    clf = ensemble.GradientBoostingClassifier(**params)

    clf.fit(X_train, y_train)
    clf_pred = clf.predict(X_test)
    print(np.unique(y_test))

    print('BNN:\n{}'.format(classification_report(y_test, point_pred.argmax(axis=0))))
    print('Baseline:\n{}'.format(classification_report(y_test, clf_pred)))

    print("Log-Loss BNN: {}".format(log_loss(y_test, point_pred.argmax(axis=0))))
    print("Log-Loss BNN (probabilistic): {}".format(log_loss(y_test, point_pred.T)))
    print("Log-Loss Baseline: {}".format(log_loss(y_test, clf_pred)))
