from sklearn.datasets import load_digits
from likurai.layer import BayesianDense, BayesianConv2D, Flatten, MaxPooling2D, Likelihood
from likurai.model import BayesianNeuralNetwork
from likurai import floatX
from likurai import shared
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Define some constants for the model architecture
    HIDDEN_SIZE = 6

    # Load the dataset
    digits = load_digits()
    X, y = digits['data'], digits['target']
    X = X.reshape((X.shape[0], 8, 8, 1))
    img_width = X.shape[1]
    img_height = X.shape[2]
    X = X.astype(floatX)
    y = y.astype(floatX).reshape(-1, 1)

    # We'll arbitrarily leave out the last 100 data points as our test set
    X_train, y_train = X[:-100, :], y[:-100]
    X_test, y_test = X[-100:, :], y[-100:]

    # Get some information about the dataset that we'll need for the model
    n_features = X.shape[1]
    n_samples = X.shape[0]

    # Initialize a model object
    bnn = BayesianNeuralNetwork()
    bnn.x = shared(X_train.astype(floatX))
    bnn.y = shared(y_train.astype(floatX))

    # Create our first layer (Input -> Hidden1). We specify the priors for the weights/bias in the kwargs
    with bnn.model:
        input_layer = BayesianConv2D('input', input_shape=X[0].shape, output_size=HIDDEN_SIZE, activation='relu',
                                     filter_size=(3, 3))(bnn.x)

        # # Create a hidden layer. We can also specify the shapes for the weights/bias in the kwargs
        hidden_layer_1 = BayesianConv2D('hidden1', input_shape=(HIDDEN_SIZE, img_height, img_width),
                                        output_size=HIDDEN_SIZE*2, activation='relu', filter_size=(3, 3))(input_layer)

        pool = MaxPooling2D((2, 2))(hidden_layer_1)
        flatten = Flatten()(pool)

        # Create our output layer
        output_layer = BayesianDense('output', activation='relu', shape=(HIDDEN_SIZE, 1))(flatten)

        likelihood = Likelihood('Multinomial', 'p')(output_layer, **{'observed': bnn.y, 'n': 1})

    # The model itself follows the scikit-learn interface for training/predicting
    bnn.fit(X_train, y_train, epochs=1000, method='nuts', **{'tune': 2000, 'njobs': 1, 'chains': 1})
    # bnn.fit(X_train, y_train, epochs=100000, method='advi', batch_size=32)

    # Generate predictions
    pred = bnn.predict(X_test, n_samples=1000)
    print(pred)
    # However, for simplicity's sake, we can also tell the model to just give us point-estimate predictions
    point_pred = bnn.predict(X_test, n_samples=1000, point_estimate=True)

    # Let's just make a simple baseline using a scikit model. Eventually I'll use a comparable NN
    # params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
    #           'learning_rate': 0.01, 'loss': 'ls'}
    # clf = ensemble.GradientBoostingRegressor(**params)
    #
    # clf.fit(X_train, y_train)
    # clf_pred = clf.predict(X_test)
    #
    # mse = mean_squared_error(y_test, clf_pred)
    # r2 = r2_score(y_test, clf_pred)
    #
    # print(point_pred.shape)
    # fig, ax = plt.subplots()
    # ax.scatter(point_pred, y_test)
    # ax.scatter(clf_pred, y_test, c='g')
    # plt.legend(['BNN', 'Baseline'])
    # print('R2 Score: {}, Baseline R2: {}'.format(r2_score(y_test, point_pred), r2))
    # print('MSE Score: {}, Baseline MSE: {}'.format(mean_squared_error(y_test, point_pred), mse))
    # plt.show()