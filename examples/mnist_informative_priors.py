from likurai.layer import BayesianDense, BayesianConv2D, Flatten, MaxPooling2D, Likelihood
from likurai.model import BayesianNeuralNetwork
from likurai import floatX
from likurai import shared
from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from keras import layers
from keras.models import Sequential
from keras import backend as K


def make_keras_model():
    print(K.image_data_format())
    model = Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     data_format='channels_first',
                     input_shape=(1, 28, 28)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', data_format='channels_first'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])
    print(model.summary())
    return model


def load_data():
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    img_rows, img_cols = 28, 28
    x_train, y_train, x_test, y_test = load_data()
    y_train = OneHotEncoder().fit_transform(y_train.reshape(-1, 1))
    y_test = OneHotEncoder().fit_transform(y_test.reshape(-1, 1))

    model = make_keras_model()
    model.fit(x_train, y_train, epochs=25, batch_size=128)

    # Initialize a model object
    bnn = BayesianNeuralNetwork()
    bnn.x = shared(x_train.astype(floatX))
    bnn.y = shared(y_train.astype(floatX))

    # Create our first layer (Input -> Hidden1). We specify the priors for the weights/bias in the kwargs
    with bnn.model:
        input_layer = BayesianConv2D('input', filters=32, channels=1, mu=model.layers[0].get_weights()[0],
                                     activation='relu', filter_size=(3, 3))(bnn.x)

        # # Create a hidden layer. We can also specify the shapes for the weights/bias in the kwargs
        hidden_layer_1 = BayesianConv2D('hidden1', filters=64, channels=32,
                                        mu=model.layers[1].get_weights()[0], activation='relu',
                                        filter_size=(3, 3))(input_layer)

        pool = MaxPooling2D((2, 2))(hidden_layer_1)
        flatten = Flatten()(pool)
        dense = BayesianDense('dense_hidden', neurons=128, input_size=54, mu=model.layers[4].get_weights()[0], activation='relu')(flatten)

        # Create our output layer
        output_layer = BayesianDense('output', neurons=10, input_size=128, mu=model.layers[5].get_weights()[0], activation='softmax')(dense)

        likelihood = Likelihood('Multinomial', 'p')(output_layer, **{'observed': bnn.y, 'n': 1, 'jitter': 1.0e-7})

    # The model itself follows the scikit-learn interface for training/predicting
    # bnn.fit(X_train, y_train_one_hot, epochs=300, method='nuts', **{'tune': 1000, 'njobs': 1, 'chains': 1,
    #                                                                 'init': 'adapt_diag'})

    bnn.fit(x_train, y_train, epochs=10000, method='svgd', batch_size=32)
    bnn.save_model('mnist_informative.pickle')

    # Generate predictions
    # pred = bnn.predict(X_test, n_samples=1000)
    # print(pred)
    # However, for simplicity's sake, we can also tell the model to just give us point-estimate predictions
    point_pred = bnn.predict(x_test, n_samples=1000, point_estimate=True)
    print('BNN:\n{}'.format(classification_report(y_test, point_pred.argmax(axis=0))))
