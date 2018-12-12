from likurai.model import BayesianNeuralNetwork
import pymc3 as pm
import matplotlib.pyplot as plt
from likurai.util import get_class_exemplar


if __name__ == '__main__':
    bnn = BayesianNeuralNetwork()
    bnn.load_model('mnist_informative.pickle')
    bnn.x.set_value(bnn.train_x)
    bnn.y.set_value(bnn.train_y)
    with bnn.model:
        draws = pm.sample_ppc(bnn.trace[0], 1000)['likelihood']

    best_threes = get_class_exemplar(draws.mean(axis=0), 3, 3)
    worst_threes = get_class_exemplar(draws.mean(axis=0), 3, 3, best=False, labels=bnn.train_y)
    for i, three in enumerate(best_threes):
        fig, ax = plt.subplots()
        plt.imshow(bnn.train_x[three].reshape(8, 8))
        plt.title("Best three [{}]".format(i))

    for i, three in enumerate(worst_threes):
        fig, ax = plt.subplots()
        plt.imshow(bnn.train_x[three].reshape(8, 8))
        plt.title("Worst three [{}]".format(i))
    plt.show()
