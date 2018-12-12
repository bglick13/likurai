from likurai.model import BayesianNeuralNetwork
import pymc3 as pm


if __name__ == '__main__':
    bnn = BayesianNeuralNetwork()
    bnn.load_model('classification_example.pickle')
    bnn.x.set_value(bnn.train_x)
    bnn.y.set_value(bnn.train_y)
    with bnn.model:
        draws = pm.sample_ppc(bnn.trace[0], 1000)
    print(draws)