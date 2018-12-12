**Why líkurai?**

Líkurai is a combination of líkur (the Icelandic word for probability) and AI (the English word for AI)

**What is líkurai?**

This is the home for all the Probabilistic AI tools I have developed over the course of my various personal projects.

I'm attempting to make it an easy to use API so that people can quickly get up and running with some fun stuff like Bayesian Neural Networks (BNN).

**What can I do with líkurai?**

For now, I've implemented some wrappers to easily do Bayedsian Deep Learning. Specifically,

- Bayesian Neural Network (Regression or Classification)
- Hierarchical Bayesian Neural Network (Regression or Classification)

Here's a short code snippet to show you just how simple it is...

```python
# Create our network object
bnn = BayesianNeuralNetwork()

# Unlike traditional DL libraries, the model needs to be aware of the data before construction
bnn.x = shared(X_train.astype(floatX))
bnn.y = shared(y_train.astype(floatX))

# We create network layers using the model's context - this is to play nicely with PyMC3, which líkurai wraps
with bnn.model:
    # We create a Dense input layer to our network
    input_layer = BayesianDense('input', input_size=n_features, neurons=HIDDEN_SIZE, activation='relu')(bnn.x)

    # Create a hidden layer. The API is functional so we call the layer with its inputs
    hidden_layer_1 = BayesianDense('hidden1', input_size=HIDDEN_SIZE, neurons=HIDDEN_SIZE, activation='relu')(input_layer)

    # Create our output layer
    output_layer = BayesianDense('output', input_size=HIDDEN_SIZE, neurons=1, activation='relu')(hidden_layer_1)

    # Unlike traditional DL libraries, we need a Likelihood layer which specifies the distribution to condition the network 0n
    likelihood = Likelihood('Gamma', 'alpha')(output_layer,
                                              **{'jitter': 1e-7, 'observed': bnn.y,
                                                  'beta': {'dist': 'HalfCauchy', 'name': 'beta', 'beta': 3.}})

# The model itself follows the scikit-learn interface for training/predicting
# Worth noting that ADVI will work for toy problems, but NUTS is a better sampling method for more complex data
bnn.fit(X_train, y_train, epochs=100000, method='advi', batch_size=32, n_models=1)

# Generate predictions
pred = bnn.predict(X_test, n_samples=1000)

```

The TODO list includes the following:
- Speed improvements. The downside of Bayesian DL is it's significantly slower than traditional DL
    - Use weights from a traditional NN as informative priors
    - Train ensembles of models using ADVI instead of NUTS
- Visualization/explanation
    - Explore the decision/uncertainty surface for features
    - Explore the level of specialization in Hierarchical BNNs
    - Sample posterior predictive with train data to generate "exemplars" for a given class   


**Okay but why Bayesian?**

Becuase the world is uncertain! Sure, traditional neural networks are great for certain tasks like playing Atari games or doing image classification, but even that gets tricky pretty quickly.


For example, how would an animal classifier do on the image below?

![alt text](https://imgix.bustle.com/rehost/2016/9/13/f479670c-be2b-4cb1-b224-27605abd2a68.jpg?w=970&h=582&fit=crop&crop=faces&auto=format&q=70)

Is it a cat? No. Is it a dog? Also no. A traditional neural network will be forced to guess, and has no real way to express that fact.

Of course, a BNN won't get it "right" either, since "right" doesn't exist in this example. However, a BNN can be *more* right by expressing how wrong it thinks it is.

**Fine, but does it work?**

Yeah! What'd you expect me to say?

It's worked on a bunch of super secret personal projects. That's the whole point of me making this library.

I'm working on making some examples on traditional datasets (MNIST!) so you can have a real comparison, but until then, just take my word for it.