'''
Demo of how to use lambda functions (anonymous functions) to tightly
integrate theano calls into peano. Anonymous functions enables peano 
to support new theano functions without modification.
'''

import time
import numpy as np
import theano
import theano.tensor as T
import peano
import peano.pops as P

dtype = theano.config.floatX

mnist_net = P.nnet.Sequential('mnist_net')
mnist_net.add(P.nnet.Linear(784, 500))
mnist_net.add(lambda x: T.nnet.relu(x, alpha=.1))
mnist_net.add(P.nnet.Linear(500, 500))
mnist_net.add(lambda x: T.nnet.relu(x, alpha=.1))
mnist_net.add(P.nnet.Linear(500, 10))
mnist_net.add(P.nnet.BatchNormalization(10))
mnist_net.add(T.nnet.softmax)

v = T.matrix(dtype=dtype)
y_true = T.matrix(dtype=dtype)
y_s = mnist_net.apply(v)

cost = T.nnet.categorical_crossentropy(y_s, y_true).mean()
misclass_cost = T.neq(T.argmax(y_true, axis=1), T.argmax(y_s, axis=1)).mean()

params = mnist_net.params
gparams = T.grad(cost, wrt=params)
updates = peano.optimizer.adam_update(params, gparams)

learn_mlp_fn = theano.function(inputs = [v, y_true],
                                outputs = cost,
                                updates = updates)

misclass_mlp_fn = theano.function(inputs = [v, y_true],
                                    outputs = misclass_cost)

from pylearn2.datasets import mnist
from pylearn2.space import CompositeSpace, VectorSpace

ds = mnist.MNIST(which_set = 'train', start=0, stop=50000)
val = mnist.MNIST(which_set = 'train', start=50000, stop=60000)
val_X, val_y = val.get_data()
val_y = np.squeeze(np.eye(10)[val_y]).astype(dtype)

data_space = VectorSpace(dim=784)
label_space = VectorSpace(dim= 10)

for i in range(200):
    cost = 0.
    misclass = 0.
    ds_iter = ds.iterator(mode='sequential', batch_size=100, data_specs=(CompositeSpace((data_space, label_space)), ('features', 'targets')))
    t0 = time.time()
    for X,y in ds_iter:
        cost += learn_mlp_fn(X,y)
    print misclass_mlp_fn(val_X, val_y), cost, time.time()-t0