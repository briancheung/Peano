'''
Example of how to build a convolutional network to classify MNIST digits
'''

import time
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
import peano
import peano.pops as P

dtype = theano.config.floatX

mnist_net = P.nnet.Sequential('mnist_net')
mnist_net.add(P.convolutional.Conv2D(16, 1, 4, 4))
mnist_net.add(lambda x: downsample.max_pool_2d(x, (2,2)))
mnist_net.add(T.nnet.relu)
mnist_net.add(P.convolutional.Conv2D(32, 16, 4, 4))
mnist_net.add(lambda x: downsample.max_pool_2d(x, (2,2)))
mnist_net.add(T.nnet.relu)
mnist_net.add(lambda x: T.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]*x.shape[3]) ))
mnist_net.add(P.nnet.Linear(32*5*5, 10))
mnist_net.add(P.nnet.logsoftmax)

v = T.tensor4(dtype=dtype)
y_true = T.matrix(dtype=dtype)
y_s = mnist_net.apply(v)

cost = P.cost.cross_entropy_logdomain(y_true, y_s)
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
        cost += learn_mlp_fn(X.reshape(X.shape[0], 1, 28, 28),y)
    print misclass_mlp_fn(val_X.reshape(val_X.shape[0], 1, 28, 28), val_y), cost, time.time()-t0