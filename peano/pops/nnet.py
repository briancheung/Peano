import types
import numpy as np
import theano
import theano.tensor as T
from utils import sample_weights
from pop import Pop

dtype = theano.config.floatX

def batch_normalize(x, epsilon=1e-6):
    return (x - x.mean(-2, keepdims=True))/(x.std(-2, keepdims=True) + epsilon)

def stable_softmax(x):
    return T.nnet.softmax(x - x.max(axis=1, keepdims=True))

class Sequential(Pop):
    def __init__(self, name):
        self.params = []
        self.ops = []

        self.name = name
        self.param_count = 0

    def add(self, op):
        if isinstance(op, Pop):
            for param in op.params:
                param_str = self.name + '_' + str(self.param_count)
                if param.name is None:
                    param.name = param_str
                else:
                    param.name = param_str + '_' + param.name
                self.param_count += 1

            self.params += op.params

        self.ops.append(op)

    def apply(self, x):
        hvals = x
        for op in self.ops:
            hvals = op(hvals)

        return hvals

class Linear(Pop):
    def __init__(self, n_in, n_output):
        self.W = theano.shared(sample_weights(n_in, n_output))
        self.b = theano.shared(np.zeros(n_output, dtype=dtype))
        self.params = [self.W, self.b]

    def apply(self, x):
        return x.dot(self.W) + self.b

class ZeroBias(Pop):
    def __init__(self, n_in, n_output, selection_threshold=1.):
        self.W = theano.shared(sample_weights(n_in, n_output))
        self.params = [self.W]

        self.selection_threshold = selection_threshold

    def apply(self, x):
        pre_hidden = x.dot(self.W)
        return (pre_hidden > self.selection_threshold)*pre_hidden

class ElementwiseLinear(Pop):
    def __init__(self, n_in):
        self.gamma = theano.shared(np.squeeze(sample_weights(1,n_in)))
        self.beta = theano.shared(np.zeros(n_in, dtype=dtype))

        self.params = [self.gamma, self.beta]

    def apply(self, x):
        return self.gamma*x + self.beta

class BatchNormalization(Pop):
    def __init__(self, n_in, epsilon=1e-6):
        self.epsilon = epsilon

        # TODO: Figure out if np.ones(n_in, dtype=dtype) is 
        # consistently worse for gamma
        self.gamma = theano.shared(np.squeeze(sample_weights(1,n_in)))
        self.beta = theano.shared(np.zeros(n_in, dtype=dtype))

        self.params = [self.gamma, self.beta]

    def apply(self, x):
        # TODO: Find a better way to approximate means at test time
        x_n = batch_normalize(x, epsilon=self.epsilon)
        return self.gamma*x_n + self.beta

class Lateral(Pop):
    def __init__(self, n_in):
        self.a1 = ElementwiseLinear(n_in)
        self.a2 = ElementwiseLinear(n_in)
        self.a3 = ElementwiseLinear(n_in)
        self.a4 = ElementwiseLinear(n_in)
        self.a5 = ElementwiseLinear(n_in)

        self.params = self.a1.params + \
                        self.a2.params + \
                        self.a3.params + \
                        self.a4.params + \
                        self.a5.params

    def apply(self, z, u):
        a1f, a2f, a3f, a4f, a5f = self.a1.apply(u), self.a2.apply(u), self.a3.apply(u), self.a4.apply(u), self.a5.apply(u)
        zh = a1f*z + a2f*T.nnet.sigmoid(a3f*z + a4f) + a5f

        return zh
