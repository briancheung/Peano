import numpy as np
import theano

dtype = theano.config.floatX

def sample_weights(sizeX, sizeY):
    W = np.random.randn(sizeX, sizeY)
    return W.astype(dtype)/np.sqrt(sizeX)

# def sample_weights(sizeX, sizeY):
# Initialization from Lasagne
# https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py#L322
#         shape = (sizeX, sizeY)

#         if len(shape) < 2:
#             raise RuntimeError("Only shapes of length 2 or more are "
#                                "supported.")

#         flat_shape = (shape[0], np.prod(shape[1:]))
#         a = np.random.normal(0.0, 1.0, flat_shape)
#         u, _, v = np.linalg.svd(a, full_matrices=False)
#         # pick the one with the correct shape
#         q = u if u.shape == flat_shape else v
#         q = q.reshape(shape)
#         return q.astype(dtype)

# def sample_weights(sizeX, sizeY):
#     W = np.random.randn(sizeX, sizeY)
#     U,S,V = np.linalg.svd(W, full_matrices=False)
#     W = U.dot(V)

#     return W.astype(dtype)*np.sqrt(2.)
