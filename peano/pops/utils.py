import numpy as np
import theano

dtype = theano.config.floatX

def sample_weights(sizeX, sizeY):
    W = np.random.randn(sizeX, sizeY)
    return W.astype(dtype)/np.sqrt(sizeX)

# def sample_weights(sizeX, sizeY):
#     W = np.random.randn(sizeX, sizeY)
#     U,S,V = np.linalg.svd(W, full_matrices=False)
#     W = U.dot(V)

#     return W.astype(dtype)*np.sqrt(2.)
