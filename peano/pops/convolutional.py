import numpy as np
import theano
import theano.tensor as T
from utils import sample_weights
from pop import Pop

dtype = theano.config.floatX

class Conv2D(Pop):
    def __init__(self, n_filters, stack_size, n_row, n_col, 
                    stride=(1,1), border_mode='valid'):
        W_init = sample_weights(stack_size*n_row*n_col, n_filters).T
        W_init = W_init.reshape(n_filters, stack_size, n_row, n_col)
        self.W = theano.shared(W_init.astype(dtype))
        self.b = theano.shared(np.zeros(n_filters, dtype=dtype))
        self.params = [self.W, self.b]

        self.border_mode = border_mode
        self.stride = stride

    def apply(self, x):
        # x: (n_batch, stack_size, n_row, n_col)
        # conv2d may fallback  to super slow CPU conv when subsample != (1,1), better
        # to force using cuDNN for now
        # conv_val = T.nnet.conv.conv2d(x, self.W, border_mode=self.border_mode, subsample=self.stride)
        conv_val = theano.sandbox.cuda.dnn.dnn_conv(x, self.W, border_mode=self.border_mode, subsample=self.stride)
        return conv_val + self.b.dimshuffle(0,'x','x')