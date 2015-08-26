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

class RecurrentConv2D(Pop):
    def __init__(self, n_filters, stack_size, n_row, n_col, 
                    stride=(1,1), border_mode='valid', activation_fn=T.tanh):
        W_init = sample_weights(stack_size*n_row*n_col, n_filters).T
        W_init = W_init.reshape(n_filters, stack_size, n_row, n_col)
        self.W_ih = theano.shared(W_init.astype(dtype))

        W_init = sample_weights(n_filters*n_row*n_col, n_filters).T
        W_init = W_init.reshape(n_filters, n_filters, n_row, n_col)
        self.W_hh = theano.shared(W_init.astype(dtype))

        self.b_h = theano.shared(np.zeros(n_filters, dtype=dtype))

        self.params = [self.W_ih,
                        self.W_hh,
                        self.b_h]

        self.border_mode = border_mode
        self.stride = stride
        self.activation_fn = activation_fn

    def step(self, x_t, h_tm1):
        xWih = theano.sandbox.cuda.dnn.dnn_conv(x_t, self.W_ih, border_mode=self.border_mode, subsample=self.stride)
        hWhh = theano.sandbox.cuda.dnn.dnn_conv(h_tm1, self.W_hh, border_mode=self.border_mode, stride=(1,1))

        h_t = self.activation_fn(xWih + hWhh + self.b_h.dimshuffle(0,'x','x'))

        return h_t