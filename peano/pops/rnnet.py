import numpy as np
import theano
import theano.tensor as T
from utils import sample_weights
from pop import Pop

dtype = theano.config.floatX

class Simple(Pop):
    def __init__(self, n_in, n_hidden, activation_fn=T.tanh):
        self.W_ih = theano.shared(sample_weights(n_in, n_hidden))
        self.W_hh = theano.shared(sample_weights(n_hidden, n_hidden))
        self.b_h = theano.shared(np.zeros(n_hidden, dtype=dtype))

        self.h0 = theano.shared(np.zeros((1, n_hidden), dtype=dtype))

        self.params = [self.W_ih,
                        self.W_hh,
                        self.b_h,
                        self.h0]

        self.activation_fn = activation_fn

    def step(self, x_t, h_tm1):
        h_t = self.activation_fn(x_t.dot(self.W_ih) + h_tm1.dot(self.W_hh) + self.b_h)
        return h_t

    def apply(self, v):
        h_vals, _ = theano.scan(fn = self.step, 
                                  sequences = v, 
                                  outputs_info = T.tile(self.h0, (v.shape[1], 1)))
        return h_vals

class Multiplicative(Pop):
    def __init__(self, n_in, n_hidden, n_factor, activation_fn=T.tanh):
        self.W_ih = theano.shared(sample_weights(n_in, n_hidden))
        self.W_if = theano.shared(sample_weights(n_in, n_factor))
        self.W_hf = theano.shared(sample_weights(n_hidden, n_factor))
        self.W_fh = theano.shared(sample_weights(n_factor, n_hidden))
        self.b_h = theano.shared(np.zeros(n_hidden, dtype=dtype))

        self.h0 = theano.shared(np.zeros((1, n_hidden), dtype=dtype))

        self.params = [self.W_ih,
                        self.W_if,
                        self.W_hf,
                        self.W_fh,
                        self.b_h,
                        self.h0]

        self.activation_fn = activation_fn

    def step(self, x_t, h_tm1):
        f_t = h_tm1.dot(self.W_hf)*x_t.dot(self.W_if)
        h_t = self.activation_fn(x_t.dot(self.W_ih) + f_t.dot(self.W_fh) + self.b_h)
        return h_t

    def apply(self, v):
        h_vals, _ = theano.scan(fn = self.step, 
                                  sequences = v, 
                                  outputs_info = T.tile(self.h0, (v.shape[1], 1)) 
                                )
        return h_vals

class Feedback(Pop):
    def __init__(self, n_in, n_hidden, n_feedback, activation_fn=T.tanh):
        self.W_ih = theano.shared(sample_weights(n_in, n_hidden))
        self.W_hh = theano.shared(sample_weights(n_hidden, n_hidden))
        self.b_h = theano.shared(np.zeros(n_hidden, dtype=dtype))

        self.W_fh = theano.shared(sample_weights(n_feedback, n_hidden))
        self.W_ff = theano.shared(sample_weights(n_feedback, n_feedback))
        self.W_hf = theano.shared(sample_weights(n_hidden, n_feedback))
        self.b_f = theano.shared(np.zeros(n_feedback, dtype=dtype))

        self.h0 = theano.shared(np.zeros((1, n_hidden), dtype=dtype))
        self.f0 = theano.shared(np.zeros((1, n_feedback), dtype=dtype))

        self.params = [self.W_ih,
                        self.W_hh,
                        self.b_h,
                        self.W_fh,
                        self.W_ff,
                        self.W_hf,
                        self.b_f,
                        self.h0,
                        self.f0]

        self.activation_fn = activation_fn

    def step(self, x_t, h_tm1, f_tm1):
        h_t = self.activation_fn(f_tm1.dot(self.W_fh) + x_t.dot(self.W_ih) + h_tm1.dot(self.W_hh) + self.b_h)
        f_t = self.activation_fn(f_tm1.dot(self.W_ff) + h_t.dot(self.W_hf) + self.b_f)
        return h_t, f_t

    def apply(self, v):
        [h_vals, f_vals], _ = theano.scan(fn = self.step, 
                                          sequences = v, 
                                          outputs_info = [T.tile(self.h0, (v.shape[1], 1)),
                                                            T.tile(self.f0, (v.shape[1], 1))])
        return f_vals

class GatedRecurrent(Pop):
    """
    Gated Recurrent Unit

    Implementation described in Visin et. al. 2015


    """
    def __init__(self, n_in, n_hidden, n_batch=1, init_state_params=True, activation_fn=T.tanh):
        self.W = theano.shared(sample_weights(n_in, n_hidden))
        self.U = theano.shared(sample_weights(n_hidden, n_hidden))
        self.b = theano.shared(np.zeros(n_hidden, dtype=dtype))

        self.Wz = theano.shared(sample_weights(n_in, n_hidden))
        self.Uz = theano.shared(sample_weights(n_hidden, n_hidden))
        self.bz = theano.shared(np.zeros(n_hidden, dtype=dtype))

        self.Wr = theano.shared(sample_weights(n_in, n_hidden))
        self.Ur = theano.shared(sample_weights(n_hidden, n_hidden))
        self.br = theano.shared(np.zeros(n_hidden, dtype=dtype))

        self.params = [self.W,
                        self.U,
                        self.b,
                        self.Wz,
                        self.Uz,
                        self.bz,
                        self.Wr,
                        self.Ur,
                        self.br]

        self.h0 = theano.shared(np.zeros((n_batch, n_hidden), dtype=dtype))
        if init_state_params:
            self.params += [self.h0]

        self.n_batch = n_batch
        self.activation_fn = activation_fn

    def step(self, x_t, h_tm1):
        z_t = T.nnet.sigmoid(x_t.dot(self.Wz) + h_tm1.dot(self.Uz) + self.bz)
        r_t = T.nnet.sigmoid(x_t.dot(self.Wr) + h_tm1.dot(self.Ur) + self.br)
        hh_t = self.activation_fn(x_t.dot(self.W) + (h_tm1*r_t).dot(self.U) + self.b)
        h_t = (1.-z_t)*h_tm1 + z_t*hh_t

        return h_t

    def apply(self, v):
        if self.n_batch == 1:
            h_init = T.tile(self.h0, (v.shape[1], 1))
        else:
            h_init = self.h0

        h_vals, _ = theano.scan(fn = self.step,
                                sequences = v,
                                outputs_info = h_init)
        return h_vals

class LSTM(Pop):
    """
    Long short term memory

    Implementation described in Graves and Schmidhuber 2005.

    Graves, Alex, and Jurgen Schmidhuber. Framewise phoneme
    classification with bidirectional LSTM and other neural
    network architectures. Neural Networks 18.5 (2005): 602-610.
    """
    def __init__(self, n_in, n_hidden, activation_fn=T.tanh):
        n_i = n_c = n_o = n_f = n_hidden

        self.W_xi = theano.shared(sample_weights(n_in, n_i))
        self.W_hi = theano.shared(sample_weights(n_hidden, n_i))
        self.W_ci = theano.shared(sample_weights(n_c, n_i))
        self.b_i = theano.shared(np.zeros(n_i, dtype=dtype))

        self.W_xf = theano.shared(sample_weights(n_in, n_f))
        self.W_hf = theano.shared(sample_weights(n_hidden, n_f))
        self.W_cf = theano.shared(sample_weights(n_c, n_f))
        self.b_f = theano.shared(np.zeros(n_f, dtype=dtype))
        
        self.W_xc = theano.shared(sample_weights(n_in, n_c))
        self.W_hc = theano.shared(sample_weights(n_hidden, n_c))
        self.b_c = theano.shared(np.zeros(n_c, dtype=dtype))
        
        self.W_xo = theano.shared(sample_weights(n_in, n_o))
        self.W_ho = theano.shared(sample_weights(n_hidden, n_o))
        self.W_co = theano.shared(sample_weights(n_c, n_o))
        self.b_o = theano.shared(np.zeros(n_o, dtype=dtype))

        self.c0 = theano.shared(np.zeros((1, n_hidden), dtype=dtype))
        self.h0 = theano.shared(np.zeros((1, n_hidden), dtype=dtype))

        self.params = [self.W_xi,
                        self.W_hi,
                        self.W_ci,
                        self.b_i,
                        self.W_xf,
                        self.W_hf,
                        self.W_cf,
                        self.b_f,
                        self.W_xc,
                        self.W_hc,
                        self.b_c,
                        self.W_xo,
                        self.W_ho,
                        self.W_co,
                        self.b_o,
                        self.c0,
                        self.h0]

        self.activation_fn = activation_fn

    def step(self, x_t, h_tm1, c_tm1):
        i_t = T.nnet.sigmoid(x_t.dot(self.W_xi) + h_tm1.dot(self.W_hi) + c_tm1.dot(self.W_ci) + self.b_i)
        f_t = T.nnet.sigmoid(x_t.dot(self.W_xf) + h_tm1.dot(self.W_hf) + c_tm1.dot(self.W_cf) + self.b_f)
        c_t = f_t*c_tm1 + i_t*self.activation_fn(x_t.dot(self.W_xc) + h_tm1.dot(self.W_hc) + self.b_c) 
        o_t = T.nnet.sigmoid(x_t.dot(self.W_xo) + h_tm1.dot(self.W_ho) + c_t.dot(self.W_co)  + self.b_o)
        h_t = o_t*T.tanh(c_t)
        return h_t, c_t

    def apply(self, v):
        [h_vals, _], _ = theano.scan(fn=self.step, 
                                        sequences = v, 
                                        outputs_info = [T.tile(self.h0, (v.shape[1], 1)),
                                                        T.tile(self.c0, (v.shape[1], 1))] 
                                    )
        return h_vals

class GoogleLSTM(Pop):
    """
    Google LSTM 

    Implementation described in Vinyals et. al. 2014.

    Vinyals, Oriol, et al. "Grammar as a Foreign Language."
    arXiv preprint arXiv:1412.7449 (2014).
    """
    def __init__(self, n_in, n_hidden, n_batch=None, activation_fn=T.tanh):
        self.W = theano.shared(sample_weights(n_in, 4*n_hidden))
        self.T = theano.shared(sample_weights(n_hidden, 4*n_hidden))
        self.b = theano.shared(np.zeros(4*n_hidden, dtype=dtype))

        self.params = [self.W,
                        self.T,
                        self.b]

        if n_batch is None:
            self.h0 = theano.shared(np.zeros((1, n_hidden), dtype=dtype))
            self.c0 = theano.shared(np.zeros((1, n_hidden), dtype=dtype))
            self.params += [self.h0, self.c0]
            self.h_init = [T.tile(self.h0, (v.shape[1], 1)),
                            T.tile(self.c0, (v.shape[1], 1))]
        else:
            self.h0 = theano.shared(np.zeros((n_batch, n_hidden), dtype=dtype))
            self.c0 = theano.shared(np.zeros((n_batch, n_hidden), dtype=dtype))
            self.h_init = [self.h0, self.c0]

        self.n_in = n_in
        self.n_hidden = n_hidden
        self.activation_fn = activation_fn

    def step(self, x_t, h_tm1, c_tm1):
        M = x_t.dot(self.W) + h_tm1.dot(self.T) + self.b
        i_t = T.nnet.sigmoid(M[:,0:self.n_hidden])
        f_t = T.nnet.sigmoid(M[:,self.n_hidden:2*self.n_hidden])
        o_t = T.nnet.sigmoid(M[:,2*self.n_hidden:3*self.n_hidden])
        c_t = f_t*c_tm1 + i_t*self.activation_fn(M[:,3*self.n_hidden:])
        h_t = o_t*T.tanh(c_t)
        return h_t, c_t

    def apply(self, v):
        [h_vals, _], _ = theano.scan(fn=self.step, 
                                        sequences = v, 
                                        outputs_info = self.h_init)
        return h_vals

class SimpleLSTM(Pop):
    """
    Simple LSTM

    Simplifications of the LSTM suggested in http://arxiv.org/abs/1503.04069.
    No peepholes and forget gate is now coupled with input gate.
    """
    def __init__(self, n_in, n_hidden, n_batch=1, init_state_params=True, activation_fn=T.tanh):
        self.W = theano.shared(sample_weights(n_in, 3*n_hidden))
        self.T = theano.shared(sample_weights(n_hidden, 3*n_hidden))
        self.b = theano.shared(np.zeros(3*n_hidden, dtype=dtype))

        self.params = [self.W,
                        self.T,
                        self.b]

        self.h0 = theano.shared(np.zeros((n_batch, n_hidden), dtype=dtype))
        self.c0 = theano.shared(np.zeros((n_batch, n_hidden), dtype=dtype))
        if init_state_params:
            self.params += [self.h0, self.c0]

        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_batch = n_batch
        self.activation_fn = activation_fn

    def step(self, x_t, h_tm1, c_tm1):
        M = x_t.dot(self.W) + h_tm1.dot(self.T) + self.b
        i_t = T.nnet.sigmoid(M[:, 0:self.n_hidden])
        o_t = T.nnet.sigmoid(M[:, self.n_hidden:2*self.n_hidden])
        c_t = (1.-i_t)*c_tm1 + i_t*self.activation_fn(M[:, 2*self.n_hidden:])
        h_t = o_t*self.activation_fn(c_t)
        return h_t, c_t

    def apply(self, v):
        if self.n_batch == 1:
            h_init = [T.tile(self.h0, (v.shape[1], 1)),
                        T.tile(self.c0, (v.shape[1], 1))]
        else:
            h_init = [self.h0, self.c0]

        [h_vals, _], _ = theano.scan(fn=self.step, 
                                        sequences = v, 
                                        outputs_info = h_init)
        return h_vals