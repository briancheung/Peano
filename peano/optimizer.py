from collections import OrderedDict
import numpy as np
import theano
import theano.tensor as T

dtype = theano.config.floatX

def sgd_update(params, gparams, learning_rate):
    learn_updates = OrderedDict()
    for p,g in zip(params, gparams):
        dx = -learning_rate*g
        learn_updates[p] = p + dx

    return learn_updates

def momentum_update(params, gparams, learning_rate, rho=.9):
    learn_updates = OrderedDict()

    for p,g in zip(params, gparams):
        v = theano.shared((p.get_value()*0.).astype(dtype))
        dx = -learning_rate*v
        learn_updates[v] = rho*v + (1.-rho)*g
        learn_updates[p] = p + dx

    return learn_updates

# TODO: Check difference between this momentuma and above
def ilya_momentum_update(params, gparams, learning_rate, rho=.9):
    learn_updates = OrderedDict()

    for p,g in zip(params, gparams):
        v = theano.shared((p.get_value()*0.).astype(dtype))
        learn_updates[v] = rho*v - learning_rate*g
        learn_updates[p] = p + v

    return learn_updates

def adadelta_update(params, gparams, rho=.95, epsilon=1e-6):
    learn_updates = OrderedDict()
    for p,g in zip(params, gparams):
        msg = theano.shared((p.get_value()*0.).astype(dtype))
        msdx = theano.shared((p.get_value()*0.).astype(dtype))

        msg_update = rho*msg + (1.-rho)*T.sqr(g)
        rms_msdx = T.sqrt(msdx + epsilon)
        rms_msg = T.sqrt(msg_update + epsilon)
        dx = -rms_msdx/rms_msg*g
        msdx_update = rho*msdx + (1.-rho)*T.sqr(dx)

        learn_updates[msg] = msg_update
        learn_updates[msdx] = msdx_update
        learn_updates[p] = p + dx

    return learn_updates

def adam_update(params, gparams, alpha=0.001, b1=0.9, b2=0.999, epsilon=1e-8):
    learn_updates = OrderedDict()

    t = theano.shared(np.array(1.,dtype=dtype))
    b1_t = b1**t
    b2_t = b2**t
    for p,g in zip(params, gparams):
        m = theano.shared((p.get_value()*0.).astype(dtype))
        v = theano.shared((p.get_value()*0.).astype(dtype))
        m_update = b1*m + (1.-b1)*g 
        v_update = b2*v + (1.-b2)*T.sqr(g)
        m_hat = m_update/(1.-b1_t)
        v_hat = v_update/(1.-b2_t)
        dx = -alpha*m_hat/(T.sqrt(v_hat)+epsilon)
        
        learn_updates[m] = m_update
        learn_updates[v] = v_update
        learn_updates[p] = p + dx
    learn_updates[t] = t + 1.

    return learn_updates

def clip_gradients(gparams, threshold=5.):
    clipped_gparams = []
    for gparam in gparams:
        norm_gparam = T.sqrt(T.sqr(gparam).sum())
        clipped_gparams.append(T.switch(T.le(norm_gparam, threshold),
                                        gparam,
                                        (gparam/norm_gparam)*threshold))

    return clipped_gparams