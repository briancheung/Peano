import theano
import theano.tensor as T

def squared_hinge(y_true, y_pred, axis=None):
    return 0.5*T.sqr(T.maximum(1. - y_true*y_pred, 0.)).mean(axis=axis)

def cross_entropy(y_true, y_pred, axis=None):
    return T.nnet.categorical_crossentropy(y_pred, y_true).mean(axis=axis)

def cross_entropy_logdomain(y_true, log_y_pred, axis=None):
    return -(y_true*log_y_pred).mean(axis=axis)

def mean_squared_error(y_true, y_pred, axis=None):
    return 0.5*T.sqr(y_pred - y_true).mean(axis=axis)

def huber_loss(y_true, y_pred, delta=1., axis=None):
    a = y_true - y_pred
    squared_loss = 0.5*T.sqr(a)
    absolute_loss = (delta*abs(a) - 0.5*T.sqr(delta))

    cost = T.switch(T.le(abs(a), delta),
                    squared_loss,
                    absolute_loss)
    return cost.mean(axis=axis)

def misclass_error(y_true, y_pred, axis=None):
    return T.neq(T.argmax(y_true, axis=1), T.argmax(y_pred, axis=1)).mean(axis=axis)

def xcov(actset_1, actset_2, axis=None):
    N = actset_1.shape[0].astype(theano.config.floatX)
    actset_1 = actset_1-actset_1.mean(axis=0, keepdims=True)
    actset_2 = actset_2-actset_2.mean(axis=0, keepdims=True)
    cc = T.dot(actset_1.T, actset_2)/N
    cost = .5*T.sqr(cc).mean(axis=axis)
    return cost