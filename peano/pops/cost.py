import theano
import theano.tensor as T

def squared_hinge(y_true, y_pred):
    return 0.5*T.sqr(T.maximum(1. - y_true*y_pred, 0.)).mean()

def cross_entropy(y_true, y_pred):
    return T.nnet.categorical_crossentropy(y_pred, y_true).mean()

def cross_entropy_logdomain(y_true, log_y_pred):
    return -(y_true*log_y_pred).mean()

def mean_squared_error(y_true, y_pred):
    return 0.5*T.sqr(y_pred - y_true).mean()

def huber_loss(y_true, y_pred, delta=1.):
    a = y_true - y_pred
    squared_loss = 0.5*T.sqr(a)
    absolute_loss = delta*T.abs(a) - 0.5*T.sqr(delta)

    cost = T.switch(T.lt(T.abs(a), delta),
                    squared_loss,
                    absolute_loss)
    return cost

def misclass_error(y_true, y_pred):
    return T.neq(T.argmax(y_true, axis=1), T.argmax(y_pred, axis=1)).mean()

def xcov(actset_1, actset_2):
    N = actset_1.shape[0].astype(theano.config.floatX)
    actset_1 = actset_1-actset_1.mean(axis=0, keepdims=True)
    actset_2 = actset_2-actset_2.mean(axis=0, keepdims=True)
    cc = T.dot(actset_1.T, actset_2)/N
    cost = .5*T.sqr(cc).mean()
    return cost