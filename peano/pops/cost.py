import theano
import theano.tensor as T

def squared_hinge(y_true, y_pred):
    return 0.5*T.sqr(T.maximum(1. - y_true*y_pred, 0.)).mean()

def cross_entropy(y_true, y_pred):
    return T.nnet.categorical_crossentropy(y_pred, y_true).mean()

def mean_squared_error(y_true, y_pred):
    return 0.5*T.sqr(y_pred - y_true).mean()

def xcov(actset_1, actset_2):
    N = actset_1.shape[0].astype(theano.config.floatX)
    actset_1 = actset_1-actset_1.mean(axis=0, keepdims=True)
    actset_2 = actset_2-actset_2.mean(axis=0, keepdims=True)
    cc = T.dot(actset_1.T, actset_2)/N
    cost = .5*T.sqr(cc).mean()
    return cost