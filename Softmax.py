import numpy
import theano
import theano.tensor as T


class Softmax(object):
    """
    Softmax layer. Code is based on the Logistic Regression used in the Theano deep learning tutorial:
    http://deeplearning.net/tutorial/code/logistic_sgd.py
    """

    def __init__(self, input_, n_in, n_out):
        """
        :type input_:   theano.tensor.TensorType
        :param input_:  symbolic variable that describes the input (one minibatch)

        :type n_in:     int
        :param n_in:    number of input units

        :type n_out:    int
        :param n_out:   number of output units
        """
        self.input = input_

        self.W = theano.shared(value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX), name='W', borrow=True)
        self.b = theano.shared(value=numpy.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)

        self.p_y_given_x = T.nnet.softmax(T.dot(input_, self.W) + self.b)
        self.params = [self.W, self.b]


    def get_predictions(self, input_):
        return T.argmax(T.nnet.softmax(T.dot(input_, self.W) + self.b), axis=1)


    def negative_log_likelihood(self, y):
        cost = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        return cost