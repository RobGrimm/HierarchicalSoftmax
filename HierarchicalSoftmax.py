import numpy
import theano
import theano.tensor as T


class HierarchicalSoftmax(object):
    """
    2-level Hierarchical Softmax layer. Adapted from the Hierarchical Softmax layer from the liza-groundhog package:
    https://github.com/lisa-groundhog/GroundHog
    """

    def __init__(self, input_, target, n_in, n_out):
        """
        :type input_:   theano.tensor.TensorType
        :param input_:  symbolic variable that describes the input (one minibatch)

        :type target:   theano.tensor.TensorType
        :param target:  symbolic variable that describes the out class (one minibatch)

        :type n_in:     int
        :param n_in:    number of input units

        :type n_out:    int
        :param n_out:   number of output units
        """
        self.n_out = n_out

        # output layer is a 2-level binary tree
        # a predicted class label is defined as a a fixed arbitrary path through this tree
        # we thus need sqrt(n_out) nodes in the first level of the tree
        self.n_level1_nodes = numpy.ceil(numpy.sqrt(n_out)).astype('int64')
        # and at most sqrt(n_out) nodes in the second level
        self.n_level2_nodes = numpy.ceil(n_out/float(self.n_level1_nodes)).astype('int64')

        # define weight matrix 'W' and bias 'b' for first layer in output tree
        self.W = theano.shared(value=numpy.zeros((n_in,  self.n_level1_nodes), dtype=theano.config.floatX),
                               name='W', borrow=True)
        self.b = theano.shared(value=numpy.zeros((self.n_level1_nodes,), dtype=theano.config.floatX),
                               name='b', borrow=True)

         # define weight matrix 'U' and bias 'c' for second layer in output tree
        self.U = theano.shared(value=numpy.zeros((n_in,  self.n_level2_nodes), dtype=theano.config.floatX),
                               name='W', borrow=True)
        self.c = theano.shared(value=numpy.zeros((self.n_level2_nodes,), dtype=theano.config.floatX),
                               name='b', borrow=True)

        self.params = [self.W, self.b, self.U, self.c]

        self.cost = -T.mean(T.log(self.forward_prop(input_, target)))


    def get_predictions(self, input_):
        return self.forward_prop(input_)


    def forward_prop(self, input_, target=None):
        """
        If target is 'None', compute the probability of taking the correct path through the output tree.
        Else, compute the probability for each possible path (= each possible output class).
        """
        # compute all possible predictions [ time complexity is O(n_out) ]
        if target is None:
            level1_vals = T.nnet.softmax(T.dot(input_, self.W) + self.b).flatten()
            level2_vals = T.nnet.softmax(T.dot(input_, self.U) + self.c).flatten()
            """works only for batch_size=1, at the moment.
             can do a nested scan to make it work for more than 1 mini batch """
            result, updates = theano.scan(fn=lambda k, array_: k * array_, sequences=level1_vals, non_sequences=level2_vals)
            output_ = result.flatten()[:self.n_out]

        # compute only batch_size predictions [ time complexity is O(2 x sqrt(n_out)) = O(sqrt(n_out)) ]
        else:
            # propagate input to level 1
            level1_vals = T.nnet.softmax(T.dot(input_, self.W) + self.b)

            # propagate input to level 2
            level2_vals = T.nnet.softmax(T.dot(input_, self.U) + self.c)

            # to each class label, assign a pair of nodes in layer1 and layer2 of the output tree
            level1_idx = target // self.n_level1_nodes
            level2_idx = target % self.n_level2_nodes

            # calculate cost of taking correct path through tree to
            bs = input_.shape[0]
            level1_val = level1_vals[T.arange(bs), level1_idx]
            level2_val = level2_vals[T.arange(bs), level2_idx]
            output_ = level1_val * level2_val

        return output_