import numpy
import theano
import theano.tensor as T


class HierarchicalSoftmax(object):
    """
    2-level Hierarchical Softmax layer. Adapted from the Hierarchical Softmax layer from the lisa-groundhog package:
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

        # output layer is a 2-level graph
        # a predicted class label is defined as a a fixed arbitrary path through this graph
        # we thus need at least sqrt(n_out) nodes in the first level
        # (ceil of the scalar x is the smallest integer i, such that i >= x)
        self.n_level1_nodes = numpy.ceil(numpy.sqrt(n_out)).astype('int64')
        # and at most sqrt(n_out) nodes in the second level -- note that sometimes we may end up
        # with a graph that has a few more possible paths than there are output classes
        self.n_level2_nodes = numpy.ceil(n_out/float(self.n_level1_nodes)).astype('int64')

        # define weight matrix 'W1' and bias 'b1' for first level in output graph
        self.W1 = theano.shared(value=numpy.zeros((n_in,  self.n_level1_nodes), dtype=theano.config.floatX),
                                name='W1', borrow=True)
        self.b1 = theano.shared(value=numpy.zeros((self.n_level1_nodes,), dtype=theano.config.floatX),
                                name='b1', borrow=True)

         # define weight matrix 'W2' and bias 'b2' for second level in output graph
        self.W2 = theano.shared(value=numpy.zeros((n_in,  self.n_level2_nodes), dtype=theano.config.floatX),
                                name='W2', borrow=True)
        self.b2 = theano.shared(value=numpy.zeros((self.n_level2_nodes,), dtype=theano.config.floatX),
                                name='b2', borrow=True)

        self.params = [self.W1, self.b1, self.W2, self.b2]

        self.p_y_given_x = self.forward_prop(input_, target)
        self.cost = -T.mean(T.log(self.p_y_given_x)[T.arange(target.shape[0]), target])


    def get_predictions(self, input_):
        return T.argmax(self.forward_prop(input_), axis=1)


    def forward_prop(self, input_, targets=None):
        """
        If target is 'None', compute the probability of taking the correct path through the output graph.
        Else, compute the probability for each possible path (= each possible output class).
        """
        level1_vals = T.nnet.softmax(T.dot(input_, self.W1) + self.b1)
        level2_vals = T.nnet.softmax(T.dot(input_, self.W2) + self.b2)

        batch_size = input_.shape[0]

        # compute all possible predictions [ time complexity is O(n_out) ]
        if targets is None:

            def _path_probas(idx):
                lev1_vec, lev2_vec = level1_vals[idx], level2_vals[idx]
                result, updates = theano.scan(fn=lambda k, array_: k * array_,
                                              sequences=lev1_vec,
                                              non_sequences=lev2_vec)
                return result.flatten()

            output_, updates = theano.scan(fn=_path_probas, sequences=T.arange(batch_size))

            # since we may have more possible paths through the graph than output classes,
            # ignore the remaining paths
            output_ = output_[:, :self.n_out]

        # compute only batch_size predictions [ time complexity is O(2 x sqrt(n_out)) = O(sqrt(n_out)) ]
        else:
            # to each class label, assign a pair of nodes in layer1 and layer2 of the output graph
            level1_idx = targets // self.n_level1_nodes
            level2_idx = targets % self.n_level2_nodes

            # calculate probability of taking correct path through the graph
            level1_val = level1_vals[T.arange(batch_size), level1_idx]
            level2_val = level2_vals[T.arange(batch_size), level2_idx]
            target_probas = level1_val * level2_val

            # output is a matrix of predictions, with dimensionality (batch_size, n_out)
            # since we only have a probability for the correct label,
            #  we assign a probability of zero to all other labels
            output_ = T.zeros((batch_size, self.n_out))
            output_ = T.set_subtensor(output_[T.arange(batch_size), targets], target_probas)

        return output_