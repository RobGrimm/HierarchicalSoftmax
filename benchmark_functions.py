import os
import time
import numpy
import theano

import theano.tensor as T

from HierarchicalSoftmax import HierarchicalSoftmax
from Softmax import Softmax

from matplotlib import pyplot
# set parameters for plots
pyplot.rcParams.update({'figure.figsize': (25, 20), 'font.size': 25})


########################################################################################################################

# helper functions for plotting

def save_plot_to(plot_dir, plot_name):
    pyplot.savefig(plot_dir + plot_name, additional_artists=get_paras_for_centering_legend_below_plot(),
                   bbox_inches='tight')
    pyplot.close()


def get_paras_for_centering_legend_below_plot():
    # get matplotlib parameters for centering the legend below plots
    pyplot.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
    lgd = pyplot.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
    art = [lgd]
    return art


########################################################################################################################


def generate_data(n_classes, n_training_examples, input_size):
    """
    Generate dummy training data.

    Arguments:
        - n_classes: how many output classes there should be in the data set
        - n_training_examples: how many training examples there should be
        - input_size: length of each input vector

    Returns:
        - train_set_x: array of input vectors
        - train_set_y: array of integer classes, to be predicted from vectors in 'train_set_x'
    """
    numpy.random.seed(123)
    train_set_x = [numpy.random.rand(input_size) for i in range(n_training_examples)]

    # balance training data for class
    # if training data cannot evenly be divided by number of classes,
    # assign class 0 to the remaining data
    interval = n_training_examples / n_classes
    remainder = n_training_examples % n_classes
    train_set_y = [i for j in range(interval) for i in range(n_classes)] + [0 for j in range(remainder)]

    assert len(train_set_x) == len(train_set_y)

    train_set_x = theano.shared(numpy.asarray(train_set_x, dtype=theano.config.floatX), borrow=True)
    train_set_y = theano.shared(numpy.asarray(train_set_y, dtype=theano.config.floatX), borrow=True)
    train_set_y = T.cast(train_set_y, 'int32')

    return train_set_x, train_set_y


def generate_data_train_softmax(n_classes, n_training_examples, input_size, n_epochs, learning_rate=0.1, batch_size=10,
                                hierarchical=False):
    """
    Train either a flat or hierarchical softmax model on randomly generated data and return the predicted class for a
    single random test example, the average training loss at the last epoch,
    and the time it took to train the model.

    Arguments:
         - n_classes:           how many output classes there should be in the randomly generated data set
         - n_training_examples: how many training examples there should be
         - input_size:          length of each randomly generated input vector
         - n_epochs:            number training epochs
         - learning_rate:       learning rate of the softmax model
         - batch_size:          batch size for softmax model
         - hierarchical:        whether to generate_data_train_softmax with hierachical softmax
                                (use flat softmax otherwise)

    Returns:
        - predicted: the predicted class for a single randomly generated test example
        - avg_loss: the average training loss over generate_data_train_softmax batches at the last training epoch
        - total_train_time: the time it took, in minutes, to generate_data_train_softmax the model
    """
    train_set_x, train_set_y = generate_data(n_classes, n_training_examples, input_size)
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    print 'we have %s train batches' % n_train_batches

    # allocate symbolic variables for the data
    index = T.lscalar()     # index to a [mini]batch
    x = T.matrix('x')       # data, presented as rasterized images
    y = T.ivector('y')      # labels, presented as 1D vector of [int] labels

    # instantiate hierarchical softmax model and calculate gradients
    if hierarchical:

        softmax = HierarchicalSoftmax(input_=x, target=y, n_in=input_size, n_out=n_classes)
        cost = softmax.cost

        g_W = T.grad(cost=cost, wrt=softmax.W1)
        g_b = T.grad(cost=cost, wrt=softmax.b1)
        g_U = T.grad(cost=cost, wrt=softmax.W2)
        g_c = T.grad(cost=cost, wrt=softmax.b2)

        updates = [(softmax.W1, softmax.W1 - learning_rate * g_W),
                   (softmax.b1, softmax.b1 - learning_rate * g_b),
                   (softmax.W2, softmax.W2 - learning_rate * g_U),
                   (softmax.b2, softmax.b2 - learning_rate * g_c)]

    # instantiate flat softmax model and calculate gradients
    else:
        softmax = Softmax(input_=x, n_in=input_size, n_out=n_classes)
        cost = softmax.negative_log_likelihood(y)

        g_W = T.grad(cost=cost, wrt=softmax.W)
        g_b = T.grad(cost=cost, wrt=softmax.b)

        updates = [(softmax.W, softmax.W - learning_rate * g_W),
                   (softmax.b, softmax.b - learning_rate * g_b)]

    # compile a Theano function `train_model` that returns the cost and at
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # train the model
    start_time = time.time()
    avg_loss = None
    for epoch in range(n_epochs):

        costs_over_batches = []
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            costs_over_batches.append(minibatch_avg_cost)

        avg_loss = numpy.mean(costs_over_batches)

        print 'Epoch: %s' % epoch
        print 'Loss: %s' % avg_loss
        print 'Time since beginning of training: %s' % ((time.time() - start_time) / 60)
        print

    total_train_time = ((time.time() - start_time) / 60)
    print 'Training took: %s' % total_train_time
    print '\n\n'

    # generate a single random test example
    numpy.random.seed(444)
    input_ = numpy.asarray([numpy.random.rand(input_size)])

    predictions = softmax.get_predictions(input_)

    # compute class prediction
    preds_eval = predictions.eval()

    return preds_eval, avg_loss, total_train_time


def benchmark_softmax(n_classes_range, plot_name, n_data_points=50000, input_size=3, n_epochs=2):
    """
    Train a flat and a hierarchical softmax model for a range of different numbers of output classes.
    Then create three plots:

        - training time as a function of number of output classes
        - the predicted class as a function of number of classes (predicted class by model)
        - training loss at last epoch as a function of number of output classes

    The training time should increase linearly as a function of number of output classes for the flat model,
    whereas it should increase much less for the hierarchical model. Predicted classes should be equivalent most of
    the time. The training loss may differ across flat and hierarchical softmax.

    Arguments:
        - n_classes_range:  list of numbers of classes for which to generate_data_train_softmax models
        - plot_name:        name of plots -- will be saved to os.getcwd() + '/plots/'
    """
    def get_benchmark_data(hierarchical=False):
        costs = []
        times = []
        preds = []
        for n_classes in n_classes_range:
            print 'training %s softmax model with %s classes' % ('hierarchical' if hierarchical else 'flat', n_classes)
            print
            pred, cost, time = generate_data_train_softmax(n_classes, n_data_points, input_size, n_epochs,
                                                           hierarchical=hierarchical)
            costs.append(cost)
            times.append(time)
            preds.append(pred)
        return costs, times, preds

    def plot_benchmark(xs, hierarchical_ys, flat_ys, y_axis_label):
        pyplot.plot(xs, hierarchical_ys, 'o-', markersize=40, linewidth=9, label='hierarchical softmax')
        pyplot.plot(xs, flat_ys, 'o-', markersize=40, linewidth=9, label='flat softmax')
        pyplot.xlabel('nr of output classes', fontsize=40)
        pyplot.ylabel(y_axis_label, fontsize=40)
        plot_dir = os.getcwd() + '/plots/'
        save_plot_to(plot_dir, '%s_%s' % (plot_name, y_axis_label))

    # generate_data_train_softmax hierarchical softmax models
    h_cost, h_time, h_preds = get_benchmark_data(hierarchical=True)

     # generate_data_train_softmax flat softmax models
    f_cost, f_time, f_preds = get_benchmark_data(hierarchical=False)

    #plot results
    plot_benchmark(n_classes_range, h_cost, f_cost, 'cost')
    plot_benchmark(n_classes_range, h_time, f_time, 'time')
    plot_benchmark(n_classes_range, h_preds, f_preds, 'predicted_class')