from benchmark_functions import generate_data_train_softmax
from benchmark_functions import benchmark_softmax

# train hierarchical softmax on dataset with 5.000 output classes
generate_data_train_softmax(n_classes=5000, n_training_examples=50000, input_size=3, hierarchical=True, n_epochs=2)

# train flat softmax on the same data -- this will take longer
generate_data_train_softmax(n_classes=5000, n_training_examples=50000, input_size=3, hierarchical=False, n_epochs=2)


# train hierarchical softmax on data set with 1 million output classes
# will take much longer if you do it with flat softmax
generate_data_train_softmax(n_classes=1000000, n_training_examples=50000, input_size=3, hierarchical=True, n_epochs=2)


# for varying numbers of output classes (500 -- 200.000, in increments of 500),
# train both a flat and a hierarchical softmax model. keep track of the total
# training time and the training loss at the last epoch for each model.
# then plot cost and training time as a function of number of output classes, as well as the predicted class
# for a randomly generated test example.
# save plots to os.getcwd() + '\plots\'
benchmark_softmax(n_classes_range=range(500, 20500, 500), plot_name='500--20k', input_size=100, n_data_points=50000,
                  n_epochs=2)