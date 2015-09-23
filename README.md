This is a version of hierarchical softmax that is based on an implementation found here: https://github.com/lisa-groundhog/GroundHog

benchmark_functions.py contains functionality for training flat and hierarchical softmax models on randomly generated data, then comparing the models in terms of (1) predictions on unseen data, (2) training loss, and (3) runtime. Look at run.py for some examples. 

Dependencies:

- Theano (0.7.0)
- numpy (1.9.2)
- matplotlib (1.4.3)
