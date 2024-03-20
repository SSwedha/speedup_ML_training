# Speed up ML training on Pytorch (and Tensorflow)
Training ML models (say a few million trainable parameters) on small to medium-sized datasets (order of GB) over 100 epochs should not typically take more than a few hours. One of the main bottlenecks in training time is how data is handled. 

For example, when the training loss is back propagated to update the model weights, the next batch of data can already be fetched. This way, we can parallelize such tasks. 
