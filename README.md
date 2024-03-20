# Speed up ML training on Pytorch (and Tensorflow)
Training ML models (say a few million trainable parameters) on small to medium-sized datasets (order of GB) over 100 epochs should not typically take more than a few hours. One of the main bottlenecks in training time is how data is handled. 

In the subsequent sections, we will look into various techniques that enable faster data handling and prevent CUDA from running into "Out of memory" errors (if you use PyTorch and Nvidia's GPU). 
tensorflow.data API has a very documentation on explaining the concepts behind such techniques which can be found [here](https://www.tensorflow.org/guide/data_performance). I highly recommend the reader to check it out even if they do not work with Tensorflow. Since I use PyTorch for my projects, I will spend more time on it. 

# Pre-fetch data
For example, when the training loss is back propagated to update the model weights, the next batch of data can already be fetched. This is 
