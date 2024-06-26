# Speed up ML training on Pytorch (and Tensorflow)
Training ML models (say a few million trainable parameters) on small to medium-sized datasets (order of GB) over 100 epochs should not typically take more than a few hours. One of the main bottlenecks in training time is how data is handled. 

In the subsequent sections, we will look into various techniques that enable faster data handling and prevent CUDA from running into "Out of memory" errors (if you use PyTorch and Nvidia's GPU). 
tensorflow.data API has a very documentation on explaining the concepts behind such techniques which can be found [here](https://www.tensorflow.org/guide/data_performance). I highly recommend the reader check it out even if they do not work with Tensorflow. Since I use PyTorch for my projects, I will spend more time on it. 

## Pre-fetch data
For example, when the training loss is back propagated to update the model weights, the subsequent batch(es) of data can already be fetched. This is called pre-fetching which can be visualized in the following two images. 
![image](https://github.com/SSwedha/speedup_ML_training/assets/38497040/e5a7e9fc-1479-464a-b644-f8086b308967)
![image](https://github.com/SSwedha/speedup_ML_training/assets/38497040/8fa6a469-6465-4cee-bfad-668af34d4b86)
source: https://www.tensorflow.org/guide/data_performance

You can set ```prefetch_factor``` in PyTorch's Dataloader class by changing from the default value of 2 to say 4. [(PyTorch Dataloader documentation)](https://pytorch.org/docs/stable/data.html)
```python
custom_dataset = CustomDataset(*args)
dataloader = torch.utils.data.Dataloader(custom_dataset, batch_size=128, shuffle=True, prefetch_factor=4)
```
Notice that we have set _prefetch_factor_ to 4 instead of the default value of 2. You should be able to notice the model taking less time to complete one epoch of training. You can check this by either printing the time it takes to complete one epoch or on your training progress bar. 

### Compute time calculations
In case you would like to profile your code to identify bottlenecks in your model training/inference pipeline, you can use [PyTorch's profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html). However, a naive approach would be to time every function call. We can achieve this by using ```time.perf_counter()```. It returns a value (in fractional seconds) that does not have a reference from a performance counter to time short duration. Thus, only the difference between two calls of ```time.perf_counter()``` would be relevant.

``` python
for epoch in tqdm.tqdm(range(n_epoch)): # tqdm.tqdm prints a progress bar of your training
  start = time.perf_counter()
  for batch_num, (data, ground_truth) in enumerate(dataloader)):
    # forward pass data to model
    # compute loss and backpropagate it
    # update model weights
  end = perf_counter()
  print("Epoch {epoch+1} took {(end-start)}s") # this is not the actual syntax
```

## Multi-process data loading
Instead of having the main process handle data loading, we can instead create multiple sub-processes that will fetch data. This can be enabled by setting ```num_workers``` > 1. Each worker gets PyTorch's dataset class instance,```collate_fn```, and ```worker_init_fn```to initialize and fetch data whenever ```enumerate(dataloader)``` is called. If you are using iterable-style dataset, caution must be ensured that replicas of the same data are not made. For more details on multi-process data loading, click [here][https://pytorch.org/docs/stable/data.html#multi-process-data-loading].

One must be cautious when setting ```num_workers``` > 1 while using Windows to avoid the re-creation of multiple instances of the dataset class. Thus, create object instances of the dataset and dataloader within ``` if __name__ == '__main__':``` so that the workers do not execute them again. If your dataset class looks something like this:
``` python
class CustomDataset(torch.utils.data.Dataset):
  def __init__(self, data_path):
    self.data_path = data_path
    self.data = load_data(data_path) # some function that loads the data

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]
```
Notice that we are loading the data while initializing an instance of the custom dataset class. Alternatively, one can fetch the necessary data only when ```__getitem__ ``` is called. Such a code would be like this:

``` python
class CustomDataset(torch.utils.data.Dataset):
  def __init__(self, data_path):
    self.data_path = data_path
    self.filenames = get_filenames(data_path) # some function that grabs all filenames ending in a specific file format

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    data = load_data(self.filenames[idx])
    return data
```
In this case, we are only loading the filenames when an instance of the dataset class is created. However, this approach works when we all files have a fixed length of data (say the filenames are images of size 64 x 64). However, if you need to load the entire data as the files are of varying lengths or you are segmenting the data into blocks of data based on some parameter, then the former class definition could work better. (For example, say there are audio files of varying time lengths (order of mins) and that you are interested in analyzing time segments of say 15s.)

So when each worker is called, copies of the dataset will be created for the worker to fetch batches of data. This could lead to RAM running into out of memory issues. You can check this by opening the task manager.
