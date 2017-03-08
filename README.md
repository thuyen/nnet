# nnet
A collection of tools for training neural networks

  - GpuLoader: a GPU feature extraction buffer
  - ImageData: a flexible image dataset
  - BDataLoader: simliar to DataLoader class but initialize the iterator in 
    advance
  - PEDataLoader: a multiprocess-dataloader that parallels over elements 
    as suppose to over batches (the torch built-in one)
