# Distributed-BatchNorm-PyTorch

Distributed Batch Normalization implementation in PyTorch.

This module simulates the built-in PyTorch BatchNorm in distributed training
where the mean and standard-deviation are reduced individually on each device.

For example, when one uses `nn.DistributedDataParallel` (without `nn.SyncBatchNorm`) to wrap the network during
training, PyTorch's implementation of BatchNorm individually calculate the mean and standard-deviation
on each device, which results different mean and standard-deviation on each device 
and might affect the final result of the network after training.

To simulate the BatchNorm in distributed training, this distributed version uses various BatchNorm modules (with the same learnable parameters)
to process virtual mini batch. 

This module is currently only a prototype version for research usages. As mentioned below,
it has its limitations and may even suffer from some design problems. If you have any
questions or suggestions, please feel free to
[open an issue](https://github.com/PoonKinWang/Distributed-BatchNorm-PyTorch/issues) or 
[submit a pull request](https://github.com/PoonKinWang/Distributed-BatchNorm-PyTorch/pulls).

