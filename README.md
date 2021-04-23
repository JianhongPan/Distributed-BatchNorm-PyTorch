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

## Why Distributed BatchNorm?

Although training network on one GPU or using the synchronized implementation of BatchNorm on multiple devices (GPUs)
ensure the consistencies of mean and standard-deviation, it might degenerate the network performance after the training of large batch size (e.g., 256).
This is a significant issue in some standard vision tasks such as ImageNet classification, because in distributed training, the batch size per device
is usually large enough to obtain good statistics and more parameters to record mean and standard-deviation can save more details. 

However, the number of GPUs restricts BatchNorm distributed, which makes it difficult to achieve the original result (e.g., training ResNeXT on 8 GPUS). 

## Usage

To use the Distributed Batch Normalization:

```python
from DistributedBatchNorm.py import BatchNorm1d as DistBatchNorm1d

dist_bn = DistBatchNorm1d(128, vt_world_size=8)
```

Here, argument `vt_world_size` denotes the number of virtual GPUs (simulating) on each actual GPU. 
