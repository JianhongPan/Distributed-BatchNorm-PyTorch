# Distributed-BatchNorm-PyTorch

Distributed Batch Normalization implementation in PyTorch.

This module simulates the built-in PyTorch BatchNorm in distributed training
where the mean and standard deviation are reduced individually on each virtual device.

For example, most of the papers distributedly training the networks, 
and the implementation of BatchNorm individually calculate the mean and standard deviation
on each device and broadcast the running mean and running standard of the first GPU to other devices.
It is often a positive impact on the final result of the network after training with large batch size. 
However, the limit on the number of GPUs does not allow some researchers to experiment with distributed training, which might cause them to be unable to replicate some works.

To simulate the BatchNorm in distributed training, this Distributed BatchNorm uses various BatchNorm modules (with the same learnable parameters)
to split one mini-batch into several virtual mini-batches and process them independently. 

This module is currently only a prototype version for research usages. As mentioned below,
it has its limitations and may even suffer from some design problems. If you have any
questions or suggestions, please feel free to
[open an issue](https://github.com/PoonKinWang/Distributed-BatchNorm-PyTorch/issues) or 
[submit a pull request](https://github.com/PoonKinWang/Distributed-BatchNorm-PyTorch/pulls).

## Why Distributed BatchNorm?

Although training network on one GPU or using the synchronized implementation of BatchNorm on multiple devices (GPUs)
ensure the consistencies of mean and standard deviation, it might degenerate the network performance after the training of large batch size (e.g., 256).
This is a significant issue in some standard vision tasks such as ImageNet classification leading to an unexpected result. 
The number of GPUs restricts BatchNorm distribution, which makes it difficult to achieve the result of the original peper (e.g., training ResNeXT on 8 GPUS). 

## Usage
We use Function Object to make DistributedBatchNorm easy to be adopted. 
To use the Distributed Batch Normalization:

```python
from DistributedBatchNorm.py import BatchNorm1d as DistributedBatchNorm2d

BatchNorm2d = DistBatchNorm2d(vt_world_size=8)
bn1 = BatchNorm2d(num_features=16)
bn2 = BatchNorm2d(num_features=32)
bn3 = BatchNorm2d(num_features=64)
```

Here, argument `vt_world_size` denotes the number of virtual GPUs (simulating) on each physical GPU. 

It can displace all the normal BatchNorm in the model from TorchVision by a simple step:

```python
from torchvision.models.resnet import resnet50
from DistributedBatchNorm.py import BatchNorm1d as DistributedBatchNorm2d

BatchNorm2d = DistBatchNorm2d(vt_world_size=8)
model = resnet50(norm_layer=BatchNorm2d)
```

See more details in examples.py.