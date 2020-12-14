# Distributed-BatchNorm-PyTorch

Distributed Batch Normalization implementation in PyTorch.

This module simulates the built-in PyTorch BatchNorm in distributed training
where the mean and standard-deviation are reduced individually on each device.

For example, when one uses `nn.DataParallel` to wrap the network during
training, PyTorch's implementation normalize the tensor on each device using
the statistics only on that device, which accelerated the computation and
is also easy to implement, but the statistics might be inaccurate.
Instead, in this synchronized version, the statistics will be computed
over all training samples distributed on multiple devices.

Note that, for one-GPU or CPU-only case, this module behaves exactly same
as the built-in PyTorch implementation.

This module is currently only a prototype version for research usages. As mentioned below,
it has its limitations and may even suffer from some design problems. If you have any
questions or suggestions, please feel free to
[open an issue](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch/issues) or 
[submit a pull request](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch/issues).
