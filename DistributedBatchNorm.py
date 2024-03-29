import torch
from torch import Tensor
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init

from typing import Optional, Any
class DistributedBatchNorm1d():
    r"""
    BatchNorm1d Generator. 
    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, L)` or :math:`L` from input of size :math:`(N, L)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``
    """
    def __init__(self, vt_world_size, accumulation_steps=1):
        """vt_world_size: virtual_world_size"""
        if (type(vt_world_size) is int and vt_world_size > 0):
            print("virtual world size: "+str(vt_world_size))
            print("accumulation steps: "+str(accumulation_steps))
        else:
            raise ValueError("argument 'vt_world_size' must be positive integer (got "+str(vt_world_size)+").")
        self.vt_world_size = vt_world_size
        self.accumulation_steps = accumulation_steps
    def __call__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        r"""Applies Batch Normalization over a 2D or 3D input (a mini-batch of 1D
            inputs with optional additional channel dimension) as described in the paper
            `Batch Normalization: Accelerating Deep Network Training by Reducing
            Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

            .. math::

                y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

            The mean and standard-deviation are calculated per-dimension over
            the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
            of size `C` (where `C` is the input size). By default, the elements of :math:`\gamma` are set
            to 1 and the elements of :math:`\beta` are set to 0. The standard-deviation is calculated
            via the biased estimator, equivalent to `torch.var(input, unbiased=False)`.

            Also by default, during training this layer keeps running estimates of its
            computed mean and variance, which are then used for normalization during
            evaluation. The running estimates are kept with a default :attr:`momentum`
            of 0.1.

            If :attr:`track_running_stats` is set to ``False``, this layer then does not
            keep running estimates, and batch statistics are instead used during
            evaluation time as well.

            .. note::
                This :attr:`momentum` argument is different from one used in optimizer
                classes and the conventional notion of momentum. Mathematically, the
                update rule for running statistics here is
                :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
                where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
                new observed value.

            Because the Batch Normalization is done over the `C` dimension, computing statistics
            on `(N, L)` slices, it's common terminology to call this Temporal Batch Normalization.
                return BatchNorm1d(vt_world_size=self.vt_world_size, **kwargs)
            
            Shape:
                - Input: :math:`(N, C)` or :math:`(N, C, L)`
                - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

            Examples::

                >>> # With Learnable Parameters
                >>> m = nn.BatchNorm1d(100)
                >>> # Without Learnable Parameters
                >>> m = nn.BatchNorm1d(100, affine=False)
                >>> input = torch.randn(20, 100)
                >>> output = m(input)
        """
        return BatchNorm1d(num_features, eps, momentum, affine,
                 track_running_stats, vt_world_size=self.vt_world_size, accumulation_steps=self.accumulation_steps)

class DistributedBatchNorm2d():
    r"""
    BatchNorm2d Generator. 
    Args:
        vt_world_size: virtual_world_size
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``
    """
    def __init__(self, vt_world_size, accumulation_steps=1):
        """vt_world_size: virtual_world_size"""
        if (type(vt_world_size) is int and vt_world_size > 0):
            print("virtual world size: "+str(vt_world_size))
            print("accumulation steps: "+str(accumulation_steps))
        else:
            raise ValueError("argument 'vt_world_size' must be positive integer (got "+str(vt_world_size)+").")
        self.vt_world_size = vt_world_size
        self.accumulation_steps = accumulation_steps

    def __call__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        r"""Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs
            with additional channel dimension) as described in the paper
            `Batch Normalization: Accelerating Deep Network Training by Reducing
            Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

            .. math::

                y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

            The mean and standard-deviation are calculated per-dimension over
            the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
            of size `C` (where `C` is the input size). By default, the elements of :math:`\gamma` are set
            to 1 and the elements of :math:`\beta` are set to 0. The standard-deviation is calculated
            via the biased estimator, equivalent to `torch.var(input, unbiased=False)`.

            Also by default, during training this layer keeps running estimates of its
            computed mean and variance, which are then used for normalization during
            evaluation. The running estimates are kept with a default :attr:`momentum`
            of 0.1.

            If :attr:`track_running_stats` is set to ``False``, this layer then does not
            keep running estimates, and batch statistics are instead used during
            evaluation time as well.

            .. note::
                This :attr:`momentum` argument is different from one used in optimizer
                classes and the conventional notion of momentum. Mathematically, the
                update rule for running statistics here is
                :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
                where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
                new observed value.

            Because the Batch Normalization is done over the `C` dimension, computing statistics
            on `(N, H, W)` slices, it's common terminology to call this Spatial Batch Normalization.

            Shape:
                - Input: :math:`(N, C, H, W)`
                - Output: :math:`(N, C, H, W)` (same shape as input)

            Examples::

                >>> # With Learnable Parameters
                >>> m = nn.BatchNorm2d(100)
                >>> # Without Learnable Parameters
                >>> m = nn.BatchNorm2d(100, affine=False)
                >>> input = torch.randn(20, 100, 35, 45)
                >>> output = m(input)
        """
        return BatchNorm2d(num_features, eps, momentum, affine,
                 track_running_stats, vt_world_size=self.vt_world_size, accumulation_steps=self.accumulation_steps)

class DistributedBatchNorm3d():
    r"""
    BatchNorm3d Generator. 
    Args:
        vt_world_size: virtual_world_size
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, D, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``
    """
    def __init__(self, vt_world_size, accumulation_steps=1):
        """vt_world_size: virtual_world_size"""
        if (type(vt_world_size) is int and vt_world_size > 0):
            print("virtual world size: "+str(vt_world_size))
            print("accumulation steps: "+str(accumulation_steps))
        else:
            raise ValueError("argument 'vt_world_size' must be positive integer (got "+str(vt_world_size)+").")
        self.vt_world_size = vt_world_size
        self.accumulation_steps = accumulation_steps
    def __call__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        r"""Applies Batch Normalization over a 5D input (a mini-batch of 3D inputs
            with additional channel dimension) as described in the paper
            `Batch Normalization: Accelerating Deep Network Training by Reducing
            Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

            .. math::

                y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

            The mean and standard-deviation are calculated per-dimension over
            the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
            of size `C` (where `C` is the input size). By default, the elements of :math:`\gamma` are set
            to 1 and the elements of :math:`\beta` are set to 0. The standard-deviation is calculated
            via the biased estimator, equivalent to `torch.var(input, unbiased=False)`.

            Also by default, during training this layer keeps running estimates of its
            computed mean and variance, which are then used for normalization during
            evaluation. The running estimates are kept with a default :attr:`momentum`
            of 0.1.

            If :attr:`track_running_stats` is set to ``False``, this layer then does not
            keep running estimates, and batch statistics are instead used during
            evaluation time as well.

            .. note::
                This :attr:`momentum` argument is different from one used in optimizer
                classes and the conventional notion of momentum. Mathematically, the
                update rule for running statistics here is
                :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
                where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
                new observed value.

            Because the Batch Normalization is done over the `C` dimension, computing statistics
            on `(N, D, H, W)` slices, it's common terminology to call this Volumetric Batch Normalization
            or Spatio-temporal Batch Normalization.

            Shape:
                - Input: :math:`(N, C, D, H, W)`
                - Output: :math:`(N, C, D, H, W)` (same shape as input)

            Examples::

                >>> # With Learnable Parameters
                >>> m = nn.BatchNorm3d(100)
                >>> # Without Learnable Parameters
                >>> m = nn.BatchNorm3d(100, affine=False)
                >>> input = torch.randn(20, 100, 35, 45, 10)
                >>> output = m(input)
            """
        return BatchNorm3d(num_features, eps, momentum, affine,
                 track_running_stats, vt_world_size=self.vt_world_size, accumulation_steps=self.accumulation_steps)

class _NormBase(Module):
    """Common base of _InstanceNorm and _BatchNorm"""
    _version = 2
    __constants__ = ['track_running_stats', 'momentum', 'eps',
                     'num_features', 'affine']
    num_features: int
    eps: float
    momentum: float
    affine: bool
    track_running_stats: bool
    # WARNING: weight and bias purposely not defined here.
    # See https://github.com/pytorch/pytorch/issues/39670

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True
    ) -> None:
        super(_NormBase, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(_NormBase, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class _BatchNorm(_NormBase):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, vt_world_size=1, accumulation_steps=1):
        self.vt_world_size = vt_world_size
        self.accumulation_steps = accumulation_steps
        self.accumulation_step = 0
        super(_BatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        output = []
        vt_batch_size = (input.shape[0] + self.vt_world_size - 1) // self.vt_world_size # vt_batch_size = ceil(input.shape[0] / self.world_size)
        for vt_gpu_index in range(self.vt_world_size):
            # exponential_average_factor is set to self.momentum
            # (when it is available) only so that it gets updated
            # in ONNX graph when this node is exported to ONNX.
            if self.momentum is None:
                exponential_average_factor = 0.0
            else:
                exponential_average_factor = self.momentum

            if self.training and self.track_running_stats:
                # TODO: if statement only here to tell the jit to skip emitting this when it is None
                if self.num_batches_tracked is not None:
                    if vt_gpu_index == 0:
                        self.num_batches_tracked = self.num_batches_tracked + 1
                    if self.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = self.momentum

            r"""
            Decide whether the mini-batch stats should be used for normalization rather than the buffers.
            Mini-batch stats are used in training mode, and in eval mode when buffers are None.
            """
            if self.training:
                bn_training = True
            else:
                bn_training = (self.running_mean is None) and (self.running_var is None)

            r"""
            Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
            passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
            used for normalization (i.e. in eval mode when buffers are not None).
            """
            if vt_gpu_index == 0 and self.accumulation_step == 0:
                # save the running_mean and running_var before updated
                self.pre_running_mean = self.running_mean.clone()
                self.pre_running_var = self.running_var.clone()
                output += [F.batch_norm(
                    input[vt_batch_size*vt_gpu_index:vt_batch_size*(vt_gpu_index+1)],
                    # If buffers are not to be tracked, ensure that they won't be updated
                    self.running_mean if not self.training or self.track_running_stats else None,
                    self.running_var if not self.training or self.track_running_stats else None,
                    self.weight, self.bias, bn_training, exponential_average_factor, self.eps)]
            else:
                running_mean = self.pre_running_mean.clone()
                running_var = self.pre_running_var.clone()
                output += [F.batch_norm(
                    input[vt_batch_size*vt_gpu_index:vt_batch_size*(vt_gpu_index+1)],
                    # If buffers are not to be tracked, ensure that they won't be updated
                    running_mean if not self.training or self.track_running_stats else None,
                    running_var if not self.training or self.track_running_stats else None,
                    self.weight, self.bias, bn_training, exponential_average_factor, self.eps)]   

        self.accumulation_step = (self.accumulation_step + 1) % self.accumulation_steps
        return torch.cat(output, 0)

class BatchNorm1d(_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))


class BatchNorm2d(_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class BatchNorm3d(_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))

