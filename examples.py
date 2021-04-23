import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50
from DistributedBatchNorm import DistributedBatchNorm1d, DistributedBatchNorm2d, DistributedBatchNorm3d

# the demo of DistributedBatchNorm1d
print("\033[35m"+"DistributedBatchNorm1d:"+"\033[0m")
BatchNorm1d = DistributedBatchNorm1d(vt_world_size=8)

bn = BatchNorm1d(32)
out = bn(torch.rand([8,32,1000]))
print(bn)
print(out.shape)
print("")

# the demo of DistributedBatchNorm2d
print("\033[35m"+"DistributedBatchNorm2d:"+"\033[0m")
BatchNorm2d = DistributedBatchNorm2d(vt_world_size=8)

bn = BatchNorm2d(32)
out = bn(torch.rand([8,32,7,7]))
print(bn)
print(out.shape)
print("")

# the demo of DistributedBatchNorm3d
print("\033[35m"+"DistributedBatchNorm3d:"+"\033[0m")
BatchNorm3d = DistributedBatchNorm3d(vt_world_size=8)

bn = BatchNorm3d(32)
out = bn(torch.rand([8,32,7,7,7]))
print(bn)
print(out.shape)
print("")

# the demo of ResNet using DistributedBatchNorm
print("\033[35m"+"ResNet50:"+"\033[0m")
BatchNorm2d = DistributedBatchNorm2d(vt_world_size=8)

model = resnet50(norm_layer=BatchNorm2d)
out = model(torch.rand([8,3,224,224]))
print(out.shape)