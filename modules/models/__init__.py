from .res_U_net import *


def ResUNext(layers=[3, 4, 6, 3, 6, 4, 3], num_classes=1000, groups=32, width_per_group=8, export_FCL=False):
    return ResUNet(layers, num_classes=1000, groups=32, width_per_group=8, export_FCL=export_FCL)