from typing import Tuple

Hin = 14.0
Win = 14.0
kernel_size = (5, 5)

def param_calculatpr(
        Hin: float, Win: float,
        kernel_size: Tuple, stride: Tuple=(1, 1),
        padding: Tuple=(0, 0), dilation=(1, 1)) -> Tuple[float, float]:

    Hout = ((Hin + (2 * padding[0]) - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1
    Wout = ((Win + (2 * padding[1]) - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1

    return (Hout, Wout)