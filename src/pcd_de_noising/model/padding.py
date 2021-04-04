import math


def num2tuple(num):
    return num if isinstance(num, tuple) else (num, num)


def conv2d_get_padding(h_w_in, h_w_out, kernel_size=1, stride=1, dilation=1):
    h_w_in, h_w_out, kernel_size, stride, dilation = (
        num2tuple(h_w_in),
        num2tuple(h_w_out),
        num2tuple(kernel_size),
        num2tuple(stride),
        num2tuple(dilation),
    )

    p_h = (
        (h_w_out[0] - 1) * stride[0]
        - h_w_in[0]
        + dilation[0] * (kernel_size[0] - 1)
        + 1
    )
    p_w = (
        (h_w_out[1] - 1) * stride[1]
        - h_w_in[1]
        + dilation[1] * (kernel_size[1] - 1)
        + 1
    )

    return (
        math.floor(p_h / 2),
        math.ceil(p_h / 2),
        math.floor(p_w / 2),
        math.ceil(p_w / 2),
    )
