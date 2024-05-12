import itertools
from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """
    Reshape an image tensor for 2D pooling

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    """

    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    # TODO: Implement for Task 4.3.
    new_height = (height + kh - 1) // kh
    new_width = (width + kw - 1) // kw
    # Reshape the input tensor
    reshaped_input = input.reshape(batch, channel, new_height, kh, new_width, kw)

    # Transpose the dimensions to match the desired output shape
    reshaped_input = reshaped_input.permute(0, 1, 2, 4, 3, 5)

    # Reshape the tensor to the final output shape
    reshaped_input = reshaped_input.reshape(batch, channel, new_height, new_width, kh * kw)
    return reshaped_input, new_height, new_width
    raise NotImplementedError('Need to implement for Task 4.3')


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled average pooling 2D

    Args:
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns:
        Pooled tensor
    """
    batch, channel, height, width = input.shape
    # TODO: Implement for Task 4.3.
    new_input, new_height, new_width = tile(input, kernel)
    for i in range(len(batch)):
        for j in range(len(channel)):
            for nh in range(len(new_height)):
                for nw in range(len(new_width)):
                    # 在nh * height-(nh+1) * height
                    avg = 0.0
                    for x in range(len(height)):
                        for y in range(len(width)):
                            avg += input[i * batch + j * channel + nh * new_height + nw * new_width + x * height + y]
                    avg = avg / (height * width)


    raise NotImplementedError('Need to implement for Task 4.3')


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input : input tensor
        dim : dimension to apply argmax


    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        "Forward of max should be max reduction"
        # TODO: Implement for Task 4.4.
        input_shape = input.shape
        max_values = rand(input_shape[:dim] + input_shape[dim + 1:])
        max_indices = rand(input_shape[:dim] + input_shape[dim + 1:])
        for indices in itertools.product(*[range(size) for size in input_shape[:dim] + input_shape[dim + 1:]]):
            # indices 相比input_shape少了一个维度dim,因此它对于dim后面的索引是indices[dim:]
            input_idx = indices[:dim] + (slice(None),) + indices[dim:]
            # 这里就是一个向量了，那就简单了,如果还要计算indices的话就要遍历了
            res_value = input[input_idx][0]
            res_indice = 0
            for i in range(len(input[input_idx])):
                if input[input_idx][i] > res_value:
                    res_value = input[input_idx][i]
                    res_indice = i
            # max_value = max(t1[input_idx])
            max_values[indices[:dim] + indices[dim:]] = res_value
            max_indices[indices[:dim] + indices[dim:]] = res_indice
        ctx.save_for_backward(input, max_indices, dim)
        return max_values
        raise NotImplementedError('Need to implement for Task 4.4')

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        "Backward of max should be argmax (see above)"
        # TODO: Implement for Task 4.4.
        input, max_indices, dim = ctx.saved_values
        grad_input = input.zeros(input.shape)
        for indices in itertools.product(*[range(size) for size in input.shape[:dim] + input.shape[dim + 1:]]):
            grad_input[indices[:dim] + max_indices[indices[:dim] + indices[dim:]] + indices[dim:]] = 1
        return grad_input * grad_output
        raise NotImplementedError('Need to implement for Task 4.4')


def max(input: Tensor, dim: int) -> Tensor:
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Compute the softmax as a tensor.



    $z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}$

    Args:
        input : input tensor
        dim : dimension to apply softmax

    Returns:
        softmax tensor
    """
    # TODO: Implement for Task 4.4.
    exp_input = input.exp()
    sum_exp = exp_input.sum(dim)
    softmax_output = exp_input / sum_exp
    return softmax_output
    raise NotImplementedError('Need to implement for Task 4.4')


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Compute the log of the softmax as a tensor.

    $z_i = x_i - \log \sum_i e^{x_i}$

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
        input : input tensor
        dim : dimension to apply log-softmax

    Returns:
         log of softmax tensor
    """
    # TODO: Implement for Task 4.4.
    raise NotImplementedError('Need to implement for Task 4.4')


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled max pooling 2D

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor : pooled tensor
    """
    batch, channel, height, width = input.shape
    # TODO: Implement for Task 4.4.
    raise NotImplementedError('Need to implement for Task 4.4')


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """
    Dropout positions based on random noise.

    Args:
        input : input tensor
        rate : probability [0, 1) of dropping out each position
        ignore : skip dropout, i.e. do nothing at all

    Returns:
        tensor with random positions dropped out
    """
    # TODO: Implement for Task 4.4.
    if not ignore:
        bit_tensor = rand(input.shape, input.backend) > rate
    return input * bit_tensor

    raise NotImplementedError('Need to implement for Task 4.4')
