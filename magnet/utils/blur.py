import torch
import torch.nn as nn


def gaussian(window_size, sigma):
    """Generate gaussian kernel

    Args:
        window_size (int): filter size
        sigma (int): gaussian standard deviation
    Returns:
        torch.Tensor: (ksize, )
            gaussian kernel
    """

    def gauss_fcn(x):
        return -((x - window_size // 2) ** 2) / float(2 * sigma ** 2)

    gauss = torch.stack([torch.exp(torch.tensor(gauss_fcn(x))) for x in range(window_size)])
    return gauss / gauss.sum()


def get_gaussian_kernel(ksize, sigma):
    """Generate gaussian kernel

    Args:
        ksize (int): filter size, should be odd and positive
        sigma (float): standard deviation of gaussian

    Raises:
        TypeError: when kernel size is not odd positive integer

    Returns:
        torch.Tensor: (ksize, )
            gaussian kernel
    """

    if not isinstance(ksize, int) or ksize % 2 == 0 or ksize <= 0:
        raise TypeError("ksize must be an odd positive integer. Got {}".format(ksize))
    window_1d = gaussian(ksize, sigma)
    return window_1d


def get_gaussian_kernel2d(ksize, sigma):
    """Function that returns Gaussian filter matrix coefficients

    Args:
        ksize (Tuple[int, int]): filter sizes in the x and y direction.
         Sizes should be odd and positive.
        sigma (Tuple[float, float]): gaussian standard deviation in the x and y
         direction.

    Raises:
        TypeError: kernel size should be a tuple of two integers
        TypeError: sigma size should be a tuple of two integers

    Returns:
        torch.Tensor: (ksize_x, ksize_y)
            2D tensor with gaussian filter matrix coefficients.
    """
    if not isinstance(ksize, tuple) or len(ksize) != 2:
        raise TypeError("ksize must be a tuple of length two. Got {}".format(ksize))
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError("sigma must be a tuple of length two. Got {}".format(sigma))
    ksize_x, ksize_y = ksize
    sigma_x, sigma_y = sigma
    kernel_x = get_gaussian_kernel(ksize_x, sigma_x)
    kernel_y = get_gaussian_kernel(ksize_y, sigma_y)
    kernel_2d = torch.matmul(kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
    return kernel_2d


def compute_zero_padding(kernel_size):
    """Compute zero padding for specific kernel size

    Args:
        kernel_size (Tuple[int, int]): kernel size

    Returns:
        Tuple[int, int]: padding for each derection
    """
    computed = [(k - 1) // 2 for k in kernel_size]
    return computed[0], computed[1]


def get_binary_kernel2d(window_size):
    """Create a binary kernel to extract the patches

    Args:
        window_size (Tuple[int, int]): window size

    Returns:
        torch.Tensor: (H * W) x H x W
            The binary kernel
    """
    window_range = window_size[0] * window_size[1]
    kernel = torch.zeros(window_range, window_range)
    for i in range(window_range):
        kernel[i, i] += 1.0
    return kernel.view(window_range, 1, window_size[0], window_size[1])


class GaussianBlur(nn.Module):
    """Creates an operator that blurs a tensor using a Gaussian filter.

    The operator smooths the given tensor with a gaussian kernel by convolving
    it to each channel. It suports batched operation.

    Args:
        channel (int): number of channels of the input
        kernel_size (Tuple[int, int]): the size of the kernel.
        sigma (Tuple[float, float]): the standard deviation of the kernel.

    Attributes:
        kernel_size (Tuple[int, int]): the size of the kernel.
        sigma (Tuple[float, float]): the standard deviation of the kernel.
    """

    def __init__(self, channel, kernel_size, sigma):
        super(GaussianBlur, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self._padding = compute_zero_padding(kernel_size)

        kernel: torch.Tensor = get_gaussian_kernel2d(kernel_size, sigma).repeat(channel, 1, 1, 1)
        self._conv = nn.Conv2d(
            channel, channel, kernel_size=kernel_size, stride=1, padding=self._padding, groups=channel, bias=False
        )
        self._conv.weight.data = kernel

    def forward(self, x):
        """Apply gaussian blur to the input

        Args:
            x (torch.Tensor): (B, C, H, W)
                input tensor

        Raises:
            TypeError: input is torch tensor
            ValueError: shape of input tensor should be B x C x H x W

        Returns:
            torch.Tensor: (B, C, H, W)
                output tensor
        """
        if not torch.is_tensor(x):
            raise TypeError("Input x type is not a torch.Tensor. Got {}".format(type(x)))
        if not len(x.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}".format(x.shape))
        return self._conv(x)


class MedianBlur(nn.Module):
    """[summary]

    Args:
        kernel_size (Tuple([int, int])): the size of blur kernel
    Attributes:
        kernel_size (Tuple([int, int])): the size of blur kernel
    """

    def __init__(self, kernel_size):
        super(MedianBlur, self).__init__()
        self.kernel_size = kernel_size

        padding = compute_zero_padding(kernel_size)
        kernel = get_binary_kernel2d(kernel_size)
        self._conv = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self._conv.weight.data = kernel

    def forward(self, x):
        """Apply blur kernel to the input

        Args:
            x (torch.Tensor): B x C x H x W
                input tensor

        Returns:
            torch.Tensor: B x C x H x W
                output tensor
        """
        b, c, h, w = x.shape
        x = self._conv(x.reshape(b * c, 1, h, w))
        x = x.view(b, c, -1, h, w)
        return torch.median(x, dim=2)[0]
