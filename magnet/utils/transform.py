import math
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image


class SegCompose(object):
    """Compose of augmentations applied to segmentation data

    Args:
        augmenters (List([object])): list of augmenters
    """

    def __init__(self, augmenters):
        super().__init__()
        self.augmenters = augmenters

    def __call__(self, image, label):
        """Apply augmentation

        Args:
            image (np.array): h x w 3
                image
            label (np.array): h x w x 3 or h x w x 1
                label

        Returns:
            image (np.array): h x w x 3
                augmented image
            label (np.array): h x w x 3 or h x w x 1
                augmented label
        """
        for augmenter in self.augmenters:
            image, label = augmenter(image, label)
        return image, label


class OneOf(object):
    """Choose one of augmentations and apply to image and label

    Args:
        augmenters (List([object])): list of augmenters
    """

    def __init__(self, augmenters):
        super().__init__()
        self.augmenters = augmenters

    def __call__(self, image, label):
        """Apply augmentation

        Args:
            image (np.array): h x w 3
                image
            label (np.array): h x w x 3 or h x w x 1
                label

        Returns:
            image (np.array): h x w x 3
                augmented image
            label (np.array): h x w x 3 or h x w x 1
                augmented label
        """
        augmenter = random.choice(self.augmenters)
        return augmenter(image, label)


class Resize(object):
    """Resize image and label

    Args:
        size (Tuple([int, int])): size to resize (width, height)
    """

    def __init__(self, size):
        super().__init__()
        self.size = size

    def __call__(self, image, label):
        """Apply augmentation

        Args:
            image (np.array): h x w 3
                image
            label (np.array): h x w x 3 or h x w x 1
                label

        Returns:
            image (np.array): h x w x 3
                augmented image
            label (np.array): h x w x 3 or h x w x 1
                augmented label
        """
        width, height = self.size
        h, w = image.shape[0], image.shape[1]
        if width == -1:
            width = int(height / h * w)
        if height == -1:
            height = int(width / w * h)

        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        label = label if label is None else cv2.resize(label, (width, height), interpolation=cv2.INTER_NEAREST)
        return image, label


class Patching(object):
    """Patch processing image

    Args:
        scale (Tuple([int, int])): scale size (width, height)
        crop (Tuple([int, int])): crop size (width, height)
        preaugmenter (object, optional): augmenter applied before cropping. Defaults to None.
        post_augmenter (object, optional): augmenter applied after cropping. Defaults to None.
    """

    def __init__(self, scale, crop, pre_augmenter=None, post_augmenter=None):
        super().__init__()
        crops = []
        n_x = math.ceil(scale[0] / crop[0])
        step_x = int(crop[0] - (n_x * crop[0] - scale[0]) / max(n_x - 1, 1.0))
        n_y = math.ceil(scale[1] / crop[1])
        step_y = int(crop[1] - (n_y * crop[1] - scale[1]) / max(n_y - 1, 1.0))

        for x in range(n_x):
            for y in range(n_y):
                crops += [(x * step_x, y * step_y, x * step_x + crop[0], y * step_y + crop[1])]
        self.crops = crops
        self.pre_augmenter = pre_augmenter
        self.post_augmenter = post_augmenter

    def __call__(self, image):
        """Apply augmentation

        Args:
            image (np.array): H x W 3
                image

        Returns:
            List([np.array]): [h x w x 3]
                list of crops
        """
        if self.pre_augmenter is not None:
            image, _ = self.pre_augmenter(image, None)
        image_crops = []
        for crop in self.crops:
            x, y, xmax, ymax = crop
            im = image[y:ymax, x:xmax]
            if self.post_augmenter is not None:
                im, _ = self.post_augmenter(im, None)

            image_crops += [im]

        return image_crops


class RandomCrop(object):
    """Random crop image and label

    Args:
        size (Tuple([int, int])): size of input
    """

    def __init__(self, size):
        super().__init__()
        self.size = size

    def __call__(self, image, label):
        """Apply augmentation

        Args:
            image (np.array): h x w 3
                image
            label (np.array): h x w x 3 or h x w x 1
                label

        Returns:
            image (np.array): h x w x 3
                augmented image
            label (np.array): h x w x 3 or h x w x 1
                augmented label
        """
        max_x = image.shape[1] - self.size[0]
        max_y = image.shape[0] - self.size[1]

        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)

        image = image[y : y + self.size[1], x : x + self.size[0]]
        label = label if label is None else label[y : y + self.size[1], x : x + self.size[0]]

        return image, label


class RandomRotate(object):
    """Random rotate image and label

    Args:
        max_angle (int): maximum angle when applying rotation
    """

    def __init__(self, max_angle):
        super().__init__()
        self.max_angle = max_angle

    def __call__(self, image, label):
        """Apply augmentation

        Args:
            image (np.array): h x w 3
                image
            label (np.array): h x w x 3 or h x w x 1
                label

        Returns:
            image (np.array): h x w x 3
                augmented image
            label (np.array): h x w x 3 or h x w x 1
                augmented label
        """
        angle = random.randint(0, self.max_angle * 2) - self.max_angle

        image = Image.fromarray(image)
        image = image.rotate(angle, resample=Image.BILINEAR)
        image = np.array(image)

        if label is not None:
            label = Image.fromarray(label)
            label = label.rotate(angle, resample=Image.NEAREST)
            label = np.array(label)

        return image, label


class RandomFlip(object):
    """Random flip image and label"""

    def __init__(self):
        super().__init__()

    def __call__(self, image, label):
        """Apply augmentation

        Args:
            image (np.array): h x w 3
                image
            label (np.array): h x w x 3 or h x w x 1
                label

        Returns:
            image (np.array): h x w x 3
                augmented image
            label (np.array): h x w x 3 or h x w x 1
                augmented label
        """
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)
        return image, label


class RandomPair(object):
    """Random generate pairs for training

    Args:
        scales (List([Tuple([int, int])])): list of scales
        crop_size (Tuple([int, int])): crop size (width, height)
        input_size (Tuple([int, int])): input size (width, height)
    """

    def __init__(self, scales, crop_size, input_size):
        super().__init__()
        self.scales = [Resize(scale) for scale in scales]
        self.crop_size = crop_size
        self.resize = Resize(input_size)

    def __call__(self, image, label):
        """Apply augmentation

        Args:
            image (np.array): h x w 3
                image
            label (np.array): h x w x 3 or h x w x 1
                label

        Returns:
            np.array: h x w x 3
                coarse image
            np.array: h x w x 3 or h x w x 1
                coarse label
            np.array: h x w x 3
                fine image
            np.array: h x w x 3 or h x w x 1
                fine label
            Tuple([float, float, float, float]): cropping information
        """
        # Resize to get coarse scale
        coarse_image, coarse_label = self.resize(image, label)

        # Pick random scale
        fine_image, fine_label = random.choice(self.scales)(image, label)

        # Random crop
        max_x = fine_image.shape[1] - self.crop_size[0]
        max_y = fine_image.shape[0] - self.crop_size[1]

        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)

        xmin = x * 1.0 / fine_image.shape[1] * coarse_image.shape[1]
        ymin = y * 1.0 / fine_image.shape[0] * coarse_image.shape[0]
        xmax = xmin + (self.crop_size[0] * 1.0 / fine_image.shape[1] * coarse_image.shape[1])
        ymax = ymin + (self.crop_size[1] * 1.0 / fine_image.shape[0] * coarse_image.shape[0])

        fine_image = fine_image[y : y + self.crop_size[1], x : x + self.crop_size[0]]
        fine_label = fine_label[y : y + self.crop_size[1], x : x + self.crop_size[0]]

        info = (xmin, ymin, xmax, ymax)

        # Resize fine image
        fine_image, fine_label = self.resize(fine_image, fine_label)

        return coarse_image, coarse_label, fine_image, fine_label, info


class NormalizeInverse(TF.Normalize):
    """Undoes the normalization and returns the reconstructed images in the input domain.

    Args:
        mean (List([float, float, float])): mean value
        std (List([float, float, float])): standard deviation
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
