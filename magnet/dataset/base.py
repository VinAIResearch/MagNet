import os

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from magnet.utils.transform import NormalizeInverse, Patching, RandomPair, Resize


class BaseDataset(data.Dataset):
    """Base dataset generator

    Args:
        opt (Namespace): arguments to parse

    Attributes:
        phase (str): train, val phase
        root (str): root directory of dataset
        data (List([dict])): list of data information
        scales (List([int, int])): List of scale (w, h)
        crop_size (Tuple([int, int])): crop size (w, h)
        input_size (Tuple([int, int])): input size (w, h)
        ignore_label (int): index of ignored label
        label2color (Dict): mapping between label and color
        label_reading_mode (enum): label reading mode
        cropping_transform (object): transformation for training
        cropping_transforms (List(object)): list of transformations for validation
        image_transform (object): image transformation
        inverse_transform (object): inverse transformation to reconstruct image
    """

    def __init__(self, opt):
        super().__init__()

        self.phase = opt.phase
        self.root = opt.root

        # Parse the file datalist
        self.data = []
        if os.path.isfile(opt.datalist):
            with open(opt.datalist, "r") as f:
                for line in f.readlines():
                    self.data += [self.parse_info(line)]

        self.scales = opt.scales  # Scale to this size
        self.crop_size = opt.crop_size  # Crop to this size
        self.input_size = opt.input_size  # Resize to this size

        # For label parsing
        self.ignore_label = -1
        self.label2color = {}
        self.label_reading_mode = cv2.IMREAD_COLOR

        # Cropping transformation
        if self.phase == "train":
            # For training
            self.cropping_transform = RandomPair(self.scales[1:], self.crop_size, self.input_size)
        else:
            # For testing
            self.cropping_transforms = []
            for scale in self.scales:
                self.cropping_transforms += [Patching(scale, self.crop_size, Resize(scale), Resize(self.input_size))]

        # For image transformation
        self.image_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

        self.inverse_transform = NormalizeInverse([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def parse_info(self, line):
        """Parse information of each line in the filelist

        Args:
            line (str): line in filelist

        Returns:
            dict([str, str]): information of each data item
        """
        info = {}
        tokens = line.strip().split("\t")
        info["image"] = os.path.join(self.root, tokens[0])
        info["label"] = os.path.join(self.root, tokens[1])
        return info

    def __len__(self):
        return len(self.data)

    def image2class(self, label):
        """Convert image to class matrix

        Args:
            label (np.array): h x w x C (= 1 or = 3)
                label image

        Returns:
            np.array: h x w
                class matrix
        """
        l, w = label.shape[0], label.shape[1]
        classmap = np.zeros(shape=(l, w), dtype=np.uint8)

        for classnum, color in self.label2color.items():
            indices = np.where(np.all(label == tuple(color[::-1]), axis=-1))
            classmap[indices[0].tolist(), indices[1].tolist()] = classnum
        return classmap

    def class2bgr(self, label):
        """Convert class matrix to BGR image

        Args:
            label (np.array): h x w
                class matrix

        Returns:
            np.array: h x w x 3
                BGR image
        """
        l, w = label.shape[0], label.shape[1]
        bgr = np.zeros(shape=(l, w, 3)).astype(np.uint8)
        for classnum, color in self.label2color.items():
            indices = np.where(label == classnum)
            bgr[indices[0].tolist(), indices[1].tolist(), :] = color[::-1]
        return bgr

    def augment(self, image, label):
        """Augment image and label

        Args:
            image (np.array): h x w x 3
                image
            label (np.array): h x w x 3 or h x w x 1
                label image

        Returns:
            np.array: h x w x 3
                image after augmentation
            np.array: h x w x 3 or h x w x 1
                label image after augmentation
        """
        return image, label

    def slice_image(self, image):
        """Slice image to patches

        Args:
            image (np.array): H x W x 3
                image to be sliced

        Returns:
            torch.Tensor: N x 3 x h x w
                patches
            torch.Tensor: N
                indices of patches
        """
        image_patches = []
        scale_idx = []

        for scale_id, cropping_transform in enumerate(self.cropping_transforms):

            # Crop data for each scale
            image_crops = cropping_transform(image)

            image_patches += list(map(lambda x: self.image_transform(x), image_crops))
            scale_idx += [scale_id for _ in range(len(image_crops))]

        image_patches = torch.stack(image_patches)
        scale_idx = torch.tensor(scale_idx)

        return image_patches, scale_idx

    def __getitem__(self, index):

        # Get data information
        image_path = self.data[index]["image"]
        label_path = self.data[index]["label"]

        # Read image
        image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        # Read label
        label = cv2.imread(label_path, self.label_reading_mode)

        # Augment data for training phase
        if self.phase == "train":
            image, label = self.augment(image, label)

        # For training
        if self.phase == "train":
            # Random scale, crop, resize image and label
            coarse_image, _, fine_image, fine_label, info = self.cropping_transform(image, label)

            coarse_image = self.image_transform(coarse_image)
            fine_image = self.image_transform(fine_image)

            fine_label = self.image2class(fine_label)
            fine_label = torch.from_numpy(fine_label).type(torch.LongTensor)

            return {
                "coarse_image": coarse_image,
                "fine_image": fine_image,
                "fine_label": fine_label,
                "coord": torch.tensor(info).unsqueeze(0),
            }

        # For testing

        # Convert label
        label = self.image2class(label)
        image_patches, scale_idx = self.slice_image(image)

        return {
            "image_patches": image_patches,
            "scale_idx": scale_idx,
            "label": label,
            "name": image_path.split("/")[-1],
        }
