import os

import numpy as np
import cv2

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from magnet.utils.transform import Resize, Patching, RandomPair, NormalizeInverse


class BaseDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()

        self.phase = opt.phase
        self.root = opt.root

        self.data = []
        with open(opt.datalist, "r") as f:
            for line in f.readlines():
                self.data += [self.parse_info(line)]

        self.scales = opt.scales  # Scale to this size
        self.crop_size = opt.crop_size  # Crop to this size
        self.input_size = opt.input_size  # Resize to this size

        self.ignore_label = -1
        self.label2color = {}
        self.label_reading_mode = cv2.IMREAD_COLOR

        # Transformation
        # For training
        if self.phase == "train":
            self.transform = RandomPair(self.scales[1:], self.crop_size, self.input_size)
        else:
            # For testing
            self.patch_transforms = []
            for scale in self.scales:
                self.patch_transforms += [Patching(scale, self.crop_size, Resize(scale), Resize(self.input_size))]

        self.image_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

        self.inverse_transform = NormalizeInverse([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def parse_info(self, line):
        info = {}
        tokens = line.strip().split("\t")
        info["image"] = os.path.join(self.root, tokens[0])
        info["label"] = os.path.join(self.root, tokens[1])
        return info

    def __len__(self):
        return len(self.data)

    def image2class(self, label):
        l, w = label.shape[0], label.shape[1]
        classmap = np.zeros(shape=(l, w), dtype=np.uint8)

        for classnum, color in self.label2color.items():
            indices = np.where(np.all(label == tuple(color[::-1]), axis=-1))
            classmap[indices[0].tolist(), indices[1].tolist()] = classnum
        return classmap

    def class2bgr(self, label):
        l, w = label.shape[0], label.shape[1]
        bgr = np.zeros(shape=(l, w, 3)).astype(np.uint8)
        for classnum, color in self.label2color.items():
            indices = np.where(label == classnum)
            bgr[indices[0].tolist(), indices[1].tolist(), :] = color[::-1]
        return bgr

    def augment(self, image, label):
        return image, label

    def __getitem__(self, index):

        image_path = self.data[index]["image"]
        label_path = self.data[index]["label"]

        # Read image
        image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        # Read label
        label = cv2.imread(label_path, self.label_reading_mode)

        if self.phase == "train":
            image, label = self.augment(image, label)

        # Cropping
        if self.phase == "train":
            # Random scale, crop, resize image and label
            coarse_image, _, fine_image, fine_label, info = self.transform(image, label)

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

        # Cropping with scales
        image_patches = []
        scale_idx = []

        label = self.image2class(label)
        for scale_id, patch_transform in enumerate(self.patch_transforms):
            image_crops = patch_transform(image)

            image_patches += list(map(lambda x: self.image_transform(x), image_crops))
            scale_idx += [scale_id for _ in range(len(image_crops))]
        image_patches = torch.stack(image_patches)
        scale_idx = torch.tensor(scale_idx)
        return {
            "image_patches": image_patches,
            "scale_idx": scale_idx,
            "label": label,
            "name": image_path.split("/")[-1],
        }
