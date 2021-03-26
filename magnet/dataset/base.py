import os

import numpy as np
import cv2

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from magnet.utils.transform import SegCompose, OneOf, Resize, RandomCrop, Patching

class BaseDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        
        self.phase = opt.phase
        self.root = opt.root
        
        self.data = []
        with open(opt.datalist, "r") as f:
            for line in f.readlines():
                self.data += [self.parse_info(line)]
        
        self.scales = opt.scales                # Scale to this size
        self.crop_size = opt.crop_size          # Crop to this size
        self.input_size = opt.input_size        # Resize to this size

        self.label2color = {

        }

        # Transformation
        # For training
        if self.phase == "train":
            transform_list = []
            for scale in self.scales:
                transform_list += [SegCompose([Resize(scale), RandomCrop(self.crop_size), Resize(self.input_size)])]
            self.transform = OneOf(transform_list)
        else:
            # For testing
            self.patch_transforms = []
            for scale in self.scales:
                self.patch_transforms += [Patching(scale, self.crop_size, Resize(scale), Resize(self.input_size))]
        
        self.image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )

    def parse_info(self, line):
        info = {}
        tokens = line.strip().split("\t")
        info["image"] = os.path.join(self.root, tokens[0])
        info["label"] = os.path.join(self.root, tokens[1])
        return info

    def __len__(self):
        return len(self.data)

    def bgr2class(self, label):
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
        label = cv2.imread(label_path, cv2.IMREAD_COLOR)

        if self.phase == "train":
            image, label = self.augment(image, label)
        
        # Cropping
        if self.phase == "train":
            # Random scale, crop, resize image and label
            image, label = self.transform(image, label)
            
            image = self.image_transform(image)

            label = self.bgr2class(label)
            label = torch.from_numpy(label).type(torch.LongTensor)

            return {"image": image, "label": label}

        # Cropping with scales
        image_patches = []
        scale_idx = []

        label = self.bgr2class(label)
        for scale_id, patch_transform in enumerate(self.patch_transforms):
            image_crops = patch_transform(image)

            image_patches += list(map(lambda x: self.image_transform(x), image_crops))
            scale_idx += [scale_id for _ in range(len(image_crops))]
        image_patches = torch.stack(image_patches)
        scale_idx = torch.tensor(scale_idx)
        return {"image_patches": image_patches, "scale_idx": scale_idx, "label": label}

