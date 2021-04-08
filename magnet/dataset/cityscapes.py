import cv2
import numpy as np

from .base import BaseDataset


class Cityscapes(BaseDataset):
    """Cityscapes dataset generator"""

    def __init__(self, opt):
        super().__init__(opt)

        # There are some ignored classes in this dataset
        ignore_label = 255
        self.class_mapping = {
            -1: ignore_label,
            0: ignore_label,
            1: ignore_label,
            2: ignore_label,
            3: ignore_label,
            4: ignore_label,
            5: ignore_label,
            6: ignore_label,
            7: 0,
            8: 1,
            9: ignore_label,
            10: ignore_label,
            11: 2,
            12: 3,
            13: 4,
            14: ignore_label,
            15: ignore_label,
            16: ignore_label,
            17: 5,
            18: ignore_label,
            19: 6,
            20: 7,
            21: 8,
            22: 9,
            23: 10,
            24: 11,
            25: 12,
            26: 13,
            27: 14,
            28: 15,
            29: ignore_label,
            30: ignore_label,
            31: 16,
            32: 17,
            33: 18,
        }

        self.label2color = {
            0: (128, 64, 128),
            1: (244, 35, 232),
            2: (70, 70, 70),
            3: (102, 102, 156),
            4: (190, 153, 153),
            5: (153, 153, 153),
            6: (250, 170, 30),
            7: (220, 220, 0),
            8: (107, 142, 35),
            9: (152, 251, 152),
            10: (70, 130, 180),
            11: (220, 20, 60),
            12: (255, 0, 0),
            13: (0, 0, 142),
            14: (0, 0, 70),
            15: (0, 60, 100),
            16: (0, 80, 100),
            17: (0, 0, 230),
            18: (119, 11, 32),
        }
        self.ignore_label = ignore_label

        # Reading label as grayscale
        self.label_reading_mode = cv2.IMREAD_GRAYSCALE

    def image2class(self, label):
        """Overwrite the parent class to convert grayscale label image"""
        l, w = label.shape[0], label.shape[1]
        classmap = np.zeros(shape=(l, w), dtype=np.uint8)

        for k, v in self.class_mapping.items():
            classmap[label == k] = v
        return classmap
