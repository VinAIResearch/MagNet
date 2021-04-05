import cv2

from .base import BaseDataset

class Cityscapes(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt)

        ignore_label = 255
        self.class_mapping = {-1: ignore_label, 0: ignore_label, 
                              1: ignore_label, 2: ignore_label, 
                              3: ignore_label, 4: ignore_label, 
                              5: ignore_label, 6: ignore_label, 
                              7: 0, 8: 1, 9: ignore_label, 
                              10: ignore_label, 11: 2, 12: 3, 
                              13: 4, 14: ignore_label, 15: ignore_label, 
                              16: ignore_label, 17: 5, 18: ignore_label, 
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                              25: 12, 26: 13, 27: 14, 28: 15, 
                              29: ignore_label, 30: ignore_label, 
                              31: 16, 32: 17, 33: 18}

        self.label2color = { 7: (128, 64,128), 8: (244, 35,232), 
                                11: ( 70, 70, 70), 12: (102,102,156), 
                                13: (190,153,153), 17: (153,153,153), 
                                19: (250,170, 30), 20: (220,220,  0), 
                                21: (107,142, 35), 22: (152,251,152), 
                                23: ( 70,130,180), 24: (220, 20, 60), 
                                25: (255,  0,  0), 26: (  0,  0,142), 
                                27: (  0,  0, 70), 28: (  0, 60,100), 
                                31: (  0, 80,100), 32: (  0,  0,230), 33: (119, 11, 32)}
        self.ignore_label = ignore_label
        self.label_reading_mode = cv2.IMREAD_GRAYSCALE

    def image2class(self, label):
        l, w = label.shape[0], label.shape[1]
        classmap = np.zeros(shape=(l, w), dtype=np.uint8)
        
        for k, v in self.class_mapping.items():
                classmap[label == k] = v
        return classmap
