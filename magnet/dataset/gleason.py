from .base import BaseDataset

class Gleason(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt)
    
        self.label2color = {
            0: [0, 0, 0],
            1: [50, 50, 50],
            2: [100, 100, 100],
            3: [150, 150, 150],
            4: [200, 200, 200]
        }