from .base import BaseDataset


class Deepglobe(BaseDataset):
    """Deepglobe dataset generator"""

    def __init__(self, opt):
        super().__init__(opt)

        self.label2color = {
            0: [0, 0, 0],
            1: [0, 255, 255],
            2: [255, 255, 0],
            3: [255, 0, 255],
            4: [0, 255, 0],
            5: [0, 0, 255],
            6: [255, 255, 255],
        }
