from .base import BaseOptions

class TestOptions(BaseOptions):

    def __init__(self):
        super().__init__()
        parser = self.parser
        parser.add_argument('--sub_batch_size', required=True, type=int, help='batch size for patches')
        parser.add_argument('--n_patches', default=-1, type=int, help='number of patches to be refined at each stage. if n_patches=-1, all patches will be refined')
        parser.add_argument('--n_points', required=True, type=float, help='number of points to be refined at each stage')
        parser.add_argument('--smooth_kernel', default=16, type=int, help='kernel size to Gaussian blur error scores')
        parser.add_argument('--save_pred', action='store_true', help='save predictions, each image will contains: image, ground-truth, coarse pred, fine pred')
        parser.add_argument('--save_dir', default='test_results', help='saved directory')
        self.parser = parser
    
    def parse(self):
        args = super().parse()
        args.phase = "test"
        return args