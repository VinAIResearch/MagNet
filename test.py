from tqdm import tqdm
from torch.utils.data import DataLoader

from magnet.options.test import TestOptions
from magnet.dataset import get_dataset_with_name

if __name__ == "__main__":
    # Parse arguments
    opt = TestOptions().parse()

    # Create dataset
    dataset = get_dataset_with_name(opt.dataset)(opt)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers)
    
    # Test dataloader
    for idx, data in tqdm(enumerate(dataloader)):
        image_patches = data["image_patches"][0]
        scale_idx = data["scale_idx"]
        label = data["label"]
        # import pdb; pdb.set_trace()

        # Forward to get predictions at all scales

        # Refine from coarse-to-fine
