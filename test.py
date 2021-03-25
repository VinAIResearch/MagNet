import math

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from magnet.options.test import TestOptions
from magnet.dataset import get_dataset_with_name
from magnet.model import get_model_with_name
from magnet.model.refinement import RefinementMagNet
from magnet.utils.geometry import get_patch_coords, calculate_uncertainty
from magnet.utils.gaussian import GaussianBlur

def get_early_predictions(model, patches, sub_batch_size):

    early_preds = []
    n_patches = patches.shape[0]
    n_batches = math.ceil(n_patches/sub_batch_size)
    for batch_idx in range(n_batches):
        max_index = min((batch_idx + 1) * sub_batch_size, n_patches)
        batch = patches[batch_idx * sub_batch_size: max_index]
        with torch.no_grad():
            early_preds += [torch.softmax(model(batch), dim=1)]
    early_preds = torch.cat(early_preds, dim=0)
    return early_preds

if __name__ == "__main__":
    # Parse arguments
    opt = TestOptions().parse()

    sub_batch_size = opt.sub_batch_size

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Create dataset
    dataset = get_dataset_with_name(opt.dataset)(opt)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers)

    # Create model
    model = get_model_with_name(opt.model)(opt.num_classes).to(device)
    refinement_model = RefinementMagNet(opt.num_classes).to(device)
    
    # Load pretrained weights for backbone
    state_dict = torch.load(opt.pretrained)
    model.load_state_dict(state_dict)
    _ = model.eval()

    # Load pretrained weights for refinement module
    state_dict = torch.load(opt.pretrained_refinement)
    refinement_model.load_state_dict(state_dict, strict=False)
    _ = refinement_model.eval()

    # Patch coords
    patch_coords = []
    for scale in opt.scales:
        patch_coords += [get_patch_coords(scale, opt.crop_size)]

    # Allocate prediction map
    C, H, W = opt.num_classes, opt.scales[-1][1], opt.scales[-1][0]
    final_output = torch.zeros((C, H, W), dtype=torch.float).to(device)

    # Blur function
    gaussian_blur = GaussianBlur(kernel_size=(opt.smooth_kernel, opt.smooth_kernel), sigma=(1.0, 1.0))
    
    # Test dataloader
    for idx, data in tqdm(enumerate(dataloader)):
        image_patches = data["image_patches"][0]
        scale_idx = data["scale_idx"]
        label = data["label"]

        # Get early predictions at all scales
        image_patches = image_patches.to(device)
        early_preds = get_early_predictions(model, image_patches, sub_batch_size)

        import pdb; pdb.set_trace()

        # Refine from coarse-to-fine
        for idx, (coords, scale) in enumerate(zip(patch_coords, opt.scales)):
            
            # Downsample
            if idx == 0:
                final_output = F.interpolate(early_preds[0], (H, W), mode='bilinear').squeeze(0)
                continue
            
            # Filter early predictions
            scale_early_predictions = [x for i, x in enumerate(early_preds) if scale_idx[i] == idx]

            n_points = opt.n_points // len(coords)
            for coord, early_pred in zip(coords, scale_early_predictions):
                
                # Extract the previous prediction
                xmin, ymin, xmax, ymax = int(coord[0] * W), int(coord[1] * H), int(coord[2] * W), int(coord[3] * H)
                previous_pred = final_output[:, ymin: ymax, xmin: xmax].unsqueeze(0)
                previous_pred = F.interpolate(previous_pred, (opt.input_size[1], opt.input_size[1]), mode='bilinear')

                # Aggregate with current prediction
                aggregated_pred = refinement_model(previous_pred, early_pred)
                aggregated_pred = torch.softmax(aggregated_pred, dim=1)

                # Select points to refine
                # Calculate uncertainty of previous prediction
                uncertainty_score = calculate_uncertainty(previous_pred)

                # Calculate the certainty of aggregated prediction
                certainty_score = 1.0 - calculate_uncertainty(aggregated_pred)

                # Calculate scores
                error_score = fine_certainty_score * uncertainty_score

                # Smoothing scores
                error_score = gaussian_blur(error_score)

                # Replace the points in the final output with new prediction
                error_point_indices, error_point_coords = get_uncertain_point_coords_on_grid(error_score, n_points)

                C = opt.num_classes
                h, w = previous_pred.shape[2:]

                refined_point_pred = point_sample(aggregated_pred, error_point_coords)
                error_point_indices = error_point_indices.unsqueeze(1).expand(-1, opt.num_classes, -1)
                
                previous_pred = (
                            previous_pred.reshape(1, C, h * w)
                            .scatter_(2, error_point_indices, refined_point_pred)
                            .view(1, C, h, w)
                        )
                
                # Add back to the final output
                previous_pred = F.interpolate(previous_pred, (ymax - ymin, xmax - xmin), mode='bilinear')
                final_output[:, ymin: ymax, xmin: xmax] = previous_pred[0]
        
