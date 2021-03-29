import math
import time

from tqdm import tqdm

import numpy as np
import cv2

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.ops import roi_align

from magnet.options.test import TestOptions
from magnet.dataset import get_dataset_with_name
from magnet.model import get_model_with_name
from magnet.model.refinement import RefinementMagNet
from magnet.utils.geometry import get_patch_coords, calculate_uncertainty, get_uncertain_point_coords_on_grid, point_sample, ensemble
from magnet.utils.blur import GaussianBlur, MedianBlur
from magnet.utils.metrics import getIoU, confusion_matrix

@torch.no_grad()
def get_batch_predictions(model, sub_batch_size, patches, another=None):

    preds = []
    n_patches = patches.shape[0]
    n_batches = math.ceil(n_patches/sub_batch_size)
    for batch_idx in range(n_batches):
        max_index = min((batch_idx + 1) * sub_batch_size, n_patches)
        batch = patches[batch_idx * sub_batch_size: max_index]
        with torch.no_grad():
            if another is None:
                preds += [torch.softmax(model(batch), dim=1)]
            else:
                preds += [torch.softmax(model(batch, another[batch_idx * sub_batch_size: max_index]), dim=1)]
        free_mem(True)
    preds = torch.cat(preds, dim=0)
    return preds

def get_mean_iou(conf_mat, dataset):
    IoU = getIoU(conf_mat)
    if dataset == "deepglobe":
        return np.nanmean(IoU[1:])

def free_mem(clear_cache=False):
    if clear_cache:
        torch.cuda.empty_cache()

@torch.no_grad()
def main():

    # Parse arguments
    opt = TestOptions().parse()

    sub_batch_size = opt.sub_batch_size

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Create dataset
    dataset = get_dataset_with_name(opt.dataset)(opt)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers)

    # Create model
    model = get_model_with_name(opt.model)(opt.num_classes).to(device)
    model.multi_test = opt.multi_test
    refinement_model = RefinementMagNet(opt.num_classes, use_bn=True).to(device)
    
    # Load pretrained weights for backbone
    state_dict = torch.load(opt.pretrained)
    model.load_state_dict(state_dict)
    _ = model.eval()

    # Load pretrained weights for refinement module
    # state_dict = torch.load(opt.pretrained_refinement)
    # refinement_model.load_state_dict(state_dict, strict=False)
    _ = refinement_model.eval()

    # Patch coords
    patch_coords = []
    for scale in opt.scales:
        patch_coords += [torch.tensor(get_patch_coords(scale, opt.crop_size)).to(device)]

    # Allocate prediction map
    C, H, W = opt.num_classes, opt.scales[-1][1], opt.scales[-1][0]
    final_output = None

    # Blur function
    # gaussian_blur = GaussianBlur(channel=1, kernel_size=(opt.smooth_kernel, opt.smooth_kernel), sigma=(1.0, 1.0)).to(device)
    median_blur = MedianBlur(channel=1, kernel_size=(opt.smooth_kernel, opt.smooth_kernel)).to(device)
    median_blur.eval()

    conf_mat = np.zeros((opt.num_classes, opt.num_classes), dtype=np.float)
    refined_conf_mat = np.zeros((opt.num_classes, opt.num_classes), dtype=np.float)
    
    # Test dataloader
    pbar = tqdm(total=len(dataset), ascii=True)
    for idx, data in enumerate(dataloader):
        
        pbar.update(1)
        execution_time = {}
        description = ""

        image_patches = data["image_patches"][0]
        scale_idx = data["scale_idx"][0]
        label = data["label"].numpy()

        total_time = time.time()
        eval_time = 0
        coarse_pred = None

        # import pdb; pdb.set_trace()

        # Refine from coarse-to-fine
        for idx, (ratios, scale) in enumerate(zip(patch_coords, opt.scales)):
            
            # If the first scale, get the prediction only
            if idx == 0:
                
                # Get prediction 
                start_time = time.time()
                final_output = get_batch_predictions(model, 1, image_patches[0:1].to(device))
                execution_time["backbone_ff"] = execution_time.get("backbone_ff", 0) + (time.time() - start_time)
                
                coarse_pred = final_output.clone()
                continue

            # coords = [(x1 * final_output.shape[3], y1 * final_output.shape[2], x2 * final_output.shape[3], y2 * final_output.shape[2]) for x1, y1, x2, y2 in ratios]
            # coords = torch.tensor(coords).to(device)
            coords = ratios.clone()
            coords[:, 0] = coords[:, 0] * final_output.shape[3]
            coords[:, 1] = coords[:, 1] * final_output.shape[2]
            coords[:, 2] = coords[:, 2] * final_output.shape[3]
            coords[:, 3] = coords[:, 3] * final_output.shape[2]

            # Calculate uncertainty
            start_time = time.time()
            uncertainty = calculate_uncertainty(final_output)
            patch_uncertainty = roi_align(uncertainty, [coords], output_size=(opt.input_size[1], opt.input_size[0]))
            patch_uncertainty = patch_uncertainty.mean((1,2,3))
            execution_time["cal_uncertainty"] = execution_time.get("cal_uncertainty", 0) + (time.time() - start_time)

            # Choose patches with highest mean uncertainty
            start_time = time.time()
            _, selected_patch_ids = torch.sort(patch_uncertainty)

            del patch_uncertainty
            free_mem()

            if opt.n_patches != -1:
                selected_patch_ids = selected_patch_ids[:opt.n_patches]
            execution_time["choose_patch"] = execution_time.get("choose_patch", 0) + (time.time() - start_time)

            # Filter image_patches of this scale
            scale_image_patches = image_patches[scale_idx == idx]

            # Filter image_patches with selected_patch_ids
            scale_image_patches = scale_image_patches[selected_patch_ids]

            # Get early predictions
            start_time = time.time()
            scale_early_preds = get_batch_predictions(model, sub_batch_size, scale_image_patches.to(device))
            execution_time["backbone_ff"] = execution_time.get("backbone_ff", 0) + (time.time() - start_time)

            # Get coarse preds (with coords and final_output)
            start_time = time.time()
            coarse_preds = roi_align(final_output, [coords[selected_patch_ids]], output_size=(opt.input_size[1], opt.input_size[0]))
            execution_time["extract_coarse"] = execution_time.get("extract_coarse", 0) + (time.time() - start_time)

            # Refinement
            start_time = time.time()
            fine_pred = get_batch_predictions(refinement_model, sub_batch_size, coarse_preds, scale_early_preds)
            execution_time["refinement_ff"] = execution_time.get("refinement_ff", 0) + (time.time() - start_time)

            del coarse_preds, scale_early_preds
            free_mem()

            # Make grids
            
            start_time = time.time()

            selected_ratios = ratios[selected_patch_ids]
            execution_time["make_grid"] = execution_time.get("make_grid", 0) + (time.time() - start_time)
            
            fine_pred = ensemble(fine_pred, selected_ratios, scale)
            free_mem()

            # Calculate certainty of fine_pred
            start_time = time.time()
            certainty_score = 1.0 - calculate_uncertainty(fine_pred)
            uncertainty_score = F.interpolate(uncertainty, scale[::-1], mode='bilinear', align_corners=False)
            error_score = certainty_score * uncertainty_score
            del certainty_score, uncertainty_score
            free_mem()
            execution_time["cal_certainty"] = execution_time.get("cal_certainty", 0) + (time.time() - start_time)
            

            # Smoothing error score
            # with torch.no_grad():
            #     error_score = gaussian_blur(error_score)
            start_time = time.time()
            # with torch.no_grad():
            #     error_score = median_blur(error_score)
            execution_time["blur"] = execution_time.get("blur", 0) + (time.time() - start_time)
            
            # Get point coordinates
           
            if opt.n_points > 1.0:
                n_points = int(opt.n_points)
            else:
                n_points = int(scale[0] * scale[1] * opt.n_points)
            
            start_time = time.time()
            error_point_indices, error_point_coords = get_uncertain_point_coords_on_grid(error_score, n_points)
            del error_score
            free_mem()
            execution_time["get_point"] = execution_time.get("get_point", 0) + (time.time() - start_time)
            
            error_point_indices = error_point_indices.unsqueeze(1).expand(-1, opt.num_classes, -1)  

            # Get refinement prediction 
            start_time = time.time()
            fine_pred = point_sample(fine_pred, error_point_coords, align_corners=False)
            free_mem()
            execution_time["sample_point"] = execution_time.get("sample_point", 0) + (time.time() - start_time)    
            
            final_output = F.interpolate(final_output, scale[::-1], mode='bilinear', align_corners=False)
            free_mem()

            final_output = (
                            final_output.reshape(1, opt.num_classes, scale[0] * scale[1])
                            .scatter_(2, error_point_indices, fine_pred)
                            .view(1, opt.num_classes, scale[1], scale[0])
                        )
        
        execution_time["time"] = time.time() - total_time

        # Compute IoU for coarse prediction
        start_time = time.time()
        coarse_pred = F.interpolate(coarse_pred, (H, W), mode='bilinear', align_corners=False).argmax(1).cpu().numpy()
        mat = confusion_matrix(label, coarse_pred, opt.num_classes)
        conf_mat += mat
        # description += "Coarse IoU: %.2f, " % (get_mean_iou(mat, opt.dataset)*100)

        # Compute IoU for fine prediction
        final_output = final_output.argmax(1).cpu().numpy()
        mat = confusion_matrix(label, final_output, opt.num_classes)
        refined_conf_mat += mat

        # description += "Refinement IoU: %.2f" % (get_mean_iou(mat, opt.dataset)*100)
        
        description += "".join([", %s: %.2f" % (k, v) for k,v in execution_time.items() if v > 0.01])
        pbar.set_description(description)
    
    pbar.write("-------SUMMARY-------")
    pbar.write("Coarse IoU: %.2f" % (get_mean_iou(conf_mat, opt.dataset)*100))
    pbar.write("Refinement IoU: %.2f" % (get_mean_iou(refined_conf_mat, opt.dataset)*100))


if __name__ == "__main__":
    main()