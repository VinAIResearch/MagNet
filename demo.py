import math
import os
import time

import cv2
import torch
import torch.nn.functional as F
from magnet.dataset import get_dataset_with_name
from magnet.model import get_model_with_name
from magnet.model.refinement import RefinementMagNet
from magnet.options.test import TestOptions
from magnet.utils.blur import MedianBlur
from magnet.utils.geometry import (
    calculate_certainty,
    ensemble,
    get_patch_coords,
    get_uncertain_point_coords_on_grid,
    point_sample,
)
from torchvision.ops import roi_align


@torch.no_grad()
def get_batch_predictions(model, sub_batch_size, patches, another=None):
    """Inference model with batch

    Args:
        model (nn.Module): model to inference
        sub_batch_size (int): batch size
        patches (torch.Tensor): B x C x H x W
            patches to infer
        another (torch.Tensor, optional): B x C x H x W, another inputs. Defaults to None.

    Returns:
        torch.Tensor: B x C x H x W
            predictions (after softmax layer)
    """
    preds = []
    n_patches = patches.shape[0]
    n_batches = math.ceil(n_patches / sub_batch_size)

    # Process each batch
    for batch_idx in range(n_batches):
        max_index = min((batch_idx + 1) * sub_batch_size, n_patches)
        batch = patches[batch_idx * sub_batch_size : max_index]
        with torch.no_grad():
            if another is None:
                preds += [torch.softmax(model(batch), dim=1)]
            else:
                preds += [torch.softmax(model(batch, another[batch_idx * sub_batch_size : max_index]), dim=1)]
    preds = torch.cat(preds, dim=0)
    return preds


@torch.no_grad()
def main():

    # Parse arguments
    opt = TestOptions().parse()

    sub_batch_size = opt.sub_batch_size

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Create dataset
    dataset = get_dataset_with_name(opt.dataset)(opt)

    # Create model
    model = get_model_with_name(opt.model)(opt.num_classes).to(device)

    # Load pretrained weights for backbone
    state_dict = torch.load(opt.pretrained)
    model.load_state_dict(state_dict)
    _ = model.eval()

    # Create refinement models
    pretrained_weight = [opt.pretrained_refinement]
    if isinstance(opt.pretrained_refinement, list):
        assert len(opt.scales) - 1 == len(
            opt.pretrained_refinement
        ), "The number of refinement weights must match (no.scales - 1)"
        pretrained_weight = opt.pretrained_refinement

    refinement_models = []

    # Load pretrained weight of refinement modules
    for weight_path in pretrained_weight:
        refinement_model = RefinementMagNet(opt.num_classes, use_bn=True).to(device)

        # Load pretrained weights for refinement module
        state_dict = torch.load(weight_path)
        refinement_model.load_state_dict(state_dict, strict=False)
        _ = refinement_model.eval()
        refinement_models += [refinement_model]

    # Patch coords
    patch_coords = []
    for scale in opt.scales:
        patch_coords += [torch.tensor(get_patch_coords(scale, opt.crop_size)).to(device)]

    # Allocate prediction map
    _, H, W = opt.num_classes, opt.scales[-1][1], opt.scales[-1][0]
    final_output = None

    # Blur operator
    median_blur = MedianBlur(kernel_size=(opt.smooth_kernel, opt.smooth_kernel)).to(device)
    median_blur.eval()

    # Loading image
    image = cv2.cvtColor(cv2.imread(opt.image, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    ori_H, ori_W = image.shape[0], image.shape[1]
    image_patches, scale_idx = dataset.slice_image(image)

    total_time = time.time()
    intermediate_preds = []

    # Refine from coarse-to-fine
    for idx, (ratios, scale) in enumerate(zip(patch_coords, opt.scales)):

        # If the first scale, get the prediction only
        if idx == 0:

            # Get prediction
            final_output = get_batch_predictions(model, 1, image_patches[0:1].to(device))

            intermediate_preds = [final_output.clone()]
            continue

        if opt.n_patches == 0:
            continue

        # Upscale current output
        final_output = F.interpolate(final_output, scale[::-1], mode="bilinear", align_corners=False)

        coords = ratios.clone()
        coords[:, 0] = coords[:, 0] * final_output.shape[3]
        coords[:, 1] = coords[:, 1] * final_output.shape[2]
        coords[:, 2] = coords[:, 2] * final_output.shape[3]
        coords[:, 3] = coords[:, 3] * final_output.shape[2]

        # Calculate uncertainty
        uncertainty = 1.0 - calculate_certainty(final_output)
        patch_uncertainty = roi_align(uncertainty, [coords], output_size=(opt.input_size[1], opt.input_size[0]))
        patch_uncertainty = patch_uncertainty.mean((1, 2, 3))

        # Choose patches with highest mean uncertainty
        _, selected_patch_ids = torch.sort(patch_uncertainty)

        del patch_uncertainty

        if opt.n_patches != -1:
            selected_patch_ids = selected_patch_ids[: opt.n_patches]

        # Filter image_patches of this scale
        scale_image_patches = image_patches[scale_idx == idx]

        # Filter image_patches with selected_patch_ids
        scale_image_patches = scale_image_patches[selected_patch_ids]

        # Get early predictions
        scale_early_preds = get_batch_predictions(model, sub_batch_size, scale_image_patches.to(device))

        # Get coarse preds (with coords and final_output)
        coarse_preds = roi_align(
            final_output, [coords[selected_patch_ids]], output_size=(opt.input_size[1], opt.input_size[0])
        )

        # Refinement
        fine_pred = get_batch_predictions(
            refinement_models[min(len(refinement_models), idx) - 1],
            sub_batch_size,
            scale_early_preds,
            coarse_preds,
        )

        del coarse_preds, scale_early_preds

        # Make grids
        selected_ratios = ratios[selected_patch_ids]
        fine_pred, mask = ensemble(fine_pred, selected_ratios, scale)

        # Calculate certainty of fine_pred
        certainty_score = calculate_certainty(fine_pred)

        if opt.n_patches > 0:
            certainty_score[:, :, mask] = 0.0

        uncertainty_score = F.interpolate(uncertainty, scale[::-1], mode="bilinear", align_corners=False)

        # Calculate error score
        error_score = certainty_score * uncertainty_score
        del certainty_score, uncertainty_score

        # Smoothing error score
        _, _, h_e, w_e = error_score.shape
        error_score = F.interpolate(error_score, size=(opt.input_size[1], opt.input_size[0]))
        with torch.no_grad():
            error_score = median_blur(error_score)
        error_score = F.interpolate(error_score, size=(h_e, w_e))

        if opt.n_points > 1.0:
            n_points = min(int(opt.n_points), scale[0] * scale[1] * len(selected_patch_ids) / len(coords))
        else:
            n_points = int(scale[0] * scale[1] * opt.n_points * len(selected_patch_ids) / len(coords))

        # Get point coordinates
        error_point_indices, error_point_coords = get_uncertain_point_coords_on_grid(error_score, n_points)
        del error_score

        error_point_indices = error_point_indices.unsqueeze(1).expand(-1, opt.num_classes, -1)

        # Get refinement prediction
        fine_pred = point_sample(fine_pred, error_point_coords, align_corners=False)

        if opt.n_patches > 0:
            # Apply mask
            sample_mask = (
                point_sample(mask.type(torch.float).unsqueeze(0).unsqueeze(0), error_point_coords, align_corners=False)
                .type(torch.bool)
                .squeeze()
            )
            error_point_indices = error_point_indices[:, :, sample_mask]
            fine_pred = fine_pred[:, :, sample_mask]

        # Replace points with new prediction
        final_output = (
            final_output.reshape(1, opt.num_classes, scale[0] * scale[1])
            .scatter_(2, error_point_indices, fine_pred)
            .view(1, opt.num_classes, scale[1], scale[0])
        )
        intermediate_preds.append(final_output.clone())

    processing_time = time.time() - total_time
    print("Done processing image in %.2f seconds" % processing_time)

    if opt.save_pred:
        image_name = opt.image.split("/")[-1].split(".")[0]

        print("Saving output to {}/{}".format(opt.save_dir, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        os.makedirs(os.path.join(opt.save_dir, image_name), exist_ok=True)

        for scale, pred in zip(opt.scales, intermediate_preds):
            pred = F.interpolate(pred, (H, W), mode="bilinear", align_corners=False).argmax(1).cpu().numpy()

            # Convert predictions to images
            pred = dataset.class2bgr(pred[0])
            pred = cv2.resize(pred, (ori_W, ori_H))
            pred = (0.5 * image + 0.5 * pred).astype("uint8")

            # Save predictions
            pred_path = os.path.join(opt.save_dir, image_name, "{}x{}.jpg".format(scale[0], scale[1]))
            cv2.imwrite(pred_path, pred)


if __name__ == "__main__":
    main()
