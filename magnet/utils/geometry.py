import math

import torch
import torch.nn.functional as F


def get_patch_coords(size, input_size):
    """Get patch coordinations

    Args:
        size (Tuple([int, int])): size to be sliced (width, height)
        input_size (Tuple([int, int])): patch size (width, height)

    Returns:
        List(Tuple([float, float, float, float])): list of coordinates (ratios) (xmin, ymin, xmax, ymax)
    """
    coords = []

    n_x = math.ceil(size[0] / input_size[0])
    step_x = int(input_size[0] - (n_x * input_size[0] - size[0]) / max(n_x - 1, 1.0))
    n_y = math.ceil(size[1] / input_size[1])
    step_y = int(input_size[1] - (n_y * input_size[1] - size[1]) / max(n_y - 1, 1.0))

    for x in range(n_x):
        for y in range(n_y):
            coords += [
                (
                    x * step_x / size[0],
                    y * step_y / size[1],
                    (x * step_x + input_size[0]) / size[0],
                    (y * step_y + input_size[1]) / size[1],
                )
            ]

    return coords


def ensemble(patches, coords, output_size):
    """Ensemble patches with corresponding coordinates

    Args:
        patches (torch.Tensor): B x C x H x W
            patches to ensemble
        coords (torch.Tensor): B x 4
            coordinates of patches (ratios)
        output_size (Tuple([int, int])): output size (width, height)

    Returns:
        torch.Tensor: 1 x C x output_size[1] x output_size[0]
            output ensembled by patches
        torch.Tensor: output_size[1] x output_size[0]
            mask of output, some places would have no patch
    """
    _, C, _, _ = patches.shape

    output = torch.zeros((1, C, output_size[1], output_size[0]), device=patches.device, dtype=patches.dtype)
    mask = torch.zeros((output_size[1], output_size[0]), device=patches.device, dtype=torch.float)
    mask_ones = torch.ones((output_size[1], output_size[0]), device=patches.device, dtype=torch.float)

    if len(coords.shape) == 1:
        coords = [coords]

    # Resize patches
    xmin, ymin, xmax, ymax = (
        int((coords[0][0] * output_size[0]).round()),
        int((coords[0][1] * output_size[1]).round()),
        int((coords[0][2] * output_size[0]).round()),
        int((coords[0][3] * output_size[1]).round()),
    )
    patches = F.interpolate(patches, (ymax - ymin, xmax - xmin), mode="bilinear", align_corners=False)

    # Ensemble patches
    for patch, (xmin, ymin, xmax, ymax) in zip(patches, coords):
        xmin, ymin, xmax, ymax = (
            int((xmin * output_size[0]).round()),
            int((ymin * output_size[1]).round()),
            int((xmax * output_size[0]).round()),
            int((ymax * output_size[1]).round()),
        )
        output[:, :, ymin:ymax, xmin:xmax] = patch
        mask[ymin + 10 : ymax - 10, xmin + 10 : xmax - 10] += 1.0

    # Handle overlapping regions
    output = output / torch.maximum(mask.unsqueeze(0).unsqueeze(0), mask_ones.unsqueeze(0).unsqueeze(0))
    mask = mask.type(torch.bool)

    return output, mask


def calculate_certainty(seg_probs):
    """Calculate the uncertainty of segmentation probability

    Args:
        seg_probs (torch.Tensor): B x C x H x W
            probability map of segmentation

    Returns:
        torch.Tensor: B x 1 x H x W
            uncertainty of input probability
    """
    top2_scores = torch.topk(seg_probs, k=2, dim=1)[0]
    res = (top2_scores[:, 0] - top2_scores[:, 1]).unsqueeze(1)
    return res


def get_uncertain_point_coords_on_grid(uncertainty_map, num_points):
    """Find `num_points` most uncertain points from `uncertainty_map` grid.

    Args:
        uncertainty_map (Tensor): A tensor of shape (N, 1, H, W) that contains uncertainty
            values for a set of points on a regular H x W grid.
        num_points (int): The number of points P to select.

    Returns:
        point_indices (Tensor): A tensor of shape (N, P) that contains indices from
            [0, H x W) of the most uncertain points.
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains [0, 1] x [0, 1] normalized
            coordinates of the most uncertain points from the H x W grid.
    """
    R, _, H, W = uncertainty_map.shape
    h_step = 1.0 / float(H)
    w_step = 1.0 / float(W)

    num_points = min(H * W, num_points)

    uncertainty_map = uncertainty_map.view(R, H * W)

    if num_points == H * W:
        point_indices = torch.arange(start=0, end=H * W, device=uncertainty_map.device).view(1, -1).repeat(R, 1)
    else:
        point_indices = torch.topk(uncertainty_map, k=num_points, dim=1, sorted=False)[1]

    point_coords = torch.zeros(R, num_points, 2, dtype=torch.float, device=uncertainty_map.device)

    point_coords[:, :, 0] = w_step / 2.0 + (point_indices % W).to(torch.float) * w_step
    point_coords[:, :, 1] = h_step / 2.0 + (point_indices // W).to(torch.float) * h_step

    # Sort indices
    sorted_values = torch.sqrt(point_coords[:, :, 0] ** 2 + point_coords[:, :, 1] ** 2)
    indices = torch.argsort(sorted_values, 1)

    for i in range(R):
        point_coords[i] = point_coords[i].gather(0, torch.stack([indices[i], indices[i]], dim=1))
        point_indices[i] = point_indices[i].gather(0, indices[i])

    return point_indices, point_coords


def point_sample(input, point_coords, **kwargs):
    """A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output
