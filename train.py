import os
import json
from datetime import datetime

import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.ops import roi_align

from tensorboardX import SummaryWriter

from magnet.dataset import get_dataset_with_name
from magnet.options.train import TrainOptions
from magnet.model import get_model_with_name
from magnet.model.refinement import RefinementMagNet
from magnet.utils.loss import OhemCrossEntropy
from magnet.utils.metrics import get_freq_iou, get_mean_iou, confusion_matrix, get_overall_iou
from magnet.utils.geometry import calculate_uncertainty, get_uncertain_point_coords_on_grid, point_sample

def main():

    # Parse arguments
    opt = TrainOptions().parse()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Create logger
    date_time = datetime.now().strftime("%d%m%Y-%H%M%S")
    log_dir = os.path.join(opt.log_dir, opt.task_name, date_time)
    writer = SummaryWriter(logdir=log_dir)

    # Save config
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        f.write(json.dumps(vars(opt), indent=4))

    # Create dataset
    dataset = get_dataset_with_name(opt.dataset)(opt)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    # Create model
    model = get_model_with_name(opt.model)(opt.num_classes).to(device)
    print("Load model weight from", opt.pretrained)
    state_dict = torch.load(opt.pretrained)
    model.load_state_dict(state_dict)
    model.eval()

    # Create refinement module
    refinement_model = RefinementMagNet(opt.num_classes, use_bn=True).to(device)
    if os.path.isfile(opt.pretrained_refinement):
        print("Load refinement weight from", opt.pretrained_refinement)
        state_dict = torch.load(opt.pretrained_refinement)
        refinement_model.load_state_dict(state_dict, strict=False)

    print("Number of training parameters:", sum(p.numel() for p in refinement_model.parameters() if p.requires_grad))

    # Create optimizer
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, refinement_model.parameters()), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.decay)

    # Create learning rate scheduler
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)

    criteria = OhemCrossEntropy(ignore_label=dataset.ignore_label)
    global_step = 0
    for epoch in range(opt.epochs):

        # Training
        refinement_model.train()

        pbar = tqdm(total=len(dataloader))

        epoch_mat_coarse = np.zeros((opt.num_classes, opt.num_classes), dtype=np.float)
        epoch_mat_fine = np.zeros((opt.num_classes, opt.num_classes), dtype=np.float)
        epoch_mat_aggre = np.zeros((opt.num_classes, opt.num_classes), dtype=np.float)
        mean_loss = []

        for idx, data in enumerate(dataloader):
            coarse_image = data["coarse_image"].to(device)
            fine_image = data["fine_image"].to(device)
            fine_label = data["fine_label"].to(device)
            coords = [x for x in data["coord"].to(device)]

            # Get early predictions
            with torch.no_grad():
                coarse_pred = model(coarse_image).softmax(1)
                fine_pred = model(fine_image).softmax(1)

            # Crop preds
            # import pdb; pdb.set_trace()
            crop_preds = roi_align(coarse_pred, coords, output_size=(opt.input_size[1], opt.input_size[0]))

            # Refinement forward
            optimizer.zero_grad()
            logits = refinement_model(crop_preds, fine_pred)

            # Calculate loss
            loss = criteria(logits, fine_label)
            loss.backward()
            optimizer.step()
            description = "loss: %.2f, " % (loss)
            mean_loss += [float(loss)]
            writer.add_scalar("step_loss", loss, global_step)

            description += "lr: " + str(optimizer.param_groups[0]["lr"]) + ", "

            # Calculate confusion matrix
            fine_label = fine_label.cpu().numpy()
            coarse_mat = confusion_matrix(fine_label, crop_preds.argmax(1).cpu().numpy(), opt.num_classes, ignore_label=dataset.ignore_label)
            epoch_mat_coarse += coarse_mat
            fine_mat = confusion_matrix(fine_label, fine_pred.argmax(1).cpu().numpy(), opt.num_classes, ignore_label=dataset.ignore_label)
            epoch_mat_fine += fine_mat

            # Aggregate features
            with torch.no_grad():
                uncertainty_score = calculate_uncertainty(crop_preds)
                certainty_score = 1.0 - calculate_uncertainty(fine_pred)

                error_score = certainty_score * uncertainty_score

                b, c, h, w = crop_preds.shape

                n_points = int(h * w / 2)
                error_point_indices, error_point_coords = get_uncertain_point_coords_on_grid(error_score, n_points)
                error_point_indices = error_point_indices.unsqueeze(1).expand(-1, opt.num_classes, -1)
                alter_pred = point_sample(logits.softmax(1), error_point_coords, align_corners=False)
                
                aggre_pred = (
                            crop_preds.reshape(b, c, h * w)
                            .scatter_(2, error_point_indices, alter_pred)
                            .view(b, c, h, w)
                        )

                aggre_mat = confusion_matrix(fine_label, aggre_pred.argmax(1).cpu().numpy(), opt.num_classes, ignore_label=dataset.ignore_label)
                epoch_mat_aggre += aggre_mat

            IoU_coarse = get_freq_iou(coarse_mat, opt.dataset)
            description += "IoU coarse: %.2f, " %(IoU_coarse * 100)

            IoU_fine = get_freq_iou(fine_mat, opt.dataset)
            description += "IoU fine: %.2f, " %(IoU_fine* 100)
            
            IoU_aggre = get_freq_iou(aggre_mat, opt.dataset)
            description += "IoU aggre: %.2f" %(IoU_aggre* 100)

            writer.add_scalars("step_IoU", {"coarse": IoU_coarse, "fine": IoU_fine, "aggre": IoU_aggre}, global_step)
            
            description = "Epoch {}/{}: ".format(epoch+1, opt.epochs) + description

            pbar.set_description(description)
            pbar.update(1)
            global_step += 1
        
        lr_scheduler.step()

        # Log epoch loss, lr, IoU
        writer.add_scalar("epoch_loss", sum(mean_loss)/len(mean_loss), global_step=epoch+1)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step=epoch+1)
        writer.add_scalars("epoch_IoU", {
            "coarse": get_overall_iou(epoch_mat_coarse, opt.dataset),
            "fine": get_overall_iou(epoch_mat_fine, opt.dataset),
            "aggre": get_overall_iou(epoch_mat_aggre, opt.dataset)
        }, 
        global_step=epoch + 1)

        # Save model
        torch.save(refinement_model.state_dict(), os.path.join(log_dir, "epoch{}.pth".format(epoch + 1)))

if __name__ == "__main__":
    main()