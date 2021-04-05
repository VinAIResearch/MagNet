import numpy as np

def confusion_matrix(x, y, n, ignore_label=None, mask=None):
    if mask is None:
        mask = np.ones_like(x) == 1
    k = (x >= 0) & (y < n) & (x != ignore_label) & (mask.astype(np.bool))
    return np.bincount(n * x[k].astype(int) + y[k], minlength=n**2).reshape(n, n)

def getIoU(conf_matrix):
    if conf_matrix.sum() == 0:
        return 0
    with np.errstate(divide='ignore',invalid='ignore'):
        union = np.maximum(1.0, conf_matrix.sum(axis=1) + conf_matrix.sum(axis=0) - np.diag(conf_matrix))
        intersect = np.diag(conf_matrix)
        IU = intersect / union
        # IU = np.nan_to_num(np.diag(conf_matrix) / (conf_matrix.sum(1) + conf_matrix.sum(0) - np.diag(conf_matrix)).astype(np.float))
    return IU

def getFreq(conf_matrix):
    if conf_matrix.sum() == 0:
        return 0
    with np.errstate(divide='ignore',invalid='ignore'):
        freq = conf_matrix.sum(axis=1) / conf_matrix.sum()
    return freq

def get_mean_iou(conf_mat, dataset):
    IoU = getIoU(conf_mat)
    if dataset == "deepglobe":
        return np.nanmean(IoU[1:])
    elif dataset == ["gleason", "cityscapes"]:
        return np.nanmean(IoU)

def get_freq_iou(conf_mat, dataset):
    IoU = getIoU(conf_mat)
    freq = getFreq(conf_mat)
    if dataset == "deepglobe":
        return (IoU[1:] * freq[1:]).sum()/ freq[1:].sum()
    elif dataset in ["gleason", "cityscapes"]:
        return (IoU * freq).sum()

def get_overall_iou(conf_mat, dataset):
    if dataset in ["deepglobe", "cityscapes":
        return get_mean_iou(conf_mat, dataset)
    elif dataset == "gleason":
        return get_freq_iou(conf_mat, dataset)
    else:
        raise "Not implementation for dataset {}".format(dataset)