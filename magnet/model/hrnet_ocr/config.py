from model.base import BasicBlock, Bottleneck

SMALL_CONFIG = {
    "FINAL_CONV_KERNEL": 1,
    "STAGE1": {
        "NUM_MODULES": 1,
        "NUM_BRANCHES": 1,
        "BLOCK": "BOTTLENECK",
        "NUM_BLOCKS": [1],
        "NUM_CHANNELS": [32],
        "FUSE_METHOD": "SUM"
    },
    "STAGE2": {
        "NUM_MODULES": 1,
        "NUM_BRANCHES": 2,
        "BLOCK": "BASIC",
        "NUM_BLOCKS": [2, 2],
        "NUM_CHANNELS": [16, 32],
        "FUSE_METHOD": "SUM"
    },
    "STAGE3": {
        "NUM_MODULES": 1,
        "NUM_BRANCHES": 3,
        "BLOCK": "BASIC",
        "NUM_BLOCKS": [2, 2, 2],
        "NUM_CHANNELS": [16, 32, 64],
        "FUSE_METHOD": "SUM"
    },
    "STAGE4": {
        "NUM_MODULES": 1,
        "NUM_BRANCHES": 4,
        "BLOCK": "BASIC",
        "NUM_BLOCKS": [2, 2, 2, 2],
        "NUM_CHANNELS": [16, 32, 64, 128],
        "FUSE_METHOD": "SUM"
    },
}

SMALL_CONFIG_V2 = {
    "FINAL_CONV_KERNEL": 1,
    "STAGE1": {
        "NUM_MODULES": 1,
        "NUM_BRANCHES": 1,
        "BLOCK": "BOTTLENECK",
        "NUM_BLOCKS": [2],
        "NUM_CHANNELS": [64],
        "FUSE_METHOD": "SUM"
    },
    "STAGE2": {
        "NUM_MODULES": 1,
        "NUM_BRANCHES": 2,
        "BLOCK": "BASIC",
        "NUM_BLOCKS": [2, 2],
        "NUM_CHANNELS": [18, 36],
        "FUSE_METHOD": "SUM"
    },
    "STAGE3": {
        "NUM_MODULES": 3,
        "NUM_BRANCHES": 3,
        "BLOCK": "BASIC",
        "NUM_BLOCKS": [2, 2, 2],
        "NUM_CHANNELS": [18, 36, 72],
        "FUSE_METHOD": "SUM"
    },
    "STAGE4": {
        "NUM_MODULES": 2,
        "NUM_BRANCHES": 4,
        "BLOCK": "BASIC",
        "NUM_BLOCKS": [2, 2, 2, 2],
        "NUM_CHANNELS": [18, 36, 72, 144],
        "FUSE_METHOD": "SUM"
    },
}

LARGE_CONFIG = {
    "FINAL_CONV_KERNEL": 1,
    "STAGE1": {
        "NUM_MODULES": 1,
        "NUM_BRANCHES": 1,
        "BLOCK": "BOTTLENECK",
        "NUM_BLOCKS": [4],
        "NUM_CHANNELS": [64],
        "FUSE_METHOD": "SUM"
    },
    "STAGE2": {
        "NUM_MODULES": 1,
        "NUM_BRANCHES": 2,
        "BLOCK": "BASIC",
        "NUM_BLOCKS": [4, 4],
        "NUM_CHANNELS": [48, 96],
        "FUSE_METHOD": "SUM"
    },
    "STAGE3": {
        "NUM_MODULES": 4,
        "NUM_BRANCHES": 3,
        "BLOCK": "BASIC",
        "NUM_BLOCKS": [4, 4, 4],
        "NUM_CHANNELS": [48, 96, 192],
        "FUSE_METHOD": "SUM"
    },
    "STAGE4": {
        "NUM_MODULES": 3,
        "NUM_BRANCHES": 4,
        "BLOCK": "BASIC",
        "NUM_BLOCKS": [4, 4, 4, 4],
        "NUM_CHANNELS": [48, 96, 192, 384],
        "FUSE_METHOD": "SUM"
    },
}

blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}

config_hrnet_w18_ocr = {
    "ALIGN_CORNERS": True,
    "NAME": "seg_hrnet_ocr",
    "NUM_OUTPUTS": 2,
    "OCR": {
        "DROPOUT": 0.05,
        "KEY_CHANNELS": 256,
        "MID_CHANNELS": 512,
        "SCALE": 1
    },
    "EXTRA": SMALL_CONFIG_V2
}

config_hrnet_w48_ocr = {
    "ALIGN_CORNERS": True,
    "NAME": "seg_hrnet_ocr",
    "NUM_OUTPUTS": 2,
    "OCR": {
        "DROPOUT": 0.05,
        "KEY_CHANNELS": 256,
        "MID_CHANNELS": 512,
        "SCALE": 1
    },
    "EXTRA": LARGE_CONFIG
}