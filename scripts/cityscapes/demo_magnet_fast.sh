python demo.py --dataset cityscapes \
--image $1 \
--scales 256-128,512-256,1024-512,2048-1024 \
--crop_size 256 128 \
--input_size 256 128 \
--model hrnet18+ocr \
--pretrained checkpoints/cityscapes_hrnet.pth \
--pretrained_refinement checkpoints/cityscapes_refinement_512.pth checkpoints/cityscapes_refinement_1024.pth checkpoints/cityscapes_refinement_2048.pth \
--num_classes 19 \
--n_points 0.9 \
--n_patches 4 \
--smooth_kernel 5 \
--save_pred \
--save_dir test_results/demo