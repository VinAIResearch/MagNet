python test.py --dataset gleason \
--root data/gleason \
--datalist data/list/gleason/val.txt \
--scales 625-625,1250-1250,2500-2500,5000-5000 \
--crop_size 625 625 \
--input_size 625 625 \
--num_workers 4 \
--model psp \
--pretrained checkpoints/gleason_psp.pth \
--pretrained_refinement checkpoints/epoch24.pth \
--num_classes 5 \
--sub_batch_size 1 \
--n_points 1.0 \
--n_patches -1 \
--smooth_kernel 11 \
--save_pred \
--save_dir test_results/gleason