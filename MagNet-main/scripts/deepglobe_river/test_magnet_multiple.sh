python test.py --dataset deepglobe_river \
--root data/deepglobe_river \
--datalist data/list/deepglobe_river/val_gravel.txt \
--scales 612-612,1224-1224,2448-2448 \
--crop_size 612 612 \
--input_size 508 508 \
--num_workers 8 \
--model fpn \
--pretrained backbone/exp21_same_classweight/deepglobe_river/resnet_fpn_train_612x612_sgd_lr1e-2_wd5e-4_bs_12_epoch484_gravel/swa_state.pth \
--pretrained_refinement runs_exp21/deepglobe_river_refinement_swa612x1224/30032022-132112/epoch50.pth  runs_exp21/deepglobe_river_refinement_swa1224x2448/30032022-113055/epoch50.pth \
--num_classes 6 \
--sub_batch_size 1 \
--n_points 0.75 \
--n_patches -1 \
--smooth_kernel 11 \
--save_pred \
--save_dir test_results/exp21_swa_deepglobe_river_val