python train.py --dataset deepglobe_river \
--root data/deepglobe_river \
--datalist data/list/hairs_magnet_v1/train.txt \
--scales 612-612,1224-1224,2448-2448 \
--crop_size 612 612 \
--input_size 508 508 \
--num_workers 8 \
--model fpn \
--pretrained backbone/exp609_hairs_bright_classweight_seed5/deepglobe_river/hairs_config_seed5/best.pth \
--num_classes 6 \
--batch_size 8 \
--task_name deepglobe_river_refinement \
--lr 0.001 \
--log_dir runs_exp609_seed5