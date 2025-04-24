#!/usr/bin/env bash

python main_2d.py \
--distributed \
--data_dir ./example_training_dataset_patch_256 \
--json_list ./example_training_dataset_patch_256/example_data_list.json \
--sam_base_model vit_b \
--logdir vit_b_unfreeze_encoder_256_20c_8bkp_saug \
--save_checkpoint --max_epochs 200 \
--lrschedule warmup_cosine --warmup_epochs 5 --val_every 5 \
--data_aug \
--seed 42 \
--num_prompt 20 \
--out_channels 20 \
--max_bk_prompt 8 \
--label_prompt \
--enable_auto_branch \
--checkpoint /mnt/3td1/nvidia/Nvidia_080624_upload/256res_model_best.pt \
--optim_lr 1e-4 \
--actual_img_size 256 # tune this to 256, 512, 1024

