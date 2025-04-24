# Download dataset, data split file, and checkpoint
WSI Dataset ([images](https://aced-idp.org/files/d92d360a-0fde-555c-8aa0-b86025c3d527), [masks]())

Trained foreground/background prediction models: 

1.512x512 model ([link](https://aced-idp.org/files/3647a4e3-5a48-57d7-807e-4f3b84eaa45d))

2.1024x1024 model ([link](https://aced-idp.org/files/8c2ea323-85c1-5749-ac0e-dbb2bf946ab3))

3.2048x2048 model ([link](https://aced-idp.org/files/00336cf5-80ea-5f8f-a0b2-924f3124f54d))
# Install dependency
```bash 
pip install --upgrade pip
pip install -r requirements.txt
```
# Run Automatic Inference

```bash 
python python inference_sam.py 
--data_dir "Path to dataset root"
--json_list "Path to json data split file"
--ckpt "Path to pretrained checkpoint"
--logdir "Path to log dir"
--label_prompt --enable_auto_branch --out_channels 16

```

# Run Sliding Window Inference

```bash 
python sliding_window_inference_sam.py 
--data_dir "Path to dataset root"
--json_list "Path to json data split file"
--ckpt "Path to pretrained checkpoint"
--logdir "Path to log dir"
--out_channels  2 # for background/ foreground prediction
# --out_channels  20 # for predicting all classes
--save_infer
--infer_only # This flag will enable inference only, so you don't have to provide GT.
--label_prompt 
--enable_auto_branch 
--window_size "the sliding window size(e.g., 512, 1024, 2048 depending on the checkpoint)"

```

# Run Fine-tuning

```bash 
python main_2d.py \
--distributed \
--data_dir ./example_training_dataset_patch_256 \
--json_list ./example_training_dataset_patch_256/example_data_list.json \
--sam_base_model vit_b \
--logdir vit_b_unfreeze_encoder_256_20c_8bkp_saug \
--save_checkpoint --max_epochs 200 \
--lrschedule warmup_cosine --warmup_epochs 5 --val_every 5 \
--data_aug --seed 0 \
--num_prompt 20 \
--out_channels 20 \
--max_bk_prompt 8 \
--label_prompt \
--enable_auto_branch \
--checkpoint /mnt/3td1/nvidia/Nvidia_080624_upload/256res_model_best.pt \ # path to the pretrained model
--optim_lr 1e-4 \ # tune lr to  5e-5, 1e-5 for finetuning
--actual_img_size 256 # tune this to 256, 512, 1024

```