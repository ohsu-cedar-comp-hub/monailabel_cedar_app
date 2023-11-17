# Download dataset, data split file, and checkpoint
Please download those files from TBD.
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

# Run Sliding Window Inference (Point Prompt)

```bash 
python python sliding_window_inference_sam.py 
--data_dir "Path to dataset root"
--json_list "Path to json data split file"
--ckpt "Path to pretrained checkpoint"
--logdir "Path to log dir"
--out_channels 16
--points_val_pos 1
--points_val_neg 0
--save_infer
--point_prompt

```