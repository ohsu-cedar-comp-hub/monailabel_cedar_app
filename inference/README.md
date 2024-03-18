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

# Run Sliding Window Inference (Automatic fg/bk prediction)

```bash 
python sliding_window_inference_sam.py 
--data_dir "Path to dataset root"
--json_list "Path to json data split file"
--ckpt "Path to pretrained checkpoint"
--logdir "Path to log dir"
--out_channels 2
--save_infer
--infer_only # This flag will enable inference only, so you don't have to provide GT.
--label_prompt 
--enable_auto_branch 
--window_size "the sliding window size(e.g., 512, 1024, 2048 depending on the checkpoint)"

```