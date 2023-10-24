# Download dataset, data split file, and checkpoint
Please download those files from [Google Drive](https://drive.google.com/drive/folders/1qlsKfEzV5EOP5uO-dppcZZR1KgEByMxb?usp=sharing).
# Install dependency
```bash 
pip install --upgrade pip
pip install -r requirements.txt
```
# Run Automatic inference

```bash 
python python inference_sam.py 
--data_dir "Path to dataset root"
--json_list "Path to json data split file"
--ckpt "Path to pretrained checkpoint"
--logdir "Path to log dir"
--label_prompt --enable_auto_branch --out_channels 16

```