# Running Inference and Fine-tuning
This guide provides instructions for performing inference and fine-tuning a Segment Anything Model (SAM)-based functional tissue unit (FTU) annotation model, which integrates with the MiroSCOPE platform for QuPath. Here, scripts for performing inference outside of the platform, and for fine-tuning custom checkpoints with new annotated data, are offered.

<br />

First, navigate to the training_inference directory:
```bash 
cd training_inference
```
<br />

## Install Dependencies
Create a new conda environment and install all required packages:
```bash 
conda create -n monai-env python=3.9 -y
conda activate monai-env

pip install --upgrade pip
pip install -r requirements.txt
```
<br />

## Inference (Sliding Window)

The sliding window approach is used to perform inference on large images, avoiding issues at tile borders.

<br />


**Download a pre-trained checkpoint**

Download a checkpoint from [Synapse](https://www.synapse.org/Synapse:syn66304443) or fine-tune a custom one (see fine-tuning section)

Checkpoints on Synapse:

| File Name                   | Description                                           | `--out_channels` |
|-----------------------------|-------------------------------------------------------|------------------|
| `Fg-bg2_prostate.pt`        | Prostate segmentation only                            | `2`              |
| `multiclass_prostate.pt`    | Prostate segmentation + classification                | `20`             |
| `Fg-bg_breast_pilot.pt`     | Breast segmentation                                   | `2`              |
| `multiclass_breast_pilot.pt`| Breast segmentation + classification                  | `14`             |

<br />

**Run inference**
```bash 
python sliding_window_inference_sam.py \
--data_dir "/path/to/images" \
--ckpt "/path/to/checkpoint.pt" \
--logdir "/path/to/output" \
--out_channels 20 \
--save_infer \
--infer_only \
--label_prompt \
--enable_auto_branch \
--window_size 1024 

```
Notes: 
- `--out_channels`: Number of classes used during fine-tuning (refer to checkpoint description)
- `--window_size`: Tile dimension used during training 

The output is a mask where each pixel value corresponds to the class id

<br />

## Fine-tuning

**Prepare data**

Before fine-tuning, images and corresponding annotation masks must be tiled, and a `data_list.json` file which tracks data paths must be created. Annotation files are expected to be in GeoJSON format.
```bash 
python ft_data_prep.py \
  --image_dir "/path/to/images" \
  --anno_dir "/path/to/annotations" \
  --save_dir "/path/to/training_data"
```
<br />

**Download checkpoints**

Download the base SAM [vit_b](https://github.com/facebookresearch/segment-anything) checkpoint 

Optionally, download a pre-trained checkpoint to fine-tune from [Synapse](https://www.synapse.org/Synapse:syn66304443) (checkpoints are listed in the inference section)

<br />

**Run fine-tuning**
```bash 
python main_2d.py \
--distributed \
--data_dir "/path/to/training_data" \
--json_list "/path/to/training_data/data_list.json" \
--sam_base_model vit_b \
--sam_pretrain_ckpt "/path/to/SAM_vit_b_checkpoint.pt" \
--checkpoint "/path/to/checkpoint.pt" \
--logdir "/path/to/output" \
--save_checkpoint --max_epochs 200 \
--lrschedule warmup_cosine --warmup_epochs 5 --val_every 5 \
--data_aug --seed 0 \
--num_prompt 20 \
--out_channels 20 \
--label_prompt \
--enable_auto_branch \
--optim_lr 1e-4 \
--actual_img_size 1024 \
--class_names "comma-delimited,string,of,class,names"

```
Notes: 
- `--logdir`: Path to output files, including model checkpoints, tensorboard logs, etc.
- `--sam_pretrain_ckpt`: Required, SAM vit_b checkpoint
- `--checkpoint`: Optional, custom pre-trained checkpoint to start from
- `--num_prompt`: Number of classes for prediction (include background as class 0)
- `--out_channels`: Number of classes for prediction (include background as class 0)
- `--actual_image_size`: Tile dimension 
- `--class_names`: List of class names (excluding background) as comma-delimited string, e.g. "normal,grade 1,grade 2"
