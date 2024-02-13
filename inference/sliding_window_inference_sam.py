# Copyright 2020 - 2023 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random
from copy import deepcopy
import json
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000
import cv2
import monai
import numpy as np
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
from pathlib import Path
from utils.data_utils import split_data
from monai.data import (
    partition_dataset,
    DataLoader,
    decollate_batch,
)
import argparse
from functools import partial
from monai.metrics import compute_dice
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    EnsureType,
    CastToTyped,
)
from monai.data.image_reader import PILReader
from vista_2d.model import sam_model_registry
from torch.cuda.amp import autocast
from monai.utils import set_determinism
from utils.monai_utils import sliding_window_inference

parser = argparse.ArgumentParser(description="VISTA 2D segmentation pipeline")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--roi_z_iter", default=1, type=int, help="roi size in z direction")
parser.add_argument("--out_channels", default=20, type=int, help="number of output channels")
parser.add_argument("--data_dir", default="/dataset/dataset0/", type=str, help="dataset directory")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument("--fold", default=0, type=int, help="fold")
parser.add_argument("--data_aug", action="store_true", help="using data augmentation in training")
parser.add_argument("--splitval", default=0, type=float,
                    help="if not zero, split the last portion to validation and validation to test")
parser.add_argument("--metric", action="store_true", help="only do metric computing based on logdir")
parser.add_argument("--sam_pretrain_ckpt", type=str, default=None,
                    help="sam_pretrain_ckpt")
parser.add_argument("--sam_base_model", type=str, default="vit_b",
                    help="sam_pretrain_ckpt")
parser.add_argument("--sam_image_size", type=int, default=1024,
                    help="sam input res")
parser.add_argument("--label_prompt", action="store_true", help="using class label prompt in training")
parser.add_argument("--point_prompt", action="store_true", help="using point prompt in training")
parser.add_argument("--max_points", default=8, type=int, help="number of max point prompts")
parser.add_argument("--points_val_pos", default=1, type=int, help="number of positive point prompts in evaluation")
parser.add_argument("--points_val_neg", default=0, type=int, help="number of negative point prompts in evaluation")
parser.add_argument("--logdir", default="/mnt/3td1/ohsu_sam_results/sam",
                    type=str, help="directory to save the eval results")
parser.add_argument("--ckpt", default="./runs/model_best.pt", type=str,
                    help="model ckpts")
parser.add_argument("--save_infer", action="store_true", help="save inference results")
parser.add_argument("--patch_embed_3d", action="store_true", help="using 3d patch embedding layer")
parser.add_argument("--use_all_files_for_val", action="store_true", help="used in validating original SAM")
parser.add_argument("--enable_auto_branch", action="store_true", help="enable automatic prediction")
parser.add_argument("--seed", default=0, type=int, help="seed")
parser.add_argument("--infer_only", action="store_true", help="only conduct inference and skip metric calculation")



def main():
    args = parser.parse_args()
    set_determinism(seed=args.seed)
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)


def main_worker(gpu, args):
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)

    args.gpu = gpu
    world_size = 1
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
        torch.cuda.set_device(args.gpu)
        torch.backends.cudnn.benchmark = True
        world_size = dist.get_world_size()

    runs = args.logdir
    Path(runs).mkdir(parents=True, exist_ok=True)

    model = sam_model_registry[args.sam_base_model](checkpoint=None,
                                                    image_size=args.sam_image_size,
                                                    encoder_in_chans=args.roi_z_iter * 3,
                                                    patch_embed_3d=args.patch_embed_3d,
                                                    enable_auto_branch=args.enable_auto_branch
                                                    )

    train_files, val_files, test_files = split_data(args)
    # validation data
    if args.rank == 0:
        print('test files', len(test_files), [os.path.basename(_['image']).split('.')[0] for _ in test_files])

    if args.use_all_files_for_val:
        test_files = train_files + test_files

    test_files = partition_dataset(
        data=test_files, shuffle=False, num_partitions=world_size, even_divisible=False
    )[args.rank]

    if args.infer_only:
        keys = ["image"]
    else:
        keys = ["image", "label"]
    val_transforms = Compose([
        LoadImaged(keys=keys, reader=PILReader, image_only=True),
        EnsureChannelFirstd(keys=keys),
        CastToTyped(keys=keys, dtype=[torch.uint8] if args.infer_only else [torch.uint8, torch.uint8]),
    ])

    test_ds = monai.data.Dataset(
        data=test_files, transform=val_transforms
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=8, pin_memory=True
    )

    device = f'cuda:{args.rank}'

    post_pred = Compose([EnsureType(), AsDiscrete(threshold=0.0, dtype=torch.uint8)])
    model_dict = torch.load(args.ckpt, map_location=device)
    if "state_dict" in model_dict.keys():
        model_dict = model_dict["state_dict"]

    model.load_state_dict(model_dict, strict=True)
    print(f"Load {len(model_dict)} keys from checkpoint {args.ckpt}, current model has {len(model.state_dict())} keys")

    model.cuda(args.gpu)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu,
                                                          find_unused_parameters=True)

    model.eval()
    dice_ = torch.zeros([1, args.out_channels - 1], device=device)
    count_ = torch.zeros([1, args.out_channels - 1], device=device)
    with torch.no_grad():
        for idx, batch_data in enumerate(test_loader):
            if args.infer_only:
                labels_l = None
            else:
                # only take 1 batch
                labels_l = batch_data["label"].as_subclass(torch.Tensor)[:, :1, ...]
                # remove some rare labels (16, 17, 18, 19)
                mapping_index = labels_l >= args.out_channels
                if mapping_index.any():
                    labels_l[mapping_index] = 0

            file_name = batch_data["image"].meta['filename_or_obj'][0].split("/")[-1].split(".")[0]

            _device_in = "cpu"
            _device_out = "cpu"
            with autocast(enabled=True):
                val_outputs = None
                torch.cuda.empty_cache()
                val_outputs = sliding_window_inference(
                    inputs=batch_data["image"].as_subclass(torch.Tensor).half().to(_device_in),
                    roi_size=[512, 512],
                    sw_batch_size=1,
                    predictor=model,
                    mode="gaussian",
                    overlap=0.25,
                    sw_device=device,
                    device=_device_out,
                    labels=labels_l.to(_device_in) if labels_l is not None else labels_l,
                    progress=True,
                    val_point_sampler=partial(prepare_sam_test_input,
                                              args=args
                                              )
                )

            y_pred = torch.stack(post_pred(decollate_batch(val_outputs)))

            if not args.infer_only:
                dice = compute_dice(
                    y_pred=y_pred,
                    y=labels_l,
                    num_classes=args.out_channels,
                    include_background=False
                )
                print(f"Rank: {args.rank}, Done[{idx + 1}/{len(test_loader)}], {file_name}")
                print(dice)
                nan_idx = torch.isnan(dice)
                dice_ += torch.nan_to_num(dice).to(device)
                count = torch.ones_like(dice).to(device)
                count[nan_idx] = 0
                count_ += count

            if args.save_infer:
                val_outputs = val_outputs * torch.cat([torch.zeros(1, 1, *val_outputs.shape[-2:]),
                                                  torch.ones(1, val_outputs.shape[1]-1, *val_outputs.shape[-2:])],
                                                  dim=1)
                val_outputs = (val_outputs > 0.5).half() * val_outputs
                save_np = np.transpose(torch.argmax(val_outputs, dim=1).squeeze().cpu().numpy().astype(np.uint8))

                cv2.imwrite(os.path.join(args.logdir, file_name+"_pred.tif"), save_np)

    if args.distributed:
        dist.barrier()
        dist.all_reduce(dice_, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(count_, op=torch.distributed.ReduceOp.SUM)

    if args.rank == 0:
        f = open(os.path.join(args.logdir, f'metric_pp{args.points_val_pos}_np{args.points_val_neg}.json'), 'w')
        metric = {}
        mean_dice_batch = dice_ / count_

        mean_dice_class = torch.nanmean(mean_dice_batch, dim=0).cpu().numpy().tolist()
        mean_dice_all = torch.nanmean(mean_dice_batch).cpu().numpy().tolist()
        metric['mean_dice'] = mean_dice_class
        metric['all_mean_dice'] = mean_dice_all

        json.dump(metric, f)
        f.close()
        print('mean dice class:', mean_dice_class)
        print('dice final', mean_dice_all)

def prepare_sam_test_input(inputs, labels, args, previous_pred=None):
    unique_labels = torch.tensor([i for i in range(0, args.out_channels)]).cuda(args.rank)

    if labels is not None:
        # preprocess make the size of lable same as high_res_logit
        batch_labels = torch.stack([labels == unique_labels[i] for i in range(len(unique_labels))], dim=0).float()
    else:
        batch_labels = torch.zeros(1)

    prepared_input = [{"image": inputs, "original_size": tuple(inputs.shape)[1:]}]
    if args.label_prompt:
        labels_prompt = unique_labels.unsqueeze(-1)
        prepared_input[0].update(
            {"labels": labels_prompt})

    if args.point_prompt:
        # need labels to simulate user's click when doing interactive inference
        assert labels is not None
        point_coords, point_labels = generate_point_prompt(batch_labels, args, points_pos=args.points_val_pos,
                                                           points_neg=args.points_val_neg, previous_pred=previous_pred)
        prepared_input[0].update(
            {"point_coords": point_coords, "point_labels": point_labels})

    return prepared_input, batch_labels.unsqueeze(1).cuda(args.rank), unique_labels

def apply_coords_torch(coords, original_size, sam_image_size) -> np.ndarray:
    """
    Expects a numpy array of length 2 in the final dimension. Requires the
    original image size in (H, W) format.
    """
    old = original_size
    new = sam_image_size
    coords = deepcopy(coords).float()
    # Here, we can apply a same scale factor to h and w, because we first pad the input to a square image along the
    # longest side then resize it to sam_image_size. In other words, the scale factor is determined by the longest side.
    coords[..., 0] = coords[..., 0] * (new / old)
    coords[..., 1] = coords[..., 1] * (new / old)
    return coords


def sample_points(labelpoints, n_points):
    idx = torch.randperm(len(labelpoints), dtype=torch.long, device=labelpoints.device)[:n_points]
    return [labelpoints[idx]]


def generate_point_prompt(batch_labels_, args, points_pos=None, points_neg=None, previous_pred=None):
    max_point = args.max_points
    Np = points_pos if points_pos is not None else min(max_point,
                                                       int(np.abs(random.gauss(mu=0, sigma=max_point // 2))) + 1)
    Nn = points_neg if points_neg is not None else min(max_point, int(np.abs(random.gauss(mu=0, sigma=max_point // 2))))
    # To follow original SAM, with equal probability either a foreground point
    # is selected randomly for the target mask
    _point = []
    _point_label = []
    b, h, w = batch_labels_.shape
    device = batch_labels_.device
    for i in range(b):
        plabels = batch_labels_[i, ...]
        if torch.sum(plabels) == 0:
            # bk prompt
            n_placeholder = Np + Nn
            _point.append(torch.cat([torch.zeros((1, 2), device=device)] * n_placeholder, dim=0))
            _point_label.append(torch.tensor([-1] * n_placeholder).to(device))
            continue

        nlabels = (plabels == 0.0).float()
        if previous_pred is not None:
            ppred = previous_pred[i, 0, ...]
            npred = (previous_pred[i, 0, ...] == 0.0).float()

            # False positive mask (pixels that are predicted as positive but are actually negative)
            fp_mask = torch.logical_and(nlabels, ppred)
            # False negative mask (pixels that are predicted as negative but are actually positive)
            fn_mask = torch.logical_and(plabels, npred)
            # we sample positive points from false negative pred.
            # we sample negative points from false positive pred.
            plabelpoints = torch.nonzero(fn_mask)
            nlabelpoints = torch.nonzero(fp_mask)

        else:
            plabelpoints = torch.nonzero(plabels)
            nlabelpoints = torch.nonzero(nlabels)
        # 1 indicates a foreground point and 0 indicates a background point.
        # -1 indicates a dummy non-point as the placeholder.
        n_placeholder = (Np + Nn - min(len(plabelpoints), Np) - min(len(nlabelpoints), Nn))

        # Use torch.randperm to generate indices on a GPU tensor
        _point.append(torch.cat(
            sample_points(plabelpoints, min(len(plabelpoints), Np))
            +
            sample_points(nlabelpoints, min(len(nlabelpoints), Nn))
            +
            [torch.zeros((1, 2), device=device)] * n_placeholder, dim=0))
        _point_label.append(torch.tensor(
            [1] * min(len(plabelpoints), Np) + [0] * min(len(nlabelpoints), Nn) + [-1] * n_placeholder).to(device))

    point = torch.stack(_point)
    point_label = torch.stack(_point_label)
    point_coords = apply_coords_torch(point, max(h, w), args.sam_image_size)

    return point_coords, point_label
if __name__ == "__main__":
    main()
