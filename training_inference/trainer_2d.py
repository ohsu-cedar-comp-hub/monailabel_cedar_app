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

import os
import time

import numpy as np
import json
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather
from monai.data import decollate_batch
import torch.nn.functional as F
from copy import deepcopy
import random
from monai.metrics import compute_dice
import torch.distributed as dist


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


def prepare_sam_training_input(inputs, labels, args, model):
    unique_labels = torch.unique(labels).as_tensor().long()

    # remove some rare labels (16, 17, 18, 19)
    # mapping_index = unique_labels < args.out_channels
    # unique_labels = unique_labels[mapping_index]

    if args.skip_bk:
        unique_labels = unique_labels[1:]

    if len(unique_labels) == 0:
        return None, None, None, torch.ones(1).cuda(args.rank)

    # random sample args.num_prompt prompts, this will help to manage the GPU memory upper bound.
    if len(unique_labels) > args.num_prompt:
        idxs = random.sample(range(len(unique_labels)), args.num_prompt)
        idxs = torch.tensor(idxs)
        unique_labels = unique_labels[idxs]

    # add args.max_bk_prompt background labels to every batch
    background_labels = list(set([i for i in range(1, args.out_channels)]) - set(unique_labels.cpu().numpy()))
    random.shuffle(background_labels)
    if len(background_labels) > 0:
        unique_labels = torch.cat([unique_labels, torch.tensor(background_labels[:args.max_bk_prompt]).cuda(args.rank)])

    # preprocess make the size of label same as low_res_logit
    batch_labels_ = torch.stack([labels == unique_labels[i] for i in range(len(unique_labels))], dim=0).float()

    if args.distributed:
        batch_labels = model.module.preprocess(batch_labels_, is_input=False)
    else:
        batch_labels = model.preprocess(batch_labels_, is_input=False)

    # TODO: we currently only use class-label and points prompt.

    prepared_input = [{"image": inputs, "original_size": tuple(labels.shape)}]
    if args.label_prompt:
        labels_prompt = unique_labels.unsqueeze(-1)
        prepared_input[0]["labels"] = labels_prompt

    if args.point_prompt:
        point_coords, point_labels = generate_point_prompt(batch_labels_, args)
        prepared_input[0]["point_coords"] = point_coords
        prepared_input[0]["point_labels"] = point_labels

    if args.label_prompt and args.point_prompt:
        # if we use both two kinds of prompts, then we randomly drop one kind.
        if random.uniform(0, 1) < args.drop_label_prob:
            prepared_input[0].pop("labels")
        else:
            if random.uniform(0, 1) < args.drop_point_prob:
                prepared_input[0].pop("point_coords")
                prepared_input[0].pop("point_labels")

    return prepared_input, batch_labels.unsqueeze(1).cuda(args.rank), batch_labels_, torch.zeros(1).cuda(args.rank)


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    # we need to make sure the number of 2.5D input is an odd number.
    assert args.roi_z_iter % 2 == 1
    for idx, batch_data in enumerate(loader):
        # only take 1 batch
        inputs_l = batch_data["image"]
        labels_l = batch_data["label"]
        # TODO: we only support batch_size = 1 for data loader.
        inputs_l = inputs_l.squeeze()
        labels_l = labels_l.squeeze()

        data, target, target_original, skip = prepare_sam_training_input(inputs_l.cuda(args.rank),
                                                                         labels_l.cuda(args.rank), args, model)

        if torch.cuda.device_count() > 1:
            dist.all_reduce(skip, op=dist.ReduceOp.SUM)
        if skip[0] > 0:
            continue  # some rank has no foreground, skip this batch

        for param in model.parameters():
            param.grad = None

        with autocast(enabled=args.amp):
            outputs = model(data, is_train=True)
        loss = loss_func(outputs[0]["low_res_logits"], target)

        if args.amp:
            scaler.scale(loss).backward()
            if args.clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

        _loss = loss.detach()

        if args.distributed:
            loss_list = distributed_all_gather([_loss], out_numpy=True,
                                               )
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0),
                n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(_loss.item(), n=args.batch_size)
        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg


def train_epoch_iterative(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    # we need to make sure the number of 2.5D input is an odd number.
    assert args.roi_z_iter % 2 == 1
    for idx, batch_data in enumerate(loader):
        # only take 1 batch
        inputs_l = batch_data["image"]
        labels_l = batch_data["label"]
        # TODO: we only support batch_size = 1 for data loader.
        inputs_l = inputs_l.squeeze()
        labels_l = labels_l.squeeze()
        n_z_before_pad = labels_l.shape[-1]

        n_slice = args.roi_z_iter
        # pad the z direction, so we can easily extract 2.5D input and predict labels for the center slice
        pd = (n_slice // 2, n_slice // 2)
        inputs_l = F.pad(inputs_l, pd, "constant", 0)
        labels_l = F.pad(labels_l, pd, "constant", 0)
        _loss = torch.tensor(0.).cuda(args.rank)
        for _k in range(min(args.num_patch, n_z_before_pad)):
            # Return random integers from `low` (inclusive) to `high` (exclusive).
            start_idx = int(np.random.randint(low=n_slice // 2, high=(n_slice // 2 + n_z_before_pad)))

            inputs = inputs_l[..., start_idx - n_slice // 2:start_idx + n_slice // 2 + 1].permute(2, 0, 1)

            # we only need the label for the center slice
            labels = labels_l[..., start_idx - n_slice // 2:start_idx + n_slice // 2 + 1][..., n_slice // 2]

            data, target, target_original, skip = prepare_sam_training_input(inputs.cuda(args.rank),
                                                                             labels.cuda(args.rank),
                                                                             args, model)
            for param in model.parameters():
                param.grad = None

            with autocast(enabled=args.amp):
                if args.distributed:
                    image_embeddings = model.module.get_image_embeddings(data)
                else:
                    image_embeddings = model.get_image_embeddings(data)

            if skip:
                with autocast(enabled=args.amp):
                    if args.distributed:
                        outputs = model.module.get_mask_prediction(data, image_embeddings)
                    else:
                        outputs = model.get_mask_prediction(data, image_embeddings)
                loss = loss_func(outputs[0]["low_res_logits"], target) * 0.0
            else:
                # iterative training
                loss = 0
                drop_iter = random.randint(0, args.num_iterative_step - 2)
                for i in range(args.num_iterative_step):
                    with autocast(enabled=args.amp):
                        if args.distributed:
                            outputs = model.module.get_mask_prediction(data, image_embeddings)
                        else:
                            outputs = model.get_mask_prediction(data, image_embeddings)
                    loss += loss_func(outputs[0]["low_res_logits"], target)
                    if i == args.num_iterative_step - 1:
                        # no need to perform the following operations after the last step
                        continue
                    # we also supply the mask prediction from the previous iteration
                    # as an additional prompt to our model (follow original SAM).
                    data[0]["mask_inputs"] = outputs[0]["low_res_logits"].detach()
                    if i == drop_iter:
                        # for drop iter, no additional points are sampled (follow original SAM).
                        continue

                    previous_point_coords = data[0].get("point_coords", None)
                    previous_point_labels = data[0].get("point_labels", None)

                    if previous_point_coords is None and args.no_more_points_for_cp_only:
                        # if no point prompt at the first prompt generation,
                        # we will not add more additional pointa during iterative training.
                        continue

                    # sample one pos and on neg point based on previous prediction
                    previous_pred = (F.sigmoid(outputs[0]["high_res_logits"].detach()) > 0.5).float()
                    point_coords, point_labels = generate_point_prompt(target_original,
                                                                       args=args,
                                                                       points_pos=1,
                                                                       points_neg=1,
                                                                       previous_pred=previous_pred)

                    if previous_point_coords is not None:
                        data[0]["point_coords"] = torch.cat([previous_point_coords, point_coords], dim=1)
                        data[0]["point_labels"] = torch.cat([previous_point_labels, point_labels], dim=1)
                    else:
                        data[0]["point_coords"] = point_coords
                        data[0]["point_labels"] = point_labels

            if args.amp:
                scaler.scale(loss).backward()
                if args.clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()

            _loss += (loss.detach() / args.num_iterative_step)
        _loss /= min(args.num_patch, n_z_before_pad)
        if args.distributed:
            loss_list = distributed_all_gather([_loss], out_numpy=True,
                                               )
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0),
                n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(_loss.item(), n=args.num_patch)
        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg


def prepare_sam_test_input(inputs, labels, args, previous_pred=None):
    unique_labels = torch.tensor([i for i in range(0, args.out_channels)]).cuda(args.rank)

    # preprocess make the size of lable same as high_res_logit
    batch_labels = torch.stack([labels == unique_labels[i] for i in range(len(unique_labels))], dim=0).float()

    prepared_input = [{"image": inputs, "original_size": tuple(labels.shape)}]
    if args.label_prompt:
        labels_prompt = unique_labels.unsqueeze(-1)
        prepared_input[0].update(
            {"labels": labels_prompt})

    if args.point_prompt:
        point_coords, point_labels = generate_point_prompt(batch_labels, args, points_pos=args.points_val_pos,
                                                           points_neg=args.points_val_neg, previous_pred=previous_pred)
        prepared_input[0].update(
            {"point_coords": point_coords, "point_labels": point_labels})

    return prepared_input, batch_labels.unsqueeze(1).cuda(args.rank), unique_labels


def val_epoch(model, loader, epoch, acc_func, args, iterative=False, post_label=None, post_pred=None):
    model.eval()
    start_time = time.time()
    metric_dim = args.out_channels - 1
    # for metric, index 2*c is the dice for class c, and 2*c + 1 is the not-nan counts for class c
    metric = torch.zeros(metric_dim * 2, dtype=torch.float).cuda(args.rank)
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            # only take 1 batch
            inputs_l = batch_data["image"]
            labels_l = batch_data["label"]

            # TODO: we only support batch_size = 1 for data loader.
            inputs_l = inputs_l.squeeze()
            labels_l = labels_l.squeeze()

            data, target, _ = prepare_sam_test_input(inputs_l.cuda(args.rank), labels_l.cuda(args.rank), args)

            with autocast(enabled=args.amp):
                outputs = model(data)
                logit = outputs[0]["high_res_logits"]

            y_pred = torch.stack(post_pred(decollate_batch(logit)), 0)

            value = torch.full((1, metric_dim), float('nan')).cuda(args.rank)
            # TODO: we compute metric for each prompt for simplicity in validation.
            acc_batch = compute_dice(y_pred=y_pred.permute(1, 0, 2, 3),
                                     y=target.permute(1, 0, 2, 3),
                                     include_background=False)

            for _v_i, _v_c in enumerate(acc_batch[0]):
                value[0, _v_i] = _v_c

            for _c in range(metric_dim):
                val0 = torch.nan_to_num(value[0, _c], nan=0.0)
                val1 = 1.0 - torch.isnan(value[0, _c]).float()
                metric[2 * _c] += val0
                metric[2 * _c + 1] += val1

            f_name = batch_data["image"].meta["filename_or_obj"]
            print(f"Rank: {args.rank}, Case: {f_name}, Acc: {value.cpu().numpy().tolist()} ")

            if args.rank == 0:
                avg_acc = torch.nanmean(value)
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx + 1, len(loader)),
                    "acc",
                    avg_acc,
                    "time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()
    if torch.cuda.device_count() > 1:
        dist.all_reduce(metric, op=torch.distributed.ReduceOp.SUM)

    return metric


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_func,
        acc_func,
        args,
        scheduler=None,
        start_epoch=0,
        post_label=None,
        post_pred=None,
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0
    best_epoch = -1
    val_MA = None
    best_log = {}
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        if args.rank == 0:
            if scheduler is not None:
                print("Current lr:", scheduler.get_last_lr())
            else:
                print("Current lr:", optimizer.param_groups[0]['lr'])

        if args.label_prompt and args.point_prompt:
            if epoch < args.label_prompt_warm_up_epoch:
                # during warm up, we drop class label prompt embedding with less prob,
                # since class label prompt embedding layer is trained from scratch.
                args.drop_label_prob = 0.2
                args.drop_point_prob = 0.5
            else:
                # after warmp up, we evenly drop two kinds of prompts
                args.drop_label_prob = 0.5
                args.drop_point_prob = 0.5
            print("rank:", args.rank, "label_prompt (train):", args.label_prompt, ", label_drop_prob:",
                  args.drop_label_prob,
                  "| point_prompt (train):", args.point_prompt, ", point_drop_prob:", args.drop_point_prob)

        # we don't perform iterative training for the first args.iterative_training_warm_up_epoch epochs
        if epoch > args.iterative_training_warm_up_epoch:
            if args.reuse_img_embedding:
                if args.rank == 0:
                    print("Iterative Training: Reuse image embedding!")
                train_loss = train_epoch_iterative(
                    model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
                )
            else:
                if args.rank == 0:
                    print("Iterative Training: Don't reuse image embedding!")
                raise NotImplementedError
        else:
            print(f" Rank: {args.rank} Single-step Training")
            print("rank:", args.rank, "label_prompt (train):", args.label_prompt,
                  "| point_prompt (train):", args.point_prompt)
            train_loss = train_epoch(
                model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
            )

        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)

        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            if args.rank == 0:
                print("Start validation")
                print("label_prompt (val):", args.label_prompt, "point_prompt (val):", args.point_prompt)
            epoch_time = time.time()
            metric = val_epoch(
                model,
                val_loader,
                iterative=False,
                epoch=epoch,
                acc_func=acc_func,
                args=args,
                post_label=post_label,
                post_pred=post_pred
            )

            metric_dim = args.out_channels - 1
            metric = metric.tolist()
            metric_class = np.zeros(metric_dim)
            if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
                class_names = [
                    "Gleason pattern 3",
                    "Gleason pattern 4, cribriform glands",
                    "Gleason pattern 4, poorly-formed glands",
                    "Gleason pattern 4, fused glands",
                    "Gleason pattern 4, glomeruloid glands",
                    "Gleason pattern 5, single cells",
                    "Gleason pattern 5, sheets",
                    "Gleason pattern 5, cords",
                    "Gleason pattern 5, solid nests",
                    "Gleason pattern 5, necrosis",
                    "Prostatic intraepithelial neoplasia",
                    "Benign gland",
                    "Glandular atrophy",
                    "Nerve",
                    "Vein / artery",
                    "Chronic inflammation",
                    "Background stroma",
                    "Not segmented correctly",
                    "Tissue processing artifact",
                ]
                avg_metric = 0
                valid = 0
                for _c in range(metric_dim):
                    if metric[2 * _c + 1] > 0:
                        v = metric[2 * _c] / metric[2 * _c + 1]
                        avg_metric += v
                        valid += 1
                    else:
                        v = torch.nan
                    metric_class[_c] = v
                    try:
                        writer.add_scalar(f"val_class/acc_{_c + 1}_{class_names[_c]}", v, epoch)
                        print(f"Evaluation metric - class {_c + 1} {class_names[_c]}: {v:.4f}")
                    except BaseException:
                        writer.add_scalar(f"val_class/acc_{_c + 1}", v, epoch)
                        print(f"Evaluation metric - class {_c + 1} : {v:.4f}")

                val_avg_acc = avg_metric / valid

            if args.rank == 0:
                print(
                    "Final validation  {}/{},".format(epoch, args.max_epochs - 1),
                    f"Acc {val_avg_acc:.4f},",
                    "Previous Best validation at epoch {} is {:.4f},".format(best_epoch, val_acc_max),
                    "time {:.2f}s".format(time.time() - epoch_time),

                )
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    best_log[epoch] = float(val_acc_max)
                    best_epoch = epoch
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max, filename=f'model_best.pt', optimizer=optimizer,
                            scheduler=scheduler
                        )
                with open(os.path.join(args.logdir, 'train.log'), 'w') as f:
                    json.dump(best_log, f)
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint and args.save_ckpt_every_val:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename=f"model_e{epoch}.pt")

        if scheduler is not None:
            scheduler.step()

    if args.rank == 0 and writer is not None:
        writer.close()

    print("Training Finished !, Best Accuracy: ", val_acc_max, "at epoch", best_epoch)

    return val_acc_max
