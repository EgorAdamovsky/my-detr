# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import cv2
import torch
from matplotlib import pyplot as plt
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from torchvision.transforms import functional as F


# ТРЕНИРОВКА 1 ЭПОХА
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        ## ВИЗУАЛИЗАЦИЯ ################################################################################################
        # img = samples.tensors[0].permute(1, 2, 0).cpu().numpy()
        # arr_min = img.min()
        # arr_max = img.max()
        # img = (img - arr_min) / (arr_max - arr_min)
        # plt.imsave('D:/output.png', img, format='png')
        ################################################################################################################

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        # print(res)
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator


def video_to_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames, fps


def prepare_image(image):
    image = F.to_tensor(image).type('torch.FloatTensor')  # Converts to [0,1], (H,W,C) -> (C,H,W)
    image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet norm
    image = image.unsqueeze(0)  # Добавляем batch dim: [1, C, H, W]
    return image


def save_frames_to_video(frames, output_path, fps=30, frame_size=None):
    """
    Сохраняет список кадров в видеофайл.

    :param frames: Список кадров (np.ndarray), RGB или BGR.
    :param output_path: Путь для сохранения выходного видео.
    :param fps: Частота кадров.
    :param frame_size: Размер кадра (ширина, высота). Если None, берётся размер первого кадра.
    """
    if frames and len(frames[0].shape) == 3 and frames[0].shape[2] == 3:
        converted_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames]
    else:
        converted_frames = frames  # если уже в BGR
    if not converted_frames:
        raise ValueError("Список кадров пустой.")
    height, width = converted_frames[0].shape[:2]
    if frame_size:
        width, height = frame_size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # кодек для .mp4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in converted_frames:
        out.write(frame)
    out.release()
    print(f"Видео успешно сохранено: {output_path}")


@torch.no_grad()
def evaluate_video(model, criterion, postprocessors, data_loader, base_ds, device, args):
    model.eval()
    criterion.eval()
    frames, fps = video_to_frames(args.path_video)
    for f in range(len(frames)):
        if f >= args.prevs:
            samples = prepare_image(frames[f])
            for p in range(args.prevs):
                samples = torch.cat([samples, prepare_image(frames[f - p])], dim=1, out=None)
            samples = samples.to(device)
            out = model(samples)
            pass
            img = frames[f]
            s = out["pred_boxes"].size()[1]
            wh = img.shape
            counter = 0
            for i in range(min(s, 500)):
                x = int(wh[1] * float(out["pred_boxes"][0, i, 0].cpu().detach().numpy()))
                y = int(wh[0] * float(out["pred_boxes"][0, i, 1].cpu().detach().numpy()))
                w = int(wh[1] * float(out["pred_boxes"][0, i, 2].cpu().detach().numpy()))
                h = int(wh[0] * float(out["pred_boxes"][0, i, 3].cpu().detach().numpy()))
                max_index = torch.argmax(out['pred_logits'][0][i]).item()
                color = (128, 128, 128)
                if max_index == 1:
                    color = (255, 0, 0)
                elif max_index == len(out['pred_logits'][0][i]) - 1:
                    color = (0, 255, 0)
                thickness = 2 if max_index == 1 else 1
                if max_index == 1:
                    counter += 1
                cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), color, thickness)
            if args.path_video_out != "":
                frames[f] = img
            else:
                cv2.imshow("FRAME", img)
                cv2.waitKey(1)
            print("Кадр: " + str(f))
    if args.path_video_out != "":
        save_frames_to_video(frames, args.output_dir + "\\" + args.path_video_out, fps=int(fps))
