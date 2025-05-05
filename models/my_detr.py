import random
import time

import cv2
import torch
import torch.nn.functional as F
from torch import nn
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
# matplotlib.use('TkAgg')


# НАША ОСНОВНАЯ МОДЕЛЬ
class DETR(nn.Module):

    def __init__(self, backbone, transformer, num_classes, num_queries, args):
        super().__init__()
        self.num_queries = num_queries  # количество запросов
        self.transformer = transformer  # трансформер
        hidden_dim = transformer.d_model  # размерность скрытых состояний трансформера (обычно 256)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # линейный слой для предсказания классов
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)  # полносвязная сеть для предсказания боксов
        self.query_embed = nn.Embedding(num_queries, hidden_dim)  # эмбеддинги запросов
        self.args = args

        ## ЭКСПЕРИМЕНТ 2 ###############################################################################################
        ## увеличивается глубина
        backbone_num_channels = int(backbone.num_channels * (args.prevs + 1))
        ################################################################################################################

        backbone_hidden_dim = int(hidden_dim)
        self.input_proj = nn.Conv2d(backbone_num_channels, backbone_hidden_dim,
                                    kernel_size=1)  # карта признаков -> размерность Transformer'а
        self.backbone = backbone  # CNN

    # ПРОГОН ДАННЫХ
    def forward(self, samples: NestedTensor):

        ## ЭКСПЕРИМЕНТ 1 ###############################################################################################
        ## имитирую 8 кадров, которые проходят через backbone по очереди
        ## конкатенирую их в один большой тензор
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)  # тензор -> NestedTensor
        split_tensors = torch.tensor_split(samples.tensors, (self.args.prevs + 1), dim=1)
        split_samples = []
        for i in range(len(split_tensors)):
            split_samples.append(NestedTensor(split_tensors[i], samples.mask))
        src_prepared = None
        pos_prepared = None
        mask = None
        for i in range((self.args.prevs + 1)):
            features, pos = self.backbone(split_samples[i])  # карты признаков разных уровней от 1 до N
            src, mask = features[-1].decompose()  # разделяет последнюю карту признаков на саму карту и маску отступов
            if src_prepared is None:
                src_prepared = src
            else:
                src_prepared = torch.cat([src_prepared, src], dim=1, out=None)
            pos_prepared = pos[-1]
            assert mask is not None
        ################################################################################################################

        src_prepared = self.input_proj(src_prepared)  # прогон данных через 1x1 conv
        hs_full = self.transformer(src_prepared, mask, self.query_embed.weight, pos_prepared)
        hs = hs_full[0]
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        # if True:
        if random.randrange(30) == 0:
            img = split_samples[0].tensors[0, :, :, :].permute(1, 2, 0).cpu().numpy()
            arr_min = img.min()
            arr_max = img.max()
            img = (img - arr_min) / (arr_max - arr_min)
            s = out["pred_boxes"].size()[1]
            wh = img.shape
            plt.imsave('test-output.png', img, format='png')
            img = cv2.imread('test-output.png')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            counter = 0
            for i in range(min(s, 500)):
                x = int(wh[1] * float(out["pred_boxes"][0, i, 0].cpu().detach().numpy()))
                y = int(wh[0] * float(out["pred_boxes"][0, i, 1].cpu().detach().numpy()))
                w = int(wh[1] * float(out["pred_boxes"][0, i, 2].cpu().detach().numpy()))
                h = int(wh[0] * float(out["pred_boxes"][0, i, 3].cpu().detach().numpy()))
                max_index = torch.argmax(out['pred_logits'][0][i])
                color = (128, 128, 128)
                if max_index == 1:
                    color = (255, 0, 0)
                if max_index == len(out['pred_logits'][0][i]) - 1:
                    color = (0, 255, 0)
                wth = 3 if max_index == 1 else 0
                if max_index == 1:
                    counter = counter + 1
                img = cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), color, wth)
            unix_time_int = int(time.time())
            plt.imsave('my/images/' + str(counter) + '_' + str(unix_time_int) + '.png', img, format='png')

        return out


def generate_temporal_encoding(t, dim, W, H):
    """Генерация временных кодировок."""
    encoding = torch.zeros(dim, W, H)
    for i in range(dim // 2):
        encoding[2 * i] = torch.sin(torch.tensor(t / (10000 ** (2 * i / dim)))).expand(W, H)
        encoding[2 * i + 1] = torch.cos(torch.tensor(t / (10000 ** (2 * i / dim)))).expand(W, H)
    return encoding.unsqueeze(0)  # Добавляем размерность батча


class SetCriterion(nn.Module):
    """
        Этот класс вычисляет потерю для DETR. Процесс происходит в два этапа:
        1) мы вычисляем венгерское назначение между блоками истинности и выходами модели
        2) мы контролируем каждую пару сопоставленных истинности и прогноза (класс контроля и блок)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    num_classes = 2  # в теории он 1 (дым), но надо подавать на 1 больше
    device = torch.device(args.device)
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    model = DETR(backbone, transformer, num_classes=num_classes, num_queries=args.num_queries, args=args)
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef, 'loss_giou': args.giou_loss_coef}
    losses = ['labels', 'boxes', 'cardinality']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=args.eos_coef,
                             losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    return model, criterion, postprocessors
