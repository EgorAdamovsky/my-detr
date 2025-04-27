import PIL
import torchvision.transforms as T
import random
import torch
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode
from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate


def crop(image: torch.Tensor, target: dict, region: tuple) -> tuple:
    i, j, h, w = region
    cropped_image = image[..., i:i + h, j:j + w]  # Тензорная обрезка

    target = target.copy()
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.tensor([j, i, j, i], dtype=torch.float32)
        cropped_boxes = torch.min(cropped_boxes.view(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1] - cropped_boxes[:, 0]).prod(dim=1)
        target["boxes"] = cropped_boxes.view(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        target['masks'] = target['masks'][..., i:i + h, j:j + w]
        fields.append("masks")

    if "boxes" in target or "masks" in target:
        keep = torch.all(cropped_boxes.view(-1, 2, 2)[:, 1] > cropped_boxes.view(-1, 2, 2)[:, 0], dim=1)
        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image: torch.Tensor, target: dict) -> tuple:
    flipped_image = torch.flip(image, [-1])  # Горизонтальный флип тензора

    w = image.shape[-1]
    target = target.copy()

    if "boxes" in target:
        boxes = target["boxes"].clone()
        boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = torch.flip(target['masks'], [-1])

    return flipped_image, target


def resize(image: torch.Tensor, target: dict, size: tuple, max_size: int = None) -> tuple:
    original_size = image.shape[-2:]  # Получаем [H, W] из тензора
    h, w = original_size

    if isinstance(size, int):
        short_edge = min(h, w)
        long_edge = max(h, w)
        if max_size and (size * long_edge / short_edge) > max_size:
            size = int(round(max_size * short_edge / long_edge))
        new_h, new_w = (size, int(w * size / h)) if h < w else (int(h * size / w), size)
    else:
        new_h, new_w = size

    resized_image = F.resize(image, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)

    if target is None:
        return resized_image, None

    ratio_h = new_h / h
    ratio_w = new_w / w

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"] * torch.tensor([ratio_w, ratio_h, ratio_w, ratio_h])
        target["boxes"] = boxes

    if "area" in target:
        target["area"] = target["area"] * ratio_w * ratio_h

    target["size"] = torch.tensor([new_h, new_w])

    if "masks" in target:
        masks = F.resize(target['masks'].float(), [new_h, new_w], interpolation=InterpolationMode.NEAREST)
        target['masks'] = masks > 0.5

    return resized_image, target


def pad(image: torch.Tensor, target: dict, padding: tuple) -> tuple:
    pad_left, pad_top, pad_right, pad_bottom = padding
    padded_image = F.pad(image, [pad_left, pad_top, pad_right, pad_bottom])

    target = target.copy()
    target["size"] = torch.tensor(padded_image.shape[-2:])

    if "masks" in target:
        target['masks'] = F.pad(target['masks'], [pad_left, pad_top, pad_right, pad_bottom])

    return padded_image, target


# В классе RandomSizeCrop:
class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: torch.Tensor, target: dict):
        _, h_img, w_img = img.shape

        # Генерируем размеры обрезки
        w = random.randint(self.min_size, min(w_img, self.max_size))
        h = random.randint(self.min_size, min(h_img, self.max_size))

        # Генерируем координаты обрезки вручную
        i = random.randint(0, h_img - h)
        j = random.randint(0, w_img - w)

        region = (i, j, h, w)
        return crop(img, target, region)


# В классе RandomCrop:
class RandomCrop(object):
    def __init__(self, size):
        self.size = size  # (h, w)

    def __call__(self, img: torch.Tensor, target):
        _, h_img, w_img = img.shape
        h_crop, w_crop = self.size

        # Генерируем координаты обрезки
        i = random.randint(0, h_img - h_crop)
        j = random.randint(0, w_img - w_crop)

        region = (i, j, h_crop, w_crop)
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img: torch.Tensor, target):
        _, h_img, w_img = img.shape
        crop_height, crop_width = self.size
        crop_top = int(round((h_img - crop_height) / 2.))
        crop_left = int(round((w_img - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """

    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor:
    """Теперь просто проверяет, что вход - тензор"""

    def __call__(self, img: torch.Tensor, target: dict) -> tuple:
        assert isinstance(img, torch.Tensor), "Input should be a tensor"
        return img, target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        # Проверяем, что длина mean и std соответствует количеству каналов
        if len(mean) != len(std):
            raise ValueError("mean and std must have the same length")

        self.mean = mean
        self.std = std

    def __call__(self, image: torch.Tensor, target: dict) -> tuple:
        # Проверяем количество каналов
        num_channels = image.shape[0]
        if len(self.mean) != num_channels:
            raise ValueError(f"mean has {len(self.mean)} elements but image has {num_channels} channels")

        # Нормализуем каждый канал
        normalized_image = image.clone()
        for c in range(num_channels):
            normalized_image[c] = (image[c] - self.mean[c]) / self.std[c]

        # Обновляем bounding boxes (если нужно)
        if target is not None:
            target = target.copy()
            h, w = normalized_image.shape[-2:]
            if "boxes" in target:
                boxes = target["boxes"]
                boxes = box_xyxy_to_cxcywh(boxes)
                boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
                target["boxes"] = boxes

        return normalized_image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
