import math

import PIL
import torchvision.transforms as T
import random
import torch
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode
from util.box_ops import box_xyxy_to_cxcywh
from torchvision.transforms.functional import adjust_brightness, adjust_contrast
from util.misc import interpolate
from typing import Tuple


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


# ГОРИЗОНТАЛЬНЫЙ ФЛИП
def hflip(image: torch.Tensor, target: dict) -> tuple:
    flipped_image = torch.flip(image, [-1])  # Горизонтальный флип тензора
    w = image.shape[-1]
    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"].clone()
        boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        target["boxes"] = boxes
    return flipped_image, target


# ВЕРТИКАЛЬНЫЙ ФЛИП
def vflip(image: torch.Tensor, target: dict) -> tuple:
    flipped_image = torch.flip(image, [-2])  # Вертикальный флип тензора
    h = image.shape[-2]  # Высота изображения
    target = target.copy()

    if "boxes" in target:
        boxes = target["boxes"].clone()
        boxes[:, [1, 3]] = h - boxes[:, [3, 1]]  # y_min и y_max заменяются на h - y_max и h - y_min
        target["boxes"] = boxes

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


def calculate_visibility(boxes, mask):
    """
    boxes: [N, 4] в формате (x_min, y_min, x_max, y_max)
    mask: [H, W] бинарная маска
    Возвращает: visibility_ratio [N]
    """
    visibility_ratios = []
    for box in boxes:
        x1, y1, x2, y2 = box.int().tolist()
        # Вырезаем область бокса из маски
        box_mask = mask[y1:y2, x1:x2]
        if box_mask.numel() == 0:
            visibility = 0.0
        else:
            visible_pixels = box_mask.sum().float()
            total_pixels = (x2 - x1) * (y2 - y1)
            visibility = (visible_pixels / total_pixels).item()
        visibility_ratios.append(visibility)
    return torch.tensor(visibility_ratios)


def generate_grid_mask(image_size, d, ratio, rotate_angle):
    h, w = image_size
    mask = torch.ones((h, w), dtype=torch.bool)

    # Повернутая сетка
    if rotate_angle != 0:
        d = int(d / abs(math.cos(math.radians(rotate_angle))))

    for x in range(0, w, d):
        for y in range(0, h, d):
            if random.random() < ratio:
                yy = min(y + d, h)
                xx = min(x + d, w)
                mask[y:yy, x:xx] = 0

    if rotate_angle != 0:
        mask = F.rotate(mask.unsqueeze(0).float(), rotate_angle).squeeze(0) > 0.5
    return mask


class RandomBrightnessContrast(object):
    def __init__(self,
                 brightness_range: Tuple[float, float] = (0.8, 1.2),
                 contrast_range: Tuple[float, float] = (0.8, 1.2),
                 p: float = 0.5,
                 pixel_range: Tuple[float, float] = (0.0, 255.0)):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.p = p
        self.pixel_min, self.pixel_max = pixel_range

    def __call__(self, image: torch.Tensor, target: dict) -> Tuple[torch.Tensor, dict]:
        if random.random() >= self.p:
            return image, target

        # Применяем преобразования
        if random.random() > 0.5:
            image = self.adjust_brightness(image, self.get_factor(*self.brightness_range))
            image = self.adjust_contrast(image, self.get_factor(*self.contrast_range))
        else:
            image = self.adjust_contrast(image, self.get_factor(*self.contrast_range))
            image = self.adjust_brightness(image, self.get_factor(*self.brightness_range))

        # Клиппинг значений
        image = torch.clamp(image, self.pixel_min, self.pixel_max)
        return image, target

    @staticmethod
    def get_factor(min_val: float, max_val: float) -> float:
        return random.uniform(min_val, max_val)

    @staticmethod
    def adjust_brightness(image: torch.Tensor, factor: float) -> torch.Tensor:
        """Регулировка яркости для любого числа каналов"""
        delta = (factor - 1) * (image.max() - image.min())
        return image + delta

    @staticmethod
    def adjust_contrast(image: torch.Tensor, factor: float) -> torch.Tensor:
        """Регулировка контраста для любого числа каналов"""
        mean = image.mean(dim=(-2, -1), keepdim=True)
        return (image - mean) * factor + mean


class RandomShift(object):

    def __init__(self, shift_x_range=(-0.1, 0.1), shift_y_range=(-0.1, 0.1), fill=0):
        """
        Args:
            shift_x_range (tuple): Диапазон сдвига по оси X в долях от ширины (например, (-0.1, 0.1) для ±10%)
            shift_y_range (tuple): Диапазон сдвига по оси Y в долях от высоты
            fill (int): Значение для заполнения новых пикселей (по умолчанию 0)
        """
        self.shift_x_range = shift_x_range
        self.shift_y_range = shift_y_range
        self.fill = fill

    def __call__(self, image: torch.Tensor, target: dict) -> tuple:
        _, h, w = image.shape

        # Генерация сдвига в пикселях
        dx = int(random.uniform(*self.shift_x_range) * w)
        dy = int(random.uniform(*self.shift_y_range) * h)

        # Применение аффинного сдвига к изображению
        shifted_image = F.affine(image, angle=0, translate=(dx, dy), scale=1.0, shear=0, fill=self.fill)

        if target is not None:
            target = target.copy()

            # Обработка bounding boxes
            if "boxes" in target:
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] += dx  # Сдвиг X-координат
                boxes[:, [1, 3]] += dy  # Сдвиг Y-координат

                # Обрезка боксов до границ изображения
                boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=w)
                boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=h)

                # Убедимся, что x_max >= x_min и y_max >= y_min
                boxes[:, 2] = torch.max(boxes[:, 0], boxes[:, 2])
                boxes[:, 3] = torch.max(boxes[:, 1], boxes[:, 3])

                # Вычисление площади и фильтрация боксов с нулевой площадью
                area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                keep = area > 1e-6  # Минимальная площадь для сохранения бокса

                # Обновление target
                target["boxes"] = boxes[keep]
                target["area"] = area[keep]

                # Фильтрация других полей (labels, iscrowd)
                for field in ["labels", "iscrowd"]:
                    if field in target:
                        target[field] = target[field][keep]

            # Обработка масок
            if "masks" in target:
                masks = target["masks"].float()
                shifted_masks = F.affine(masks, angle=0, translate=(dx, dy), scale=1.0, shear=0, fill=0)
                target["masks"] = shifted_masks.bool()

                # Фильтрация масок в соответствии с боксами
                if "boxes" in target:
                    target["masks"] = target["masks"][keep]

        return shifted_image, target


class RandomRotation(object):
    def __init__(self, degrees=(-10, 10), p=0.5, expand=False):
        self.degrees = degrees
        self.p = p
        self.expand = expand

    def __call__(self, img: torch.Tensor, target: dict) -> tuple:
        if random.random() > self.p:
            return img, target

        angle = random.uniform(*self.degrees)
        _, h, w = img.shape
        device = img.device

        # Поворот изображения
        rotated_img = F.rotate(img, angle, interpolation=InterpolationMode.BILINEAR, expand=self.expand)

        # Расчет новых размеров
        if self.expand:
            new_h, new_w = rotated_img.shape[-2:]
        else:
            new_h, new_w = h, w

        # Центр исходного изображения
        orig_cx = (w - 1) / 2.0
        orig_cy = (h - 1) / 2.0

        # Центр нового изображения
        new_cx = (new_w - 1) / 2.0 if self.expand else orig_cx
        new_cy = (new_h - 1) / 2.0 if self.expand else orig_cy

        # Матрица преобразования
        theta = torch.deg2rad(torch.tensor(angle, device=device))
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        reverse_matrix = torch.tensor([
            [cos_theta, sin_theta, orig_cx - new_cx * cos_theta - new_cy * sin_theta],
            [-sin_theta, cos_theta, orig_cy + new_cx * sin_theta - new_cy * cos_theta]
        ], device=device)

        if "boxes" in target:
            boxes = target["boxes"].clone()
            num_boxes = boxes.shape[0]

            # Генерация углов боксов
            corners = torch.zeros((num_boxes, 4, 2), device=device)
            corners[:, 0, :] = boxes[:, [0, 1]]  # Левый верхний
            corners[:, 1, :] = boxes[:, [2, 1]]  # Правый верхний
            corners[:, 2, :] = boxes[:, [2, 3]]  # Правый нижний
            corners[:, 3, :] = boxes[:, [0, 3]]  # Левый нижний

            # Добавление однородных координат
            ones = torch.ones((num_boxes, 4, 1), device=device)
            corners_homo = torch.cat([corners, ones], dim=-1)  # [N,4,3]

            # Исправленная операция einsum
            transformed = torch.einsum('ab,ncb->nca', reverse_matrix, corners_homo)

            # Обрезка координат
            transformed[:, :, 0] = transformed[:, :, 0].clamp(min=0, max=new_w - 1)
            transformed[:, :, 1] = transformed[:, :, 1].clamp(min=0, max=new_h - 1)

            # Расчет новых боксов
            x_min, _ = transformed[:, :, 0].min(dim=1)
            y_min, _ = transformed[:, :, 1].min(dim=1)
            x_max, _ = transformed[:, :, 0].max(dim=1)
            y_max, _ = transformed[:, :, 1].max(dim=1)

            new_boxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)

            # Фильтрация невалидных боксов
            valid = (new_boxes[:, 2] > new_boxes[:, 0] + 1) & (new_boxes[:, 3] > new_boxes[:, 1] + 1)

            # Обновление целевых данных
            target["boxes"] = new_boxes[valid]
            for key in ["labels", "area", "iscrowd"]:
                if key in target:
                    target[key] = target[key][valid]

            if "area" in target:
                target["area"] = (new_boxes[valid, 2] - new_boxes[valid, 0]) * (
                            new_boxes[valid, 3] - new_boxes[valid, 1])

        return rotated_img, target


class GridMaskWithBoxProcessing:
    def __init__(self, ratio=0.5, d_range=(0.3, 0.6), rotate=(-45, 45),
                 visibility_threshold=0.2, p=0.5):
        self.ratio = ratio
        self.d_range = d_range
        self.rotate = rotate
        self.visibility_threshold = visibility_threshold
        self.p = p

    def __call__(self, img, target):
        if random.random() > self.p:
            return img, target

        # Генерация параметров маски
        d = random.uniform(*self.d_range) * min(img.shape[1], img.shape[2])
        angle = random.uniform(*self.rotate)
        mask = generate_grid_mask(img.shape[1:], d, self.ratio, angle)

        # Применение маски к изображению
        masked_img = img.clone()
        masked_img[:, ~mask] = 0  # Зануляем замаскированные пиксели

        # Обработка боксов
        if "boxes" in target:
            visibility = calculate_visibility(target["boxes"], mask)
            keep_indices = visibility > self.visibility_threshold

            # Обновляем target
            target["boxes"] = target["boxes"][keep_indices]
            target["labels"] = target["labels"][keep_indices]
            target["area"] = target["area"][keep_indices]

            # Опционально: обновляем маски (для сегментации)
            if "masks" in target:
                target["masks"] = target["masks"][keep_indices]

        return masked_img, target


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


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return vflip(img, target)
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
