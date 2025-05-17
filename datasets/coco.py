import time
from pathlib import Path  # Импорт модуля для работы с путями файловой системы

import cv2
import numpy as np
import torch  # Импорт PyTorch и его компонентов для работы с данными
import torch.utils.data  # Импорт torchvision для работы с датасетами и преобразованиями изображений
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms.functional import pil_to_tensor
import datasets.my_transforms as T  # Импорт пользовательских преобразований данных из модуля datasets


# Класс-наследник стандартного COCO датасета из torchvision
class CocoDetection(torchvision.datasets.CocoDetection):

    # КОНСТРУКТОР
    def __init__(self, img_folder, ann_file, transforms, return_masks, args):
        super(CocoDetection, self).__init__(img_folder, ann_file)  # Вызов конструктора родительского класса
        self._transforms = transforms  # Сохранение преобразований
        self.prepare = ConvertCocoPolysToMask(return_masks)  # Инициализация конвертера полигонов в маски
        self.args = args

    # ОБРАБОТКА ИЗОБРАЖЕНИЯ + АННОТАЦИИ
    def __getitem__(self, idx):

        ## ЭКСПЕРИМЕНТ #################################################################################################
        img, target = super(CocoDetection, self).__getitem__(idx)  # Получение изображения и аннотаций
        image_id = self.ids[idx]  # Получение номера изображения
        list_imgs = []
        try:
            list_imgs = self.coco.imgs[idx]['prev_imgs']  # список предыдущих кадров
        except KeyError:
            pass
        prevs = pil_to_tensor(img).type('torch.FloatTensor')  # начало формирования тензора из кадров
        if len(list_imgs) < self.args.prevs:  # если предыдущих кадров не хватает
            for i in range(self.args.prevs):  # значит просто задублировать текущий кадр
                temp = img  # забираем предыдущие кадры с конца
                prevs = torch.cat([prevs, pil_to_tensor(img).type('torch.FloatTensor')], dim=0, out=None)
                if self.args.show == 1:
                    temp.save("my/temps/" + str(i + 1) + "_orig.png")
        else:  # если предыдущих кадров хватает
            for i in range(self.args.prevs):  # значит будем с ними и работать
                temp = Image.open(str(self.root) + "\\" + list_imgs[-i])  # забираем предыдущие кадры с конца
                prevs = torch.cat([prevs, pil_to_tensor(temp).type('torch.FloatTensor')], dim=0, out=None)
                if self.args.show == 1:
                    temp.save("my/temps/" + str(i + 1) + "_orig.png")

        target = {'image_id': image_id, 'annotations': target}  # Формирование необходимой структуры
        target = self.prepare(img, target)  # Подготовка аннотаций

        if self.args.show == 1:
            img_cv = np.array(img)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
            img_cv = cv2.rectangle(img_cv,
                                   (target["boxes"][0][0].numpy().astype('int').item(),
                                    target["boxes"][0][1].numpy().astype('int').item()),
                                   (target["boxes"][0][2].numpy().astype('int').item(),
                                    target["boxes"][0][3].numpy().astype('int').item()),
                                   (255, 255, 0), 2)
            cv2.imwrite(self.args.output_dir + r"/temps/0_orig.png", img_cv)

        if self._transforms is not None:  # Применение дополнительных преобразований
            prevs, target = self._transforms(prevs, target)
            if self.args.show == 1:
                try:
                    split_prevs = torch.tensor_split(prevs, self.args.prevs + 1, dim=0)
                    for i in range(self.args.prevs + 1):
                        _img = split_prevs[i].permute(1, 2, 0).cpu().numpy()
                        arr_min = _img.min()
                        arr_max = _img.max()
                        _img = (_img - arr_min) / (arr_max - arr_min)
                        _img = (_img * 255).astype(np.uint8)
                        x = int(target["size"][1].numpy().item() * target["boxes"][0][0].numpy().item())
                        y = int(target["size"][0].numpy().item() * target["boxes"][0][1].numpy().item())
                        w = int(target["size"][1].numpy().item() * target["boxes"][0][2].numpy().item())
                        h = int(target["size"][0].numpy().item() * target["boxes"][0][3].numpy().item())
                        _img = cv2.cvtColor(_img, cv2.COLOR_RGB2BGR)
                        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
                        _img = cv2.rectangle(_img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
                                             (255, 255, 0), 2)
                        unix_time_int = int(time.time())
                        plt.imsave(self.args.output_dir + r"/temps/" + str(i) + "_trans.png", _img, format='png')
                except IndexError:
                    pass

        return prevs, target
        ## ЭКСПЕРИМЕНТ #################################################################################################


# Класс для конвертации полигонов COCO в маски
class ConvertCocoPolysToMask(object):

    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size  # Получение размеров изображения
        image_id = target["image_id"]  # Извлечение image_id из target и конвертация в тензор
        image_id = torch.tensor([image_id])
        anno = target["annotations"]  # Фильтрация аннотаций: объекты с iscrowd=1 в мусор (но у нас таких нет)
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        boxes = [obj["bbox"] for obj in anno]  # Обработка рамок в тензор
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]  # Конвертация формата [x,y,w,h] -> [x1,y1,x2,y2]
        boxes[:, 0::2].clamp_(min=0, max=w)  # Ограничение координат границами изображения
        boxes[:, 1::2].clamp_(min=0, max=h)
        classes = [obj["category_id"] for obj in anno]  # Обработка меток классов в тензор
        classes = torch.tensor(classes, dtype=torch.int64)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])  # Фильтрация плохих боксов (w или h < 0)
        boxes = boxes[keep]
        classes = classes[keep]
        target = {"boxes": boxes, "labels": classes, "image_id": image_id}  # Формирование итогового target
        area = torch.tensor([obj["area"] for obj in anno])  # Добавление доп. полей для совместимости с COCO API
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]
        target["orig_size"] = torch.as_tensor([int(h), int(w)])  # Сохранение оригинального и текущего размера
        target["size"] = torch.as_tensor([int(h), int(w)])
        return target


# Функция создания преобразований для датасета
def make_coco_transforms(image_set, rep):
    n1 = rep * [0.485, 0.456, 0.406]
    n2 = rep * [0.229, 0.224, 0.225]
    # Нормализация с параметрами ImageNet
    normalize = T.Compose([T.ToTensor(), T.Normalize(n1, n2)])
    # Множество масштабов для ресайза
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.Compose([
                T.RandomHorizontalFlip(p=0.5),  # горизонтальный флип
                T.RandomVerticalFlip(p=0.25),  # вертикальный флип
                T.RandomShift(shift_x_range=(-0.2, 0.2), shift_y_range=(-0.2, 0.2), fill=0),  # смещение
                T.RandomRotation(degrees=(-30, 30), p=0.75),
                T.RandomBrightnessContrast(brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2), p=0.75),  # цвет
                # T.GridMaskWithBoxProcessing(),
                T.RandomSelect(  # Выбор между простым ресайзом и ресайзом с кропом
                    T.RandomResize(scales, max_size=1333),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales, max_size=1333),
                    ])
                ),
            ]),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([  # Для валидации только ресайз до 800 и нормализация
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


# Функция создания датасета
def build(image_set, args):
    root = Path(args.coco_path)  # Проверка существования пути к данным
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }  # Определение путей для трейна и валидации
    img_folder, ann_file = PATHS[image_set]  # Получение путей для текущего набора данных
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set, args.prevs + 1),
                            return_masks=False,
                            args=args)
    return dataset
