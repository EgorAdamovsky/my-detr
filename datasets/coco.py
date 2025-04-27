import random
from pathlib import Path  # Импорт модуля для работы с путями файловой системы
import torch  # Импорт PyTorch и его компонентов для работы с данными
import torch.utils.data  # Импорт torchvision для работы с датасетами и преобразованиями изображений
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from pycocotools import mask as coco_mask  # Импорт инструментов для работы с масками COCO
from torchvision.transforms.functional import pil_to_tensor

import datasets.my_transforms as T  # Импорт пользовательских преобразований данных из модуля datasets


# Класс-наследник стандартного COCO датасета из torchvision
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file)  # Вызов конструктора родительского класса
        self._transforms = transforms  # Сохранение преобразований
        self.prepare = ConvertCocoPolysToMask(return_masks)  # Инициализация конвертера полигонов в маски

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(
            idx)  # Получение изображения и аннотаций из родительского класса
        image_id = self.ids[idx]  # Получение ID изображения

        ## ЭКСПЕРИМЕНТ 3 ###############################################################################################
        ## читаю доп. кадры ############################################################################################
        list_imgs = self.coco.imgs[idx]['prev_imgs']
        prevs_list = [img]
        prevs = pil_to_tensor(img).type('torch.FloatTensor')
        if len(list_imgs) < 2:
            for i in range(2):
                prevs = torch.cat([prevs, pil_to_tensor(img).type('torch.FloatTensor')], dim=0, out=None)
        else:
            for i in range(2):
                temp = Image.open(str(self.root) + "\\" + list_imgs[-i])  # с конца
                prevs = torch.cat([prevs, pil_to_tensor(temp).type('torch.FloatTensor')], dim=0, out=None)
                prevs_list.append(temp)
                # temp.save("my/temps/" + str(i + 1) + ".png")
        ## ЭКСПЕРИМЕНТ 3 ###############################################################################################

        target = {'image_id': image_id, 'annotations': target}  # Формирование структуры target с image_id и аннотациями
        img, target = self.prepare(img, target)  # Подготовка данных, картинку сюда подавать не имеет смысла
        # img.save("my/temps/0.png")

        if self._transforms is not None:  # Применение дополнительных преобразований, если заданы
            prevs, target = self._transforms(prevs, target)
            split_prevs = torch.tensor_split(prevs, 3, dim=0)
            for i in range(3):
                _img = split_prevs[i].permute(1, 2, 0).cpu().numpy()
                arr_min = _img.min()
                arr_max = _img.max()
                _img = (_img - arr_min) / (arr_max - arr_min)
                plt.imsave('my/temps/' + str(i) + '_.png', _img, format='png')

        ## ЭКСПЕРИМЕНТ 4 ###############################################################################################
        ## нормирую сам ################################################################################################
        # tensor_min = prevs.min()
        # tensor_max = prevs.max()
        # prevs = (prevs - tensor_min) / (tensor_max - tensor_min)
        # mean = prevs.mean()
        # std = prevs.std()
        # prevs = (prevs - mean) / std

        # target['boxes'][0][0] = target['boxes'][0][0] / target['size'][1]
        # target['boxes'][0][1] = target['boxes'][0][1] / target['size'][0]
        # target['boxes'][0][2] = target['boxes'][0][2] / target['size'][1]
        # target['boxes'][0][3] = target['boxes'][0][3] / target['size'][0]

        # return img, target
        return prevs, target
        ## ЭКСПЕРИМЕНТ 4 ###############################################################################################


# Класс для конвертации полигонов COCO в маски
class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        # Получение размеров изображения
        w, h = image.size
        # Извлечение image_id из target и конвертация в тензор
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        # Фильтрация аннотаций: убираем объекты с iscrowd=1
        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        # Обработка bounding boxes
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # Конвертация формата [x,y,w,h] -> [x1,y1,x2,y2]
        boxes[:, 2:] += boxes[:, :2]
        # Ограничение координат границами изображения
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # Обработка меток классов
        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        # Фильтрация некорректных боксов (ширина/высота <=0)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        # Формирование итогового target
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # Добавление дополнительных полей для совместимости с COCO API
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        # Сохранение оригинального и текущего размера изображения
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


# Функция создания преобразований для датасета
def make_coco_transforms(image_set):
    # Нормализация с параметрами ImageNet
    normalize = T.Compose([T.ToTensor(),
                           T.Normalize(
                               [0.485, 0.456, 0.406, 0.485, 0.456, 0.406, 0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.229, 0.224, 0.225])])
    # Множество масштабов для ресайза
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            # Горизонтальный флип с вероятностью 0.5
            T.RandomHorizontalFlip(),
            # Выбор между простым ресайзом и ресайзом с кропом
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        # Для валидации только ресайз до 800 и нормализация
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


# Функция создания датасета
def build(image_set, args):
    # Проверка существования пути к данным
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    # Определение путей для трейна и валидации
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]  # Получение путей для текущего набора данных
    # Создание датасета с заданными параметрами
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=False)
    return dataset
