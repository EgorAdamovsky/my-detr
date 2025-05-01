"""
ПРОВЕРКА КОРРЕКТНОСТИ ДАТАСЕТА
"""

import os
import shutil

import cv2
import numpy as np
from pycocotools.coco import COCO

PATH = r"D:\Disser\Datasets\TEST-DATASET"
# PATH = r"D:\Disser\Datasets\temps\dataset-train-9"
# PATH = r"D:\Disser\Datasets\temps\dataset-val-2"

DATASET_CONFIG = {
    "train": {
        "annotations": PATH + r"/annotations/instances_train2017.json",
        "images": PATH + r"/train2017",
        "output": PATH + r"/visualizations/train2017"
    },
    "val": {
        "annotations": PATH + r"/annotations/instances_val2017.json",
        "images": PATH + r"/val2017",
        "output": PATH + r"/visualizations/val2017"
    }
}
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]


def visualize_annotations(data_type):
    if os.path.exists(PATH + r'\visualization'):
        shutil.rmtree(PATH + r'\visualization')

    config = DATASET_CONFIG[data_type]

    # Проверка наличия файлов
    if not os.path.exists(config["annotations"]):
        print(f"[Ошибка] Аннотации для {data_type} не найдены: {config['annotations']}")
        return
    if not os.path.exists(config["images"]):
        print(f"[Ошибка] Изображения для {data_type} не найдены: {config['images']}")
        return

    # Загрузка аннотаций
    coco = COCO(config["annotations"])
    os.makedirs(config["output"], exist_ok=True)

    # Получение всех изображений
    image_ids = coco.getImgIds()

    for img_id in image_ids:
        # Загрузка изображения
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(config["images"], img_info['file_name'])

        if not os.path.exists(img_path):
            print(f"[Предупреждение] Изображение не найдено: {img_path}")
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"[Ошибка] Не удалось загрузить изображение: {img_path}")
            continue

        # Получение аннотаций
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)

        # Визуализация аннотаций
        for ann in annotations:
            category = coco.loadCats(ann['category_id'])[0]['name']
            color_idx = ann['category_id'] % len(COLORS)
            color = COLORS[color_idx]

            # Рисование bounding box
            if 'bbox' in ann:
                x, y, w, h = map(int, ann['bbox'])
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
                cv2.putText(image, category, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Рисование сегментации
            if 'segmentation' in ann and isinstance(ann['segmentation'], list):
                for seg in ann['segmentation']:
                    pts = np.array([[int(x), int(y)] for x, y in zip(seg[::2], seg[1::2])],
                                   dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(image, [pts], True, (0, 255, 0), 2)

        # Сохранение результата
        output_path = os.path.join(config["output"], f"annotated_{img_info['file_name']}")
        cv2.imwrite(output_path, image)
        print(f"[Сохранено] {output_path}")


if __name__ == "__main__":
    for data_type in DATASET_CONFIG:
        print(f"\n[Обработка] Начата визуализация для: {data_type.upper()}")
        visualize_annotations(data_type)

print("\n[Успех]")
