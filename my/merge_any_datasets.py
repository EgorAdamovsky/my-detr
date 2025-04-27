import json
import os
import shutil
from pathlib import Path


def copy_folder_contents(source_path, destination_path):
    """Копирует содержимое папки, предотвращая дублирование файлов"""
    os.makedirs(destination_path, exist_ok=True)

    for item in os.listdir(source_path):
        source_item = os.path.join(source_path, item)
        destination_item = os.path.join(destination_path, item)

        if os.path.isdir(source_item):
            shutil.copytree(source_item, destination_item, dirs_exist_ok=True)
        else:
            if not os.path.exists(destination_item):
                shutil.copy2(source_item, destination_item)
            else:
                print(f"Предупреждение: Файл {item} уже существует в целевой директории")


def merge_coco_datasets(ds_type: str,
                        output_path: str,
                        *dataset_paths: str,
                        max_prev_imgs: int = 10):
    """
    Объединяет произвольное количество COCO-датасетов

    Args:
        ds_type: Тип датасета ('train' или 'val')
        output_path: Путь для сохранения результата
        dataset_paths: Переменное количество путей к исходным датасетам
        max_prev_imgs: Максимальное количество элементов в prev_imgs
    """
    merged = {
        'info': {},
        'licenses': [],
        'images': [],
        'annotations': [],
        'categories': []
    }

    # Структуры для отслеживания
    category_map = {}
    image_id_map = {}
    ann_id_map = {}
    existing_files = set()
    next_cat_id = 0
    next_image_id = 0
    next_ann_id = 0

    # Обрабатываем все датасеты
    for ds_idx, ds_path in enumerate(dataset_paths):
        # Загружаем JSON
        json_path = Path(ds_path) / "annotations" / f"instances_{ds_type}2017.json"
        with open(json_path, 'r') as f:
            ds = json.load(f)

        # Обработка категорий
        for cat in ds.get('categories', []):
            key = (cat['name'], cat.get('supercategory', ''))
            if key not in category_map:
                category_map[key] = next_cat_id
                merged['categories'].append({
                    'id': next_cat_id,
                    'name': key[0],
                    'supercategory': key[1]
                })
                next_cat_id += 1

        # Обработка лицензий (уникальные)
        existing_licenses = {(lic['id'], lic['name']) for lic in merged['licenses']}
        for lic in ds.get('licenses', []):
            key = (lic['id'], lic['name'])
            if key not in existing_licenses:
                merged['licenses'].append(lic)
                existing_licenses.add(key)

        # Обработка info (берем из первого непустого)
        if not merged['info'] and ds.get('info'):
            merged['info'] = ds['info']

        # Обработка изображений
        current_images = []
        for img in ds.get('images', []):
            # Проверка дубликатов
            if img['file_name'] in existing_files:
                print(f"Предупреждение: Дубликат файла {img['file_name']}")
                continue

            new_img = {
                'id': next_image_id,
                'file_name': img['file_name'],
                'width': img['width'],
                'height': img['height'],
                'prev_imgs': img.get('prev_imgs', [])[-max_prev_imgs:]  # Ограничение до 10
            }
            current_images.append(new_img)
            image_id_map[img['id']] = next_image_id
            existing_files.add(img['file_name'])
            next_image_id += 1

        merged['images'].extend(current_images)

        # Обработка аннотаций
        current_anns = []
        for ann in ds.get('annotations', []):
            new_ann = {
                'id': next_ann_id,
                'image_id': image_id_map[ann['image_id']],
                'category_id': category_map.get(
                    (ds['categories'][ann['category_id'] - 1]['name'],
                     ds['categories'][ann['category_id'] - 1].get('supercategory', '')),
                    0
                ),
                'bbox': ann['bbox'],
                'area': ann['area'],
                'segmentation': ann.get('segmentation', []),
                'iscrowd': ann.get('iscrowd', 0)
            }
            current_anns.append(new_ann)
            ann_id_map[ann['id']] = next_ann_id
            next_ann_id += 1

        merged['annotations'].extend(current_anns)

        # Копирование изображений
        img_dir = Path(ds_path) / f"{ds_type}2017"
        if img_dir.exists():
            copy_folder_contents(str(img_dir), str(Path(output_path) / f"{ds_type}2017"))

    # Сохранение результата
    output_dir = Path(output_path + r"\annotations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем JSON
    with open(output_dir / f"instances_{ds_type}2017.json", 'w') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"Слияние завершено! Результат сохранен в {output_path}")


# Пример использования:
merge_coco_datasets(
    "train",
    r"D:\Disser\Datasets\TEST-DATASET",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-0",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-1",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-2",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-3",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-4",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-5",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-6",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-7",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-8",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-9",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-10",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-11",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-12",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-13",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-14",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-15",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-16",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-17",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-18",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-19",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-20",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-21",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-22",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-23",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-24",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-25",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-26",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-27",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-28",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-29",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-30",
    # r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-31",
    # # r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-32",
    # r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-33",
    # r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-34",
    # r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-35",
    max_prev_imgs=10
)

merge_coco_datasets(
    "val",
    r"D:\Disser\Datasets\TEST-DATASET",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-val-0",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-val-1",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-val-2",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-val-3",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-val-4",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-val-5",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-val-6",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-val-7",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-val-8",
    max_prev_imgs=10
)

# merge_coco_datasets(
#     "val",
#     r"D:\Disser\Datasets\TEST-DATASET",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-test-0",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-test-1",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-test-2",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-test-3",
#     max_prev_imgs=10
# )

print("Слияние завершено!")
