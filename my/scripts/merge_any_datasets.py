"""
СБОРКА ДАТАСЕТА ПО КУСКАМ
"""

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


def merge_coco_datasets(ds_type: str, output_path: str, *dataset_paths: str,
                        annotation_step: int = 1, max_prev_imgs: int = 10):
    """
    Сливает несколько COCO-датасетов в один, выбирая аннотации с заданным шагом.

    :param ds_type: тип датасета (train/val)
    :param output_path: путь для сохранения объединенного датасета
    :param dataset_paths: пути к исходным датасетам
    :param annotation_step: шаг выборки аннотаций (например, 5 — каждая пятая)
    :param max_prev_imgs: максимальное количество предыдущих кадров в prev_imgs
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
    next_cat_id = 1
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

        # Обработка аннотаций (с шагом)
        current_anns = []
        skipped_anns = 0
        for idx, ann in enumerate(ds.get('annotations', [])):
            # Пропускаем аннотации, не попадающие под шаг
            if idx % annotation_step != 0:
                skipped_anns += 1
                continue

            # Проверяем, что изображение уже добавлено
            if ann['image_id'] not in image_id_map:
                print(f"[Предупреждение] Аннотация {ann['id']} ссылается на неизвестное изображение")
                continue

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
        print(f"[INFO] Из датасета {ds_path} добавлено {len(current_anns)} аннотаций, пропущено {skipped_anns}")

        # Копирование изображений
        img_dir = Path(ds_path) / f"{ds_type}2017"
        if img_dir.exists():
            copy_folder_contents(str(img_dir), str(Path(output_path) / f"{ds_type}2017"))

    # Сохранение результата
    output_dir = Path(output_path) / "annotations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем JSON
    output_json_path = output_dir / f"instances_{ds_type}2017.json"
    with open(output_json_path, 'w') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    replace_in_file(output_json_path, '"category_id": 0', '"category_id": 1')

    print(f"Слияние завершено! Результат сохранен в {output_path}")


def replace_in_file(file_path, old_substring, new_substring):
    """
    Заменяет все вхождения подстроки в указанном файле.

    :param file_path: Путь к файлу
    :param old_substring: Подстрока, которую нужно заменить
    :param new_substring: Подстрока, на которую нужно заменить
    """
    try:
        # Чтение файла
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Замена подстроки
        updated_content = content.replace(old_substring, new_substring)

        # Перезапись файла
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(updated_content)

        print(f"[OK] Замена выполнена в файле: {file_path}")

    except FileNotFoundError:
        print(f"[Ошибка] Файл не найден: {file_path}")
    except Exception as e:
        print(f"[Ошибка] При обработке файла: {e}")


my_annotation_step_train = 10
my_annotation_step_test = 20
my_max_prev_imgs = 5

if os.path.exists(r"D:\Disser\Datasets\TEST-DATASET"):
    shutil.rmtree(r"D:\Disser\Datasets\TEST-DATASET")

merge_coco_datasets(
    "train",
    r"D:\Disser\Datasets\TEST-DATASET",
    r"D:\Disser\Datasets\temps\dataset-train-0",
    r"D:\Disser\Datasets\temps\dataset-train-1",
    r"D:\Disser\Datasets\temps\dataset-train-2",
    r"D:\Disser\Datasets\temps\dataset-train-3",
    r"D:\Disser\Datasets\temps\dataset-train-4",
    r"D:\Disser\Datasets\temps\dataset-train-5",
    r"D:\Disser\Datasets\temps\dataset-train-6",
    r"D:\Disser\Datasets\temps\dataset-train-7",
    r"D:\Disser\Datasets\temps\dataset-train-8",
    r"D:\Disser\Datasets\temps\dataset-train-9",
    annotation_step=my_annotation_step_train,
    max_prev_imgs=my_max_prev_imgs
)
# merge_coco_datasets(
#     "train",
#     r"D:\Disser\Datasets\TEST-DATASET",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-0",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-1",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-2",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-3",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-4",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-5",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-6",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-7",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-8",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-9",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-10",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-11",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-12",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-13",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-14",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-15",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-16",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-17",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-18",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-19",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-20",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-21",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-22",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-23",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-24",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-25",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-26",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-27",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-28",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-29",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-30",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-31",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-32",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-33",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-34",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-35",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-36",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-37",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-38",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-39",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-40",
#     max_prev_imgs=5
# )

pass

merge_coco_datasets(
    "val",
    r"D:\Disser\Datasets\TEST-DATASET",
    r"D:\Disser\Datasets\temps\dataset-val-0",
    r"D:\Disser\Datasets\temps\dataset-val-1",
    r"D:\Disser\Datasets\temps\dataset-val-2",
    annotation_step=my_annotation_step_test,
    max_prev_imgs=5
)

pass

# merge_coco_datasets(
#     "val",
#     r"D:\Disser\Datasets\TEST-DATASET",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-test-0",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-test-1",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-test-2",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-test-3",
#     max_prev_imgs=10
# )

print("[Успех]")
