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
    Все изображения копируются в выходную папку, но в JSON добавляются только те,
    у которых есть аннотации после фильтрации.
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

    # 1. Сначала копируем все изображения из всех датасетов в выходную папку
    output_img_dir = Path(output_path) / f"{ds_type}2017"
    for ds_path in dataset_paths:
        img_dir = Path(ds_path) / f"{ds_type}2017"
        if img_dir.exists():
            copy_folder_contents(str(img_dir), str(output_img_dir))

    # 2. Обработка всех датасетов
    for ds_idx, ds_path in enumerate(dataset_paths):
        # Загружаем JSON
        json_path = Path(ds_path) / "annotations" / f"instances_{ds_type}2017.json"
        with open(json_path, 'r') as f:
            ds = json.load(f)

        # 2.1 Обработка категорий
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

        # 2.2 Обработка лицензий
        existing_licenses = {(lic['id'], lic['name']) for lic in merged['licenses']}
        for lic in ds.get('licenses', []):
            key = (lic['id'], lic['name'])
            if key not in existing_licenses:
                merged['licenses'].append(lic)
                existing_licenses.add(key)

        # 2.3 Обработка info
        if not merged['info'] and ds.get('info'):
            merged['info'] = ds['info']

        # 3. Сборка ID изображений, прошедших фильтр по шагу
        relevant_image_ids = set()
        for idx, ann in enumerate(ds.get('annotations', [])):
            if idx % annotation_step == 0:  # Шаг по индексу аннотации
                relevant_image_ids.add(ann['image_id'])

        # 4. Формирование списка всех аннотаций для выбранных изображений
        filtered_anns = [ann for ann in ds.get('annotations', [])
                         if ann['image_id'] in relevant_image_ids]

        # 5. Обработка изображений (только те, что в relevant_image_ids)
        current_images = []
        added_files = set()  # для проверки дубликатов в текущем датасете
        for img in ds.get('images', []):
            if img['id'] not in relevant_image_ids:
                continue  # пропускаем изображение без аннотаций

            if img['file_name'] in existing_files or img['file_name'] in added_files:
                print(f"Предупреждение: Дубликат файла {img['file_name']}")
                continue

            new_img = {
                'id': next_image_id,
                'file_name': img['file_name'],
                'width': img['width'],
                'height': img['height'],
                'prev_imgs': img.get('prev_imgs', [])[-max_prev_imgs:]
            }
            current_images.append(new_img)
            image_id_map[img['id']] = next_image_id
            existing_files.add(img['file_name'])
            added_files.add(img['file_name'])
            next_image_id += 1

        merged['images'].extend(current_images)
        print(f"[INFO] Из датасета {ds_path} добавлено {len(current_images)} изображений")

        # 6. Обработка аннотаций
        current_anns = []
        for ann in filtered_anns:
            # Проверяем, что изображение уже добавлено
            if ann['image_id'] not in image_id_map:
                print(f"[Предупреждение] Аннотация {ann['id']} ссылается на неизвестное изображение")
                continue

            # Получаем информацию о категории из исходного датасета
            original_category = None
            for cat in ds.get('categories', []):
                if cat['id'] == ann['category_id']:
                    original_category = cat
                    break
            if not original_category:
                print(f"[Ошибка] Категория ID {ann['category_id']} не найдена для аннотации {ann['id']}")
                continue

            key = (original_category['name'], original_category.get('supercategory', ''))
            mapped_cat_id = category_map.get(key, 0)
            if mapped_cat_id == 0:
                print(f"[Ошибка] Категория {key} не найдена в category_map")
                continue

            new_ann = {
                'id': next_ann_id,
                'image_id': image_id_map[ann['image_id']],
                'category_id': mapped_cat_id,
                'bbox': ann['bbox'],
                'area': ann['area'],
                'segmentation': ann.get('segmentation', []),
                'iscrowd': ann.get('iscrowd', 0)
            }
            current_anns.append(new_ann)
            ann_id_map[ann['id']] = next_ann_id
            next_ann_id += 1

        merged['annotations'].extend(current_anns)
        print(f"[INFO] Из датасета {ds_path} добавлено {len(current_anns)} аннотаций")

    # 7. Сохранение результата
    output_dir = Path(output_path) / "annotations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем JSON
    output_json_path = output_dir / f"instances_{ds_type}2017.json"
    with open(output_json_path, 'w') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

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


my_annotation_step_train = 25  # 25, 15, 5
my_annotation_step_test = 75  # 75, 50, 30
my_max_prev_imgs = 5
my_path = r"D:\Disser\Datasets\TEST-DATASET"

if os.path.exists(my_path):
    shutil.rmtree(my_path)

merge_coco_datasets(
    "train",
    my_path,
    r"D:\Disser\Datasets\temps\dataset-train-1",
    r"D:\Disser\Datasets\temps\dataset-train-2",
    r"D:\Disser\Datasets\temps\dataset-train-3",
    r"D:\Disser\Datasets\temps\dataset-train-4",
    r"D:\Disser\Datasets\temps\dataset-train-5",
    r"D:\Disser\Datasets\temps\dataset-train-6",
    r"D:\Disser\Datasets\temps\dataset-train-7",
    r"D:\Disser\Datasets\temps\dataset-train-8",
    r"D:\Disser\Datasets\temps\dataset-train-9",
    r"D:\Disser\Datasets\temps\dataset-train-10",
    r"D:\Disser\Datasets\temps\dataset-train-12",
    r"D:\Disser\Datasets\temps\dataset-train-14",
    r"D:\Disser\Datasets\temps\dataset-train-15",
    r"D:\Disser\Datasets\temps\dataset-train-16",
    r"D:\Disser\Datasets\temps\dataset-train-17",
    r"D:\Disser\Datasets\temps\dataset-train-18",
    r"D:\Disser\Datasets\temps\dataset-train-20",
    r"D:\Disser\Datasets\temps\dataset-train-21",
    r"D:\Disser\Datasets\temps\dataset-train-22",
    r"D:\Disser\Datasets\temps\dataset-train-23",
    r"D:\Disser\Datasets\temps\dataset-train-24",
    r"D:\Disser\Datasets\temps\dataset-train-25",
    r"D:\Disser\Datasets\temps\dataset-train-26",
    r"D:\Disser\Datasets\temps\dataset-train-27",
    r"D:\Disser\Datasets\temps\dataset-train-28",
    r"D:\Disser\Datasets\temps\dataset-train-29",
    r"D:\Disser\Datasets\temps\dataset-train-30",
    r"D:\Disser\Datasets\temps\dataset-train-31",
    r"D:\Disser\Datasets\temps\dataset-train-32",
    r"D:\Disser\Datasets\temps\dataset-train-33",
    r"D:\Disser\Datasets\temps\dataset-train-34",
    r"D:\Disser\Datasets\temps\dataset-train-35",
    r"D:\Disser\Datasets\temps\dataset-train-36",
    r"D:\Disser\Datasets\temps\dataset-train-37",
    r"D:\Disser\Datasets\temps\dataset-train-38",
    r"D:\Disser\Datasets\temps\dataset-train-39",
    r"D:\Disser\Datasets\temps\dataset-train-40",
    r"D:\Disser\Datasets\temps\dataset-train-41",
    r"D:\Disser\Datasets\temps\dataset-train-42",
    r"D:\Disser\Datasets\temps\dataset-train-43",
    r"D:\Disser\Datasets\temps\dataset-train-44",
    r"D:\Disser\Datasets\temps\dataset-train-45",
    r"D:\Disser\Datasets\temps\dataset-train-46",
    r"D:\Disser\Datasets\temps\dataset-train-47",
    r"D:\Disser\Datasets\temps\dataset-train-48",
    r"D:\Disser\Datasets\temps\dataset-train-49",
    r"D:\Disser\Datasets\temps\dataset-train-50",
    r"D:\Disser\Datasets\temps\dataset-train-51",
    r"D:\Disser\Datasets\temps\dataset-train-52",
    r"D:\Disser\Datasets\temps\dataset-train-53",
    r"D:\Disser\Datasets\temps\dataset-train-54",
    r"D:\Disser\Datasets\temps\dataset-train-55",
    r"D:\Disser\Datasets\temps\dataset-train-56",
    r"D:\Disser\Datasets\temps\dataset-train-57",
    r"D:\Disser\Datasets\temps\dataset-train-58",
    r"D:\Disser\Datasets\temps\dataset-train-59",
    r"D:\Disser\Datasets\temps\dataset-train-60",
    r"D:\Disser\Datasets\temps\dataset-train-61",
    r"D:\Disser\Datasets\temps\dataset-train-62",
    r"D:\Disser\Datasets\temps\dataset-train-63",
    r"D:\Disser\Datasets\temps\dataset-train-64",
    r"D:\Disser\Datasets\temps\dataset-train-65",
    r"D:\Disser\Datasets\temps\dataset-train-66",
    r"D:\Disser\Datasets\temps\dataset-train-67",
    r"D:\Disser\Datasets\temps\dataset-train-68",
    r"D:\Disser\Datasets\temps\dataset-train-69",
    r"D:\Disser\Datasets\temps\dataset-train-70",
    r"D:\Disser\Datasets\temps\dataset-train-71",
    r"D:\Disser\Datasets\temps\dataset-train-72",
    r"D:\Disser\Datasets\temps\dataset-train-73",
    r"D:\Disser\Datasets\temps\dataset-train-74",
    r"D:\Disser\Datasets\temps\dataset-train-75",
    r"D:\Disser\Datasets\temps\dataset-train-76",
    r"D:\Disser\Datasets\temps\dataset-train-77",
    r"D:\Disser\Datasets\temps\dataset-train-78",
    r"D:\Disser\Datasets\temps\dataset-train-79",
    r"D:\Disser\Datasets\temps\dataset-train-80",
    r"D:\Disser\Datasets\temps\dataset-train-81",
    r"D:\Disser\Datasets\temps\dataset-train-82",
    r"D:\Disser\Datasets\temps\dataset-train-83",
    r"D:\Disser\Datasets\temps\dataset-train-84",
    r"D:\Disser\Datasets\temps\dataset-train-85",
    r"D:\Disser\Datasets\temps\dataset-train-86",
    r"D:\Disser\Datasets\temps\dataset-train-87",
    r"D:\Disser\Datasets\temps\dataset-train-88",
    r"D:\Disser\Datasets\temps\dataset-train-89",
    r"D:\Disser\Datasets\temps\dataset-train-90",
    r"D:\Disser\Datasets\temps\dataset-train-91",
    r"D:\Disser\Datasets\temps\dataset-train-92",
    r"D:\Disser\Datasets\temps\dataset-train-93",
    r"D:\Disser\Datasets\temps\dataset-train-94",
    r"D:\Disser\Datasets\temps\dataset-train-95",
    r"D:\Disser\Datasets\temps\dataset-train-96",
    r"D:\Disser\Datasets\temps\dataset-train-97",
    r"D:\Disser\Datasets\temps\dataset-train-98",
    r"D:\Disser\Datasets\temps\dataset-train-99",
    r"D:\Disser\Datasets\temps\dataset-train-100",
    annotation_step=my_annotation_step_train,
    max_prev_imgs=my_max_prev_imgs
)

pass

merge_coco_datasets(
    "val",
    my_path,
    r"D:\Disser\Datasets\temps\dataset-val-0",
    r"D:\Disser\Datasets\temps\dataset-val-1",
    r"D:\Disser\Datasets\temps\dataset-val-2",
    r"D:\Disser\Datasets\temps\dataset-val-3",
    r"D:\Disser\Datasets\temps\dataset-val-4",
    r"D:\Disser\Datasets\temps\dataset-val-5",
    r"D:\Disser\Datasets\temps\dataset-val-6",
    r"D:\Disser\Datasets\temps\dataset-val-7",
    r"D:\Disser\Datasets\temps\dataset-val-8",
    r"D:\Disser\Datasets\temps\dataset-val-9",
    annotation_step=my_annotation_step_test,
    max_prev_imgs=my_max_prev_imgs
)

pass

print("[Успех]")
