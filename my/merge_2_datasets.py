import json
import os
import shutil


def copy_folder_contents(source_path, destination_path):
    """
    Копирует содержимое source_path (файлы и подпапки) в destination_path.
    """
    # Создаем целевую директорию, если её нет
    os.makedirs(destination_path, exist_ok=True)

    # Обходим содержимое исходной папки
    for item in os.listdir(source_path):
        source_item = os.path.join(source_path, item)
        destination_item = os.path.join(destination_path, item)

        if os.path.isdir(source_item):
            # Если это папка, копируем её рекурсивно
            shutil.copytree(source_item, destination_item)
        else:
            # Если это файл, копируем его
            shutil.copy2(source_item, destination_item)  # copy2 сохраняет метаданные

    print(f"Содержимое папки успешно скопировано в {destination_path}")


def merge_coco_datasets(ds_type, ds1_path, ds2_path, output_path):
    with open(ds1_path + r"\annotations\instances_" + ds_type + "2017.json", 'r') as f:
        ds1 = json.load(f)
    with open(ds2_path + r"\annotations\instances_" + ds_type + "2017.json", 'r') as f:
        ds2 = json.load(f)

    # Обработка категорий
    merged_categories = []
    category_map = {}
    next_cat_id = 1

    # Добавляем категории из первого датасета
    for cat in ds1.get('categories', []):
        key = (cat['name'], cat.get('supercategory', ''))
        category_map[key] = cat['id']
        merged_categories.append(cat)
        next_cat_id = max(next_cat_id, cat['id'] + 1)

    # Добавляем категории из второго датасета
    cat_id_mapping = {}
    for cat in ds2.get('categories', []):
        key = (cat['name'], cat.get('supercategory', ''))
        if key not in category_map:
            category_map[key] = next_cat_id
            merged_categories.append({'id': next_cat_id, 'name': cat['name'], 'supercategory': key[1]})
            next_cat_id += 1
        cat_id_mapping[cat['id']] = category_map[key]

    # Обработка изображений
    merged_images = []
    image_id_mapping = {}
    max_image_id = max([img['id'] for img in ds1.get('images', [])] or [0])

    # Добавляем изображения из первого датасета
    existing_filenames = set()
    for img in ds1.get('images', []):
        merged_images.append(img)
        existing_filenames.add(img['file_name'])

    # Добавляем изображения из второго датасета с новыми ID
    for idx, img in enumerate(ds2.get('images', []), start=1):
        new_id = max_image_id + idx
        image_id_mapping[img['id']] = new_id
        if img['file_name'] in existing_filenames:
            print(f"Предупреждение: Дублирующийся файл {img['file_name']}")
        merged_images.append({
            'id': new_id,
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height'],
            'prev_imgs': img.get('prev_imgs', [])
        })

    # Обработка аннотаций
    merged_annotations = []
    max_ann_id = max([ann['id'] for ann in ds1.get('annotations', [])] or [0])

    # Добавляем аннотации из первого датасета
    merged_annotations.extend(ds1.get('annotations', []))

    # Добавляем аннотации из второго датасета с новыми ID
    for idx, ann in enumerate(ds2.get('annotations', []), start=1):
        new_ann = {
            'id': max_ann_id + idx,
            'image_id': image_id_mapping[ann['image_id']],
            'category_id': cat_id_mapping[ann['category_id']],
            'bbox': ann['bbox'],
            'area': ann['area'],
            'segmentation': ann.get('segmentation', []),
            'iscrowd': ann.get('iscrowd', 0)
        }
        merged_annotations.append(new_ann)

    # Формируем финальный датасет
    merged_dataset = {
        'info': ds1.get('info', {}) or ds2.get('info', {}),
        'licenses': ds1.get('licenses', []) + [lic for lic in ds2.get('licenses', [])
                                               if lic not in ds1.get('licenses', [])],
        'images': merged_images,
        'annotations': merged_annotations,
        'categories': merged_categories
    }

    # Сохраняем результат
    os.makedirs(output_path + r"\annotations", exist_ok=True)
    with open(output_path + r"\annotations\instances_" + ds_type + "2017.json", 'w') as f:
        json.dump(merged_dataset, f, indent=2, ensure_ascii=False)

    os.makedirs(output_path + "\\" + ds_type + "2017", exist_ok=True)
    copy_folder_contents(ds1_path + "\\" + ds_type + "2017", output_path + "\\" + ds_type + "2017")
    copy_folder_contents(ds2_path + "\\" + ds_type + "2017", output_path + "\\" + ds_type + "2017")


# Пример использования:
# merge_coco_datasets(
#     "train",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-train-4",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\merge-dataset-2",
#     r"D:\Disser\Datasets\TEST-DATASET\temps\merge-dataset-3"
# )
merge_coco_datasets(
    "val",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-val-0",
    r"D:\Disser\Datasets\TEST-DATASET\temps\dataset-val-1",
    r"D:\Disser\Datasets\TEST-DATASET\temps\merge-dataset-3"
)

print("Слияние завершено!")
