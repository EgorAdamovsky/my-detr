import os
import shutil
import argparse


def copy_video_files(source_dir, target_dir):
    video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', '.mpeg', '.mpg', '.m4v')
    count = 0

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(video_extensions):
                src_file = os.path.join(root, file)
                dst_file = os.path.join(target_dir, file)

                if os.path.exists(dst_file):
                    print(f"⚠️ Файл {file} уже существует в целевой папке. Пропускаю.")
                    continue

                try:
                    shutil.copy2(src_file, dst_file)
                    print(f"✅ Скопирован: {file}")
                    count += 1
                except Exception as e:
                    print(f"❌ Ошибка при копировании {file}: {str(e)}")

    return count


def main():
    parser = argparse.ArgumentParser(description='Копирование видеофайлов из подпапок в указанную директорию')
    parser.add_argument('--source', help='Исходная директория для поиска')
    parser.add_argument('--target', help='Целевая директория для копирования')
    args = parser.parse_args()

    if not os.path.isdir(args.source):
        print(f"❌ Исходная директория не существует: {args.source}")
        return

    os.makedirs(args.target, exist_ok=True)

    print(f"🔍 Поиск видеофайлов в: {args.source}")
    total = copy_video_files(args.source, args.target)
    print(f"🏁 Завершено. Всего скопировано файлов: {total}")


if __name__ == "__main__":
    main()
