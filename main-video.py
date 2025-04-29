# Импорт необходимых библиотек
import argparse  # Для обработки аргументов командной строки
import datetime  # Для работы с датами и временем
import json  # Для сериализации/десериализации данных
import random  # Для генерации случайных чисел
import time  # Для измерения времени выполнения
from pathlib import Path  # Для работы с путями файловой системы
import numpy as np  # Библиотека для работы с массивами
import torch  # Основная библиотека для глубокого обучения
from torch.utils.data import DataLoader  # Для загрузки данных в batches
import util.misc as utils  # Вспомогательные функции из модуля util
from datasets import build_dataset, get_coco_api_from_dataset  # Функции для работы с датасетами
from engine import evaluate, train_one_epoch  # Функции для обучения и оценки модели
from models import build_model  # Функция для построения модели DETR
import matplotlib  # Библиотека для визуализации
from matplotlib import pyplot as plt  # Модуль для построения графиков
import warnings  # Игнорим предупреждения

warnings.filterwarnings("ignore")  # Игнорим предупреждения
# matplotlib.use('TkAgg')  # Установка бэкенда для отображения графиков


def get_args_parser():
    # Создание парсера аргументов командной строки
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    ## ЭКСПЕРИМЕНТ #####################################################################################################
    parser.add_argument('--prevs', default=2, type=int, help='Количество предыдущих кадров')
    parser.add_argument('--show', default=1, type=int, help='Генерировать ли отладочные картинки')
    ## ЭКСПЕРИМЕНТ #####################################################################################################

    # Параметры оптимизатора
    parser.add_argument('--lr', default=1e-4, type=float, help='Скорость обучения')
    parser.add_argument('--lr_backbone', default=1e-5, type=float, help='Скорость обучения для backbone')
    parser.add_argument('--batch_size', default=2, type=int, help='Размер батча')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Регуляризация, штраф за большие веса')
    parser.add_argument('--epochs', default=5, type=int, help='Количество эпох обучения')
    parser.add_argument('--lr_drop', default=200, type=int, help='Эпоха для снижения learning rate')
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='Лимит для ограничения градиентов')

    # Параметры модели
    parser.add_argument('--frozen_weights', type=str, default=None, help='Обученная модель для заморозки весов')

    # Параметры backbone
    parser.add_argument('--backbone', default='resnet50', type=str, help='Архитектура backbone')
    parser.add_argument('--dilation', action='store_true', help='Использовать дилатацию в последнем блоке backbone')

    # Параметры Transformer
    parser.add_argument('--enc_layers', default=6, type=int, help='Количество слоев энкодера')
    parser.add_argument('--dec_layers', default=6, type=int, help='Количество слоев декодера')
    parser.add_argument('--dim_feedforward', default=2048, type=int, help='Размер промежуточного слоя в Transformer')
    parser.add_argument('--hidden_dim', default=256, type=int, help='Размер эмбеддингов (скрытый размер)')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout в Transformer')
    parser.add_argument('--nheads', default=8, type=int, help='Количество голов в Transformer')
    parser.add_argument('--num_queries', default=100, type=int, help='Количество object queries')
    parser.add_argument('--pre_norm', action='store_true', help='Использовать преднормализацию в Transformer')

    # Параметры функции потерь
    parser.add_argument('--set_cost_class', default=1, type=float, help='Совпадение класса в венгерском алгоритме 1')
    parser.add_argument('--set_cost_bbox', default=5, type=float, help='Точность бокса в венгерском алгоритме 5')
    parser.add_argument('--set_cost_giou', default=2, type=float, help='Выравнивание бокса в венгерском алгоритме 2')
    parser.add_argument('--mask_loss_coef', default=1, type=float, help='Для сегментации, игнорируем) 1')
    parser.add_argument('--dice_loss_coef', default=1, type=float, help='Для сегментации, игнорируем) 1')
    parser.add_argument('--bbox_loss_coef', default=5, type=float, help='Точность бокса в основном loss 5')
    parser.add_argument('--giou_loss_coef', default=2, type=float, help='Выравнивание бокса в основном loss 2')
    parser.add_argument('--eos_coef', default=0.1, type=float, help='Штраф за ложное срабатывание 0.1')

    # Параметры датасета
    parser.add_argument('--dataset_file', default='coco', help='Имя датасета')
    parser.add_argument('--coco_path', type=str, help='Путь к датасету COCO')
    parser.add_argument('--remove_difficult', action='store_true', help='Удалить сложные объекты')

    # Параметры вывода и сохранения
    parser.add_argument('--output_dir', default='', help='Директория для сохранения результатов')
    parser.add_argument('--device', default='cuda', help='Устройство для обучения (cuda/cpu)')
    parser.add_argument('--seed', default=42, type=int, help='Случайное seed для воспроизводимости')
    parser.add_argument('--resume', default='', help='Путь к чекпоинту для продолжения обучения')
    parser.add_argument('--start_epoch', default=0, type=int, help='Начальная эпоха при продолжении обучения')
    parser.add_argument('--eval', action='store_true', help='Режим оценки модели')
    parser.add_argument('--num_workers', default=2, type=int, help='Количество процессов для загрузки данных')

    # Параметры распределенного обучения
    parser.add_argument('--world_size', default=1, type=int, help='Количество процессов для распределенного обучения')
    parser.add_argument('--dist_url', default='env://', help='URL для настройки распределенного обучения')

    return parser


def SetSeed():
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main(args):
    # print(args)  # Вывод переданных аргументов
    device = torch.device(args.device)  # Инициализация устройства (GPU/CPU)
    SetSeed()  # Установка seed для воспроизводимости (пусть будет)
    model, criterion, postprocessors = build_model(args)  # Создание модели, функции потерь и постпроцессоров
    model.to(device)  # Перемещение модели на GPU
    model_without_ddp = model  # Модель без распределения, она нам и нужна
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Количество параметров:', n_parameters)  # Вывод количества обучаемых параметров
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
         "lr": args.lr_backbone},
    ]  # Настройка оптимизатора с разными learning rate для backbone и трансформера
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)  # Планировщик learning rate
    dataset_train = build_dataset(image_set='train', args=args)  # Создание датасета train
    dataset_val = build_dataset(image_set='val', args=args)  # Создание датасета val
    sampler_train = torch.utils.data.RandomSampler(dataset_train)  # Сэмплер для train (порядок выдачи рандомный)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)  # Сэмплер для val (порядок выдачи как есть)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size,
                                                        drop_last=True)  # Режет train на батчи
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn,
                                   num_workers=args.num_workers)  # Итоговый загрузчик train
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, drop_last=False,
                                 collate_fn=utils.collate_fn, num_workers=args.num_workers)  # Итоговый загрузчик val
    base_ds = get_coco_api_from_dataset(dataset_val)  # Получение API COCO для оценки
    output_dir = Path(args.output_dir)

    # Загрузка предобученных весов (если указано)
    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    # Восстановление из чекпоинта (если указано)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    # Режим оценки модели
    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device,
                                              args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    ####################################################################################################################

    print("Начало обучения!")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(  # Одна эпоха обучения
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
        lr_scheduler.step()  # Обновление learning rate
        if args.output_dir:  # Сохранение чекпоинтов
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        test_stats, coco_evaluator = evaluate(  # Оценка модели на валидационном наборе
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},  # Логирование статистик
                     **{f'test_{k}': v for k, v in test_stats.items()}, 'epoch': epoch, 'n_parameters': n_parameters}
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Общее время обучения:', total_time_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR скрипт обучения и оценки', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
