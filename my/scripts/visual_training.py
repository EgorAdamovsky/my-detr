import json
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


# Функция для чтения логов из файла или строки
def read_logs(logs):
    data = []
    for line in logs.splitlines():
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"Ошибка при декодировании строки: {line}")
    return data


# Функция для построения графиков
def plot_training_progress(log_data):
    epochs = [log["epoch"] for log in log_data]

    # Собираем данные для графиков
    train_loss = [log["train_loss"] for log in log_data]
    test_loss = [log["test_loss"] for log in log_data]

    train_class_error = [log["train_class_error_unscaled"] for log in log_data]
    test_class_error = [log["test_class_error_unscaled"] for log in log_data]

    train_box_error = [log["train_loss_bbox_unscaled"] for log in log_data]
    test_box_error = [log["test_loss_bbox_unscaled"] for log in log_data]

    train_giou_error = [log["train_loss_giou_unscaled"] for log in log_data]
    test_giou_error = [log["test_loss_giou_unscaled"] for log in log_data]

    train_lr = [log["train_lr"] for log in log_data]

    coco_eval_bbox = [log["test_coco_eval_bbox"] for log in log_data]

    # Создаем графики
    fig, axes = plt.subplots(2, 2, figsize=(18, 18))

    # График потерь (loss)
    g, = axes[0, 0].plot(epochs, train_loss, '-', label="Train")
    g.set_color("blue")
    g, = axes[0, 0].plot(epochs, test_loss, '-.', label="Test")
    g.set_color("red")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].set_xlim(0, len(epochs) - 1)
    axes[0, 0].set_ylim(0, 1 * max(max(train_loss), max(test_loss)))
    axes[0, 0].grid(True)

    # График потерь (класс)
    g, = axes[0, 1].plot(epochs, train_class_error, '-', label="Train")
    g.set_color("blue")
    g, = axes[0, 1].plot(epochs, test_class_error, '-.', label="Test")
    g.set_color("red")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Class Error, %")
    axes[0, 1].legend(loc='upper right')
    axes[0, 1].set_xlim(0, len(epochs) - 1)
    axes[0, 1].set_ylim(0, 1 * max(max(train_class_error), max(test_class_error)))
    axes[0, 1].grid(True)

    # График потерь (bbox)
    g, = axes[1, 0].plot(epochs, train_box_error, '-', label="Train")
    g.set_color("blue")
    g, = axes[1, 0].plot(epochs, test_box_error, '-.', label="Test")
    g.set_color("red")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Box Error")
    axes[1, 0].legend(loc='upper right')
    axes[1, 0].set_xlim(0, len(epochs) - 1)
    axes[1, 0].set_ylim(0, 1 * max(max(train_box_error), max(test_box_error)))
    axes[1, 0].grid(True)

    # График потерь (giou)
    g, = axes[1, 1].plot(epochs, train_giou_error, '-', label="Train")
    g.set_color("blue")
    g, = axes[1, 1].plot(epochs, test_giou_error, '-.', label="Test")
    g.set_color("red")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("GIoU Error")
    axes[1, 1].legend(loc='upper right')
    axes[1, 1].set_xlim(0, len(epochs) - 1)
    axes[1, 1].set_ylim(0, 1 * max(max(train_giou_error), max(test_giou_error)))
    axes[1, 1].grid(True)

    # График learning rate
    # axes[2].plot(epochs, train_lr, label="Learning Rate", marker='o', color='green')
    # axes[2].set_title("Learning Rate over Epochs")
    # axes[2].set_xlabel("Epoch")
    # axes[2].set_ylabel("Learning Rate")
    # axes[2].legend()
    # axes[2].grid(True)

    # Дополнительный график для COCOEval BBox
    COCOEvalBBox = ["AP 0.50:0.95", "AP 0.50", "AP 0.75",
                    "AP small", "AP medium", "AP large",
                    "AR maxDets=1", "AR maxDets=10", "AR maxDets=100",
                    "AR small", "AR medium", "AR large"]

    if coco_eval_bbox:
        fig_coco, ax_coco = plt.subplots(figsize=(12, 6))
        colors = ["black", "gray", "lightgray",
                  "red", "tomato", "chocolate",
                  "blue", "dodgerblue", "aqua",
                  "green", "limegreen", "lime"]

        for i, metric in enumerate(COCOEvalBBox):
            metric_values = [bbox[i] for bbox in coco_eval_bbox]
            ax_coco.plot(epochs, metric_values, label=metric, marker='.', color=colors[i % len(colors)])

        ax_coco.set_xlabel("Epoch")
        ax_coco.set_ylabel("Metric Value")
        ax_coco.grid(True, linestyle='--', alpha=0.5)

        # Легенда справа от графика
        ax_coco.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))

        # Ограничение по осям
        ax_coco.set_xlim(0, len(epochs) - 1)
        ax_coco.set_ylim(0, None)  # Автоматический верхний предел или задай вручную

    # Показываем графики
    plt.tight_layout()
    plt.show()


# Пример использования
if __name__ == "__main__":
    # Логи в виде строки (замените на путь к файлу, если логи находятся в файле)
    with open(r"D:\Disser\Programs\OTHER-MODELS\my-detr\my\out\log.txt", "r") as file:
        logs = file.read()

    # Читаем логи
    log_data = read_logs(logs)

    # Строим графики
    plot_training_progress(log_data)
