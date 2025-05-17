import json
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
path = r"D:\Disser\Programs\OTHER-MODELS\my-detr\my\out\log.txt"


# Функция для чтения логов из файла или строки
def read_logs(logs_):
    data = []
    for line in logs_.splitlines():
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"Ошибка при декодировании строки: {line}")
    return data


# Функция для построения графиков
def plot_training_progress(log_data_):
    epochs = [log["epoch"] for log in log_data_]

    # Собираем данные для графиков
    train_loss = [log["train_loss"] for log in log_data_]
    test_loss = [log["test_loss"] for log in log_data_]

    train_class_error = [log["train_class_error_unscaled"] for log in log_data_]
    test_class_error = [log["test_class_error_unscaled"] for log in log_data_]

    train_box_error = [log["train_loss_bbox_unscaled"] for log in log_data_]
    test_box_error = [log["test_loss_bbox_unscaled"] for log in log_data_]

    train_giou_error = [log["train_loss_giou_unscaled"] for log in log_data_]
    test_giou_error = [log["test_loss_giou_unscaled"] for log in log_data_]

    coco_eval_bbox = [log["test_coco_eval_bbox"] for log in log_data_]

    # Создаем графики
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))

    # График потерь (loss)
    g, = axes[0, 0].plot(epochs, train_loss, '-.', label="Train")
    g.set_color("black")
    g, = axes[0, 0].plot(epochs, test_loss, '-', label="Test")
    g.set_color("black")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].set_xlim(0, len(epochs) - 1)
    axes[0, 0].set_ylim(0, 1 * max(max(train_loss), max(test_loss)))
    axes[0, 0].grid(True)

    # График потерь (класс)
    g, = axes[0, 1].plot(epochs, train_class_error, '-.', label="Train")
    g.set_color("black")
    g, = axes[0, 1].plot(epochs, test_class_error, '-', label="Test")
    g.set_color("black")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Class Error, %")
    axes[0, 1].legend(loc='upper right')
    axes[0, 1].set_xlim(0, len(epochs) - 1)
    axes[0, 1].set_ylim(0, 1 * max(max(train_class_error), max(test_class_error)))
    axes[0, 1].grid(True)

    # График потерь (bbox)
    g, = axes[1, 0].plot(epochs, train_box_error, '-.', label="Train")
    g.set_color("black")
    g, = axes[1, 0].plot(epochs, test_box_error, '-', label="Test")
    g.set_color("black")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Box Error")
    axes[1, 0].legend(loc='upper right')
    axes[1, 0].set_xlim(0, len(epochs) - 1)
    axes[1, 0].set_ylim(0, 1 * max(max(train_box_error), max(test_box_error)))
    axes[1, 0].grid(True)

    # График потерь (giou)
    g, = axes[1, 1].plot(epochs, train_giou_error, '-.', label="Train")
    g.set_color("black")
    g, = axes[1, 1].plot(epochs, test_giou_error, '-', label="Test")
    g.set_color("black")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("GIoU Error")
    axes[1, 1].legend(loc='upper right')
    axes[1, 1].set_xlim(0, len(epochs) - 1)
    axes[1, 1].set_ylim(0, 1 * max(max(train_giou_error), max(test_giou_error)))
    axes[1, 1].grid(True)

    # Дополнительный график для COCOEval BBox
    COCOEvalBBox = ["AP 0.50:0.95", "AP 0.50", "AP 0.75",
                    "AP small", "AP medium", "AP large",
                    "AR maxDets=1", "AR maxDets=10", "AR maxDets=100",
                    "AR small", "AR medium", "AR large"]

    # Стили линий: сплошная, штриховая, пунктирно-штриховая
    linestyles = ['-', '--', '-.']  # 3 стиля под 3 линии на графике

    if coco_eval_bbox:
        # Разбиваем на группы по 3 метрики
        groups = [COCOEvalBBox[i:i + 3] for i in range(0, len(COCOEvalBBox), 3)]

        # Создаем 4 подграфика (2x2)
        figs, axs = plt.subplots(2, 2, figsize=(8, 7))
        axs = axs.flatten()

        for idx, group in enumerate(groups):
            ax = axs[idx]

            for i, metric in enumerate(group):
                metric_index = COCOEvalBBox.index(metric)
                metric_values = [bbox[metric_index] for bbox in coco_eval_bbox]
                ax.plot(
                    epochs,
                    metric_values,
                    label=metric,
                    marker='',
                    color='black',
                    linestyle=linestyles[i % len(linestyles)],  # разные стили для каждой из 3 линий
                    linewidth=1.5
                )

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Metric Value")
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(loc='upper left')
            ax.set_xlim(0, len(epochs) - 1)
            ax.set_ylim(0, None)

        # Убираем лишние оси, если их больше чем нужно
        for j in range(len(groups), len(axs)):
            figs.delaxes(axs[j])

        plt.tight_layout()
        plt.show()


# Пример использования
if __name__ == "__main__":
    with open(path, "r") as file:
        logs = file.read()

    # Читаем логи
    log_data = read_logs(logs)

    # Строим графики
    plot_training_progress(log_data)
