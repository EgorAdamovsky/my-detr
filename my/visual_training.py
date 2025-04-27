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

    train_class_error = [log["train_class_error"] for log in log_data]
    test_class_error = [log["test_class_error"] for log in log_data]

    train_lr = [log["train_lr"] for log in log_data]

    coco_eval_bbox = [log["test_coco_eval_bbox"] for log in log_data]

    # Создаем графики
    fig, axes = plt.subplots(3, 1, figsize=(10, 18))

    # График потерь (loss)
    axes[0].plot(epochs, train_loss, label="Train Loss", marker='.')
    axes[0].plot(epochs, test_loss, label="Test Loss", marker='.')
    axes[0].set_title("Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # График ошибки классификации (class error)
    axes[1].plot(epochs, train_class_error, label="Train Class Error", marker='.')
    axes[1].plot(epochs, test_class_error, label="Test Class Error", marker='.')
    axes[1].set_title("Class Error over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Class Error (%)")
    axes[1].legend()
    axes[1].grid(True)

    # График learning rate
    axes[2].plot(epochs, train_lr, label="Learning Rate", marker='o', color='green')
    axes[2].set_title("Learning Rate over Epochs")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].legend()
    axes[2].grid(True)

    # Дополнительный график для COCOEval BBox (если нужно)

    COCOEvalBBox = ["AP (Average Precision)", "AP@IoU=0.50", "AP@IoU=0.75", "AP (small)", "AP (medium)", "AP (large)", "AR (Average Recall)"]
    if coco_eval_bbox:
        fig_coco, ax_coco = plt.subplots(figsize=(10, 6))
        for i, metric in enumerate(coco_eval_bbox[0]):
            metric_values = [bbox[i] for bbox in coco_eval_bbox]
            ax_coco.plot(epochs, metric_values, label=f"COCO BBox Metric {i}", marker='.')

        ax_coco.set_title("COCOEval BBox Metrics over Epochs")
        ax_coco.set_xlabel("Epoch")
        ax_coco.set_ylabel("Metric Value")
        ax_coco.legend()
        ax_coco.grid(True)

    # Показываем графики
    plt.tight_layout()
    plt.show()


# Пример использования
if __name__ == "__main__":
    # Логи в виде строки (замените на путь к файлу, если логи находятся в файле)
    with open(r"D:\Disser\Programs\OTHER-MODELS\detr\my\video-out\log.txt", "r") as file:
        logs = file.read()

    # Читаем логи
    log_data = read_logs(logs)

    # Строим графики
    plot_training_progress(log_data)
