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

    train_box_error = [log["train_loss_bbox"] for log in log_data]
    test_box_error = [log["test_loss_bbox"] for log in log_data]

    train_giou_error = [log["train_loss_giou"] for log in log_data]
    test_giou_error = [log["test_loss_giou"] for log in log_data]

    train_lr = [log["train_lr"] for log in log_data]

    coco_eval_bbox = [log["test_coco_eval_bbox"] for log in log_data]

    # Создаем графики
    fig, axes = plt.subplots(2, 1, figsize=(10, 18))

    # График потерь (loss)
    g, = axes[0].plot(epochs, train_loss, label="Train Loss", marker='.')
    g.set_color("black")
    g, = axes[0].plot(epochs, train_box_error, label=" Train Box Loss", marker=',')
    g.set_color("black")
    g, = axes[0].plot(epochs, train_giou_error, label=" Train GIOU Loss", marker='+')
    g.set_color("black")
    g, = axes[0].plot(epochs, test_loss, label="Test Loss", marker='.')
    g.set_color("red")
    g, = axes[0].plot(epochs, test_box_error, label=" Test Box Loss", marker=',')
    g.set_color("red")
    g, = axes[0].plot(epochs, test_giou_error, label=" Test GIOU Loss", marker='+')
    g.set_color("red")
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
    # axes[2].plot(epochs, train_lr, label="Learning Rate", marker='o', color='green')
    # axes[2].set_title("Learning Rate over Epochs")
    # axes[2].set_xlabel("Epoch")
    # axes[2].set_ylabel("Learning Rate")
    # axes[2].legend()
    # axes[2].grid(True)

    # Дополнительный график для COCOEval BBox
    COCOEvalBBox = ["AP 0.50:0.95", "AP 0.50", "AP 0.75", "AP 0.50:0.95, area=small", "AP 0.50:0.95, area=medium",
                    "AP 0.50:0.95, area=large", "AR 0.50:0.95, maxDets=1", "AR 0.50:0.95, maxDets=10",
                    "AR 0.50:0.95, maxDets=100", "AR 0.50:0.95, area=small",
                    "AR 0.50:0.95, area=medium", "AR 0.50:0.95, area=large"]
    if coco_eval_bbox:
        fig_coco, ax_coco = plt.subplots(figsize=(10, 6))
        for i, metric in enumerate(coco_eval_bbox[0]):
            metric_values = [bbox[i] for bbox in coco_eval_bbox]
            ax_coco.plot(epochs, metric_values, label=f"COCO BBox Metric {COCOEvalBBox[i]}", marker='.')

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
    with open(r"D:\Disser\Programs\OTHER-MODELS\my-detr\my\out-test\log.txt", "r") as file:
        logs = file.read()

    # Читаем логи
    log_data = read_logs(logs)

    # Строим графики
    plot_training_progress(log_data)
