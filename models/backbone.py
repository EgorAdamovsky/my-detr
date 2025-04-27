# Импорт необходимых библиотек
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter  # Для извлечения промежуточных слоев модели
from util.misc import NestedTensor, is_main_process  # Структура данных для хранения тензоров и масок
from .position_encoding import build_position_encoding  # Построение позиционных кодирований


# Реализация замороженной BatchNorm2d (параметры не обучаются)
class FrozenBatchNorm2d(torch.nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        # Регистрация буферов для весов, смещений и статистик
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    # Удаление 'num_batches_tracked' из state_dict для совместимости
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    # Прямой проход: нормализация с использованием замороженных параметров
    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()  # Вычисление масштаба с учетом дисперсии
        bias = b - rm * scale  # Корректировка смещения
        return x * scale + bias  # Применение нормализации


# Базовый класс для backbone модели
class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # Заморозка параметров backbone, кроме layer2-4 при train_backbone=True
        for name, parameter in backbone.named_parameters():
            if not train_backbone or ('layer2' not in name and 'layer3' not in name and 'layer4' not in name):
                parameter.requires_grad_(False)
        # Определение возвращаемых слоев в зависимости от флага
        return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"} if return_interm_layers else {
            'layer4': "0"}
        # Использование IntermediateLayerGetter для извлечения промежуточных слоев
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels  # Количество выходных каналов

    # Прямой проход: обработка NestedTensor и генерация масок
    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)  # Применение backbone к тензорам
        out = {}
        for name, x in xs.items():
            mask = tensor_list.mask
            assert mask is not None
            # Интерполяция маски до размера текущего слоя
            mask = F.interpolate(mask[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)  # Создание NestedTensor с данными и маской
        return out


# Класс ResNet backbone с замороженной BatchNorm
class Backbone(BackboneBase):

    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool):
        backbone = getattr(torchvision.models, name)(replace_stride_with_dilation=[False, False, dilation],
                                                     pretrained=is_main_process(),
                                                     norm_layer=FrozenBatchNorm2d)  # Создание ResNet с заменой шага на дилатацию и замороженной BatchNorm
        num_channels = 512 if name in (
            'resnet18', 'resnet34') else 2048  # Определение количества каналов в зависимости от модели
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


# Класс для объединения backbone и позиционного кодирования
class Joiner(nn.Sequential):

    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    # Прямой проход: добавление позиционных кодирований к выходам backbone
    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)  # Применение backbone
        out = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            pos.append(self[1](x).to(x.tensors.dtype))  # Генерация позиционного кодирования для текущего слоя
        return out, pos  # Возврат списка выходов и позиционных эмбеддингов


# Функция для создания backbone модели
def build_backbone(args):
    position_embedding = build_position_encoding(args)  # Создание позиционного кодирования
    train_backbone = args.lr_backbone > 0  # Определение, нужно ли обучать backbone
    backbone = Backbone(args.backbone, train_backbone, False, args.dilation)  # Создание CNN с заданными параметрами
    model = Joiner(backbone, position_embedding)  # Объединение backbone и позиционного кодирования
    model.num_channels = backbone.num_channels  # Установка количества каналов
    return model
