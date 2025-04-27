import math
import torch
from matplotlib import pyplot as plt
from torch import nn
from util.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


def visualize_positional_encoding(pos_embed):
    # Создаем фиктивный вход размером 32x32
    x = torch.zeros(1, 3, 32, 32)  # (batch, channels, H, W)
    mask = torch.zeros(1, 32, 32).bool()  # Все позиции видимы (нет маскировки)
    tensor_list = NestedTensor(x, mask)

    # pos_embed = PositionEmbeddingSine(num_pos_feats=64, normalize=True)
    pos = pos_embed(tensor_list).squeeze(0)  # Убираем батч-размер

    # Визуализируем первые 4 канала
    fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(16):
        ax = axs[i // 4, i % 4]
        channel = pos[i].detach().numpy()
        ax.imshow(channel, cmap='viridis')
        ax.set_title(f'Channel {i + 1}')
        ax.axis('off')
    plt.show()


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    # visualize_positional_encoding(position_embedding)
    return position_embedding
