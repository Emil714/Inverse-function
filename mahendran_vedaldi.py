"""
Реализация метода инверсии эмбеддингов из работы:
Mahendran & Vedaldi, "Understanding Deep Image Representations by Inverting Them", CVPR 2015.

Идея: дано изображение x0 и его эмбеддинг Phi0 = Phi(x0).
Найти изображение x*, минимизирующее:

    L(x) = ||Phi(x) - Phi0||^2 / ||Phi0||^2  +  lambda_a * ||x||_alpha^alpha
                                              +  lambda_TV * TV_beta(x)

Это оптимизация в пространстве пикселей градиентным спуском.
Регуляризаторы (alpha-norm и Total Variation) играют роль natural image prior.

Запуск:
    python mahendran_vedaldi.py --image cat.jpg --layer layer3 --iters 400

Требования: PyTorch с CUDA, torchvision, Pillow, matplotlib.
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


# Стандартная нормализация ImageNet
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------------------
# 1. Энкодер: ResNet50 с возможностью извлекать эмбеддинг с заданного слоя
# ---------------------------------------------------------------------------
class ResNetFeatureExtractor(nn.Module):
    """Извлекает признаки с указанного слоя предобученной ResNet50."""

    LAYER_NAMES = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']

    def __init__(self, layer: str = 'layer3'):
        super().__init__()
        assert layer in self.LAYER_NAMES, f"layer must be one of {self.LAYER_NAMES}"
        self.layer_name = layer

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)

        # Собираем последовательность слоёв до нужного включительно.
        # ResNet50 устроена так: conv1 -> bn1 -> relu -> maxpool -> layer1..4 -> avgpool -> fc
        layers = []
        layers.append(('conv1', nn.Sequential(backbone.conv1, backbone.bn1,
                                              backbone.relu, backbone.maxpool)))
        layers.append(('layer1',  backbone.layer1))
        layers.append(('layer2',  backbone.layer2))
        layers.append(('layer3',  backbone.layer3))
        layers.append(('layer4',  backbone.layer4))
        layers.append(('avgpool', nn.Sequential(backbone.avgpool, nn.Flatten())))
        layers.append(('fc',      backbone.fc))

        # Берём слои до целевого включительно
        idx = self.LAYER_NAMES.index(layer)
        self.blocks = nn.Sequential(*[l[1] for l in layers[:idx + 1]])

    def forward(self, x_unnorm: torch.Tensor) -> torch.Tensor:
        """Принимает изображение в [0,1], сама нормализует под ImageNet."""
        mean = IMAGENET_MEAN.to(x_unnorm.device)
        std  = IMAGENET_STD.to(x_unnorm.device)
        x = (x_unnorm - mean) / std
        return self.blocks(x)


# ---------------------------------------------------------------------------
# 2. Регуляризаторы из статьи
# ---------------------------------------------------------------------------
def alpha_norm(x: torch.Tensor, alpha: int = 6) -> torch.Tensor:
    """R_alpha(x) = sum |x|^alpha  — препятствует чрезмерным значениям пикселей."""
    return x.abs().pow(alpha).sum()


def total_variation(x: torch.Tensor, beta: float = 2.0) -> torch.Tensor:
    """
    TV-регуляризатор V^beta(x) = sum [(dx_h)^2 + (dx_v)^2]^(beta/2).
    Авторы рекомендуют beta=2 — устраняет артефакты «спайков» при наличии max-pool.
    """
    dh = x[:, :, 1:, :] - x[:, :, :-1, :]   # вертикальные градиенты
    dw = x[:, :, :, 1:] - x[:, :, :, :-1]   # горизонтальные градиенты
    # Привести к одинаковому размеру для покомпонентного суммирования
    dh = dh[:, :, :, :-1]
    dw = dw[:, :, :-1, :]
    return ((dh ** 2 + dw ** 2 + 1e-8) ** (beta / 2)).sum()


# ---------------------------------------------------------------------------
# 3. Загрузка и предобработка изображения
# ---------------------------------------------------------------------------
def load_image(path: str, size: int = 224, device: str = 'cuda') -> torch.Tensor:
    """Возвращает тензор [1, 3, H, W] в [0, 1]."""
    img = Image.open(path).convert('RGB')
    transform = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor()])
    return transform(img).unsqueeze(0).to(device)


def tensor_to_np(t: torch.Tensor) -> np.ndarray:
    """Тензор [1,3,H,W] в [0,1] -> numpy [H,W,3]."""
    arr = t.detach().cpu().clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
    return arr


# ---------------------------------------------------------------------------
# 4. Главная процедура инверсии
# ---------------------------------------------------------------------------
def invert(
    target_image: torch.Tensor,
    feature_extractor: nn.Module,
    n_iters: int = 400,
    lr: float = 0.05,
    lambda_alpha: float = 1e-7,
    lambda_tv: float = 1e-3,
    alpha: int = 6,
    beta: float = 2.0,
    log_every: int = 50,
):
    """
    Восстанавливает изображение по его эмбеддингу с заданного слоя.

    target_image : тензор [1,3,H,W] в [0,1] — оригинал, чей эмбеддинг будем матчить
    feature_extractor : обученный энкодер с замороженными весами
    """
    device = target_image.device

    # 1) Получаем целевой эмбеддинг
    with torch.no_grad():
        target_phi = feature_extractor(target_image)
        norm_phi = (target_phi ** 2).sum().clamp(min=1e-8)

    # 2) Инициализируем изображение белым шумом малой амплитуды (как в статье)
    #x = torch.randn_like(target_image) * 0.01 + 0.5
    #x = x.clamp(0, 1).clone().detach().requires_grad_(True)

    # 2) Инициализируем из самого изображения + малый шум
    #x = target_image.clone() + torch.randn_like(target_image) * 0.01
    #x = x.clamp(0, 1).clone().detach().requires_grad_(True)

    x = torch.randn_like(target_image) + 0.5
    x = x.clamp(0, 1).clone().detach().requires_grad_(True)

    # 3) Оптимизатор: SGD с моментом 0.9 (как в статье)
    #optimizer = optim.SGD([x], lr=lr, momentum=0.9)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    optimizer = optim.Adam([x], lr=lr)

    history = []
    for i in range(n_iters):
        optimizer.zero_grad()

        phi = feature_extractor(x)

        # Нормализованная feature loss
        feat_loss = ((phi - target_phi) ** 2).sum() / norm_phi
        # Регуляризаторы
        reg_alpha = lambda_alpha * alpha_norm(x, alpha)
        reg_tv    = lambda_tv * total_variation(x, beta)

        loss = feat_loss + reg_alpha + reg_tv
        loss.backward()
        optimizer.step()
        #scheduler.step()

        # После шага — мягкое возвращение в [0,1] (clip без потери градиента в next iter)
        with torch.no_grad():
            x.clamp_(0, 1)

        if (i + 1) % log_every == 0 or i == 0:
            history.append((i + 1, feat_loss.item(), reg_alpha.item(), reg_tv.item()))
            print(f"iter {i+1:4d} | feat_loss {feat_loss.item():.5f} "
                  f"| alpha {reg_alpha.item():.5f} | TV {reg_tv.item():.5f}")

    return x.detach(), history


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='путь к изображению')
    parser.add_argument('--layer', type=str, default='layer3',
                        choices=ResNetFeatureExtractor.LAYER_NAMES,
                        help='слой ResNet50, эмбеддинг которого инвертируем')
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--lambda_alpha', type=float, default=1e-7)
    parser.add_argument('--lambda_tv', type=float, default=1e-3)
    parser.add_argument('--out_dir', type=str, default='./outputs')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Загружаем изображение и энкодер
    img = load_image(args.image, size=224, device=device)
    encoder = ResNetFeatureExtractor(layer=args.layer).to(device).eval()

    print(f'Inverting layer "{args.layer}"...')
    recon, history = invert(
        img, encoder,
        n_iters=args.iters, lr=args.lr,
        lambda_alpha=args.lambda_alpha, lambda_tv=args.lambda_tv,
    )

    # Сохраняем сравнение оригинал vs реконструкция
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(tensor_to_np(img));    axes[0].set_title('Original');         axes[0].axis('off')
    axes[1].imshow(tensor_to_np(recon));  axes[1].set_title(f'Reconstruction\n({args.layer})')
    axes[1].axis('off')
    plt.tight_layout()

    name = Path(args.image).stem
    out_path = Path(args.out_dir) / f'mv_{name}_{args.layer}.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
