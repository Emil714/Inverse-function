"""
Реализация метода инверсии эмбеддингов из работы:
Dosovitskiy & Brox, "Inverting Visual Representations with Convolutional Networks", CVPR 2016.

Идея: обучить up-convolutional сеть-декодер f(phi; w), которая по эмбеддингу phi
восстанавливает изображение x. Функция потерь — MSE:

    L(w) = sum_i || x_i - f(Phi(x_i); w) ||^2

После обучения инверсия = один прямой проход через декодер (~5 ms на GPU).

Запуск (обучение):
    python dosovitskiy_brox.py --mode train --layer layer3 --data_dir /path/to/imagenet/train

Запуск (инференс):
    python dosovitskiy_brox.py --mode infer --layer layer3 --image cat.jpg \
        --checkpoint checkpoints/decoder_layer3.pt

Требования: PyTorch с CUDA, torchvision, Pillow, matplotlib.
Датасет: достаточно подмножества ImageNet (~50k изображений хватит для
демонстрации; в оригинале использовался полный ImageNet).
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------------------
# 1. Энкодер (тот же, что в Mahendran & Vedaldi)
# ---------------------------------------------------------------------------
class ResNetFeatureExtractor(nn.Module):
    """Извлекает признаки с указанного слоя ResNet50. Веса заморожены."""

    LAYER_NAMES = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']

    # Размерности эмбеддингов для входа 224x224
    LAYER_SHAPES = {
        'conv1':   (64,   56, 56),
        'layer1':  (256,  56, 56),
        'layer2':  (512,  28, 28),
        'layer3':  (1024, 14, 14),
        'layer4':  (2048,  7,  7),
        'avgpool': (2048,),         # вектор
        'fc':      (1000,),         # вектор (логиты)
    }

    def __init__(self, layer: str = 'layer3'):
        super().__init__()
        assert layer in self.LAYER_NAMES
        self.layer_name = layer

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)

        layers = [
            ('conv1', nn.Sequential(backbone.conv1, backbone.bn1,
                                    backbone.relu, backbone.maxpool)),
            ('layer1',  backbone.layer1),
            ('layer2',  backbone.layer2),
            ('layer3',  backbone.layer3),
            ('layer4',  backbone.layer4),
            ('avgpool', nn.Sequential(backbone.avgpool, nn.Flatten())),
            ('fc',      backbone.fc),
        ]
        idx = self.LAYER_NAMES.index(layer)
        self.blocks = nn.Sequential(*[l[1] for l in layers[:idx + 1]])

    def forward(self, x_unnorm: torch.Tensor) -> torch.Tensor:
        mean = IMAGENET_MEAN.to(x_unnorm.device)
        std  = IMAGENET_STD.to(x_unnorm.device)
        x = (x_unnorm - mean) / std
        return self.blocks(x)


# ---------------------------------------------------------------------------
# 2. Up-convolutional декодер
# ---------------------------------------------------------------------------
class UpConvBlock(nn.Module):
    """Upsample x2 + Conv + LeakyReLU — как в оригинальной статье."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.conv(self.up(x)))


class Decoder2D(nn.Module):
    """
    Декодер для пространственных эмбеддингов (conv1, layer1..layer4).
    Принимает тензор [B, C, h, w] и возвращает изображение [B, 3, 224, 224].
    """
    def __init__(self, in_shape):
        super().__init__()
        c, h, w = in_shape
        # Сколько раз нужно увеличить на x2, чтобы получить 224
        # h может быть 7, 14, 28, 56 → upsamples = 5, 4, 3, 2
        self.start_h, self.start_w = h, w

        # Стартуем с проекционной свёртки 1x1 для приведения числа каналов к 256
        self.proj = nn.Sequential(
            nn.Conv2d(c, 256, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Считаем кол-во upsampling шагов до 224
        n_ups = 0
        cur = h
        while cur < 224:
            cur *= 2
            n_ups += 1

        # Цепочка up-conv блоков: 256 -> 256 -> 128 -> 64 -> 32 -> 16
        chans = [256, 256, 128, 64, 32, 16]
        chans = chans[:n_ups + 1]

        self.up_blocks = nn.ModuleList()
        for i in range(n_ups):
            self.up_blocks.append(UpConvBlock(chans[i], chans[i + 1]))

        # Финальная свёртка в 3 канала + sigmoid (выход в [0, 1])
        self.head = nn.Sequential(
            nn.Conv2d(chans[n_ups], 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, phi):
        x = self.proj(phi)
        for blk in self.up_blocks:
            x = blk(x)
        x = self.head(x)
        # Подгоним размер до точно 224x224 (на случай если upsample даёт чуть больше)
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = nn.functional.interpolate(x, size=(224, 224),
                                          mode='bilinear', align_corners=False)
        return x


class Decoder1D(nn.Module):
    """
    Декодер для векторных эмбеддингов (avgpool, fc).
    Принимает [B, D] -> восстанавливает [B, 3, 224, 224].
    Архитектура из статьи: 3 FC-слоя -> reshape -> 5 up-conv блоков.
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 4096),  nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4096, 4096),    nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4096, 256 * 7 * 7),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 7 -> 14 -> 28 -> 56 -> 112 -> 224
        self.up_blocks = nn.ModuleList([
            UpConvBlock(256, 256),
            UpConvBlock(256, 128),
            UpConvBlock(128,  64),
            UpConvBlock( 64,  32),
            UpConvBlock( 32,  16),
        ])
        self.head = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, phi):
        x = self.fc(phi)
        x = x.view(-1, 256, 7, 7)
        for blk in self.up_blocks:
            x = blk(x)
        x = self.head(x)
        if x.shape[-1] != 224:
            x = nn.functional.interpolate(x, size=(224, 224),
                                          mode='bilinear', align_corners=False)
        return x


def build_decoder(layer: str) -> nn.Module:
    shape = ResNetFeatureExtractor.LAYER_SHAPES[layer]
    if len(shape) == 3:
        return Decoder2D(shape)
    else:
        return Decoder1D(shape[0])


# ---------------------------------------------------------------------------
# 3. Обучение декодера
# ---------------------------------------------------------------------------
def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    encoder = ResNetFeatureExtractor(layer=args.layer).to(device).eval()
    decoder = build_decoder(args.layer).to(device).train()

    print(f'Encoder layer: {args.layer}, '
          f'embedding shape: {ResNetFeatureExtractor.LAYER_SHAPES[args.layer]}')
    print(f'Decoder params: {sum(p.numel() for p in decoder.parameters())/1e6:.2f}M')

    # Датасет: ImageFolder для ImageNet-подобных директорий.
    # Можно использовать любой каталог: ImageFolder/train, или подмножество.
    transform = T.Compose([
        T.Resize(256), T.CenterCrop(224),
        T.ToTensor(),  # [0, 1]
    ])
    dataset = ImageFolder(args.data_dir, transform=transform)
    print(f'Dataset size: {len(dataset)}')

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = optim.Adam(decoder.parameters(), lr=args.lr, betas=(0.9, 0.999))

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.sample_dir).mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(args.epochs):
        for batch_idx, (imgs, _) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)

            with torch.no_grad():
                phi = encoder(imgs)

            recon = decoder(phi)

            # MSE loss — оригинал статьи. На полносвязных слоях из-за инвариантностей
            # это даст размытость (expected pre-image), что ожидаемо.
            loss = ((recon - imgs) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.log_every == 0:
                print(f'epoch {epoch} | batch {batch_idx}/{len(loader)} | '
                      f'step {step} | mse {loss.item():.5f}')

            # Сохраняем образцы для визуального контроля
            if step % args.sample_every == 0:
                save_samples(imgs[:4], recon[:4],
                             Path(args.sample_dir) / f'samples_step{step:06d}.png',
                             title=f'Layer {args.layer}, step {step}, mse {loss.item():.4f}')

            step += 1

        # Сохраняем чекпоинт после каждой эпохи
        ckpt_path = Path(args.checkpoint_dir) / f'decoder_{args.layer}.pt'
        torch.save({
            'state_dict': decoder.state_dict(),
            'layer': args.layer,
            'epoch': epoch,
        }, ckpt_path)
        print(f'Saved checkpoint: {ckpt_path}')


# ---------------------------------------------------------------------------
# 4. Инференс — восстановление произвольного изображения
# ---------------------------------------------------------------------------
def infer(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = ResNetFeatureExtractor(layer=args.layer).to(device).eval()
    decoder = build_decoder(args.layer).to(device).eval()

    ckpt = torch.load(args.checkpoint, map_location=device)
    decoder.load_state_dict(ckpt['state_dict'])

    img = Image.open(args.image).convert('RGB')
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        phi = encoder(x)
        recon = decoder(phi)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    name = Path(args.image).stem
    out_path = Path(args.out_dir) / f'db_{name}_{args.layer}.png'
    save_samples(x, recon, out_path,
                 title=f'Dosovitskiy & Brox — layer {args.layer}')
    print(f'Saved: {out_path}')


# ---------------------------------------------------------------------------
# Утилиты визуализации
# ---------------------------------------------------------------------------
def save_samples(orig: torch.Tensor, recon: torch.Tensor,
                 path: Path, title: str = ''):
    n = orig.shape[0]
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    if n == 1:
        axes = axes.reshape(2, 1)
    for i in range(n):
        axes[0, i].imshow(orig[i].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy())
        axes[0, i].set_title('original' if i == 0 else '')
        axes[0, i].axis('off')
        axes[1, i].imshow(recon[i].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy())
        axes[1, i].set_title('reconstruction' if i == 0 else '')
        axes[1, i].axis('off')
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'infer'], required=True)
    parser.add_argument('--layer', type=str, default='layer3',
                        choices=ResNetFeatureExtractor.LAYER_NAMES)

    # train
    parser.add_argument('--data_dir', type=str, default=None,
                        help='ImageFolder root (только для train)')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--log_every', type=int, default=20)
    parser.add_argument('--sample_every', type=int, default=200)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--sample_dir', type=str, default='./train_samples')

    # infer
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default='./outputs')

    args = parser.parse_args()

    if args.mode == 'train':
        assert args.data_dir is not None, '--data_dir обязателен для train'
        train(args)
    else:
        assert args.image and args.checkpoint, '--image и --checkpoint обязательны для infer'
        infer(args)


if __name__ == '__main__':
    main()
