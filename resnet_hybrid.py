"""
Гибридный метод инверсии эмбеддингов через ResNet50 (ImageNet).

Параллельная реализация arcface_hybrid.py, но для энкодера общего назначения —
ResNet50, обученной на ImageNet. Используется для сравнения с биометрическим
энкодером ArcFace в рамках курсовой работы.

Реализованы:
1. Mahendran & Vedaldi — оптимизация в пикселях
2. Dosovitskiy & Brox — обучаемый декодер
3. Гибридный метод — комбинация двух подходов (главный вклад)
4. Метрики: PSNR, SSIM, IPS-аналог через cos-similarity эмбеддингов, LIS

Режимы:
    python resnet_hybrid.py --mode train --layer layer3 \\
        --data_dir ./imagenette2-320/train --epochs 5

    python resnet_hybrid.py --mode hybrid --image cat.jpg --layer layer3 \\
        --decoder_ckpt checkpoints/resnet_decoder_layer3.pt

    python resnet_hybrid.py --mode compare --image cat.jpg --layer layer3 \\
        --decoder_ckpt checkpoints/resnet_decoder_layer3.pt
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ============================================================================
# 1. ENCODER — ResNet50 с извлечением эмбеддинга на нужном слое
# ============================================================================
class ResNetFeatureExtractor(nn.Module):
    """Извлекает эмбеддинг с заданного слоя ResNet50."""

    LAYER_NAMES = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4',
                   'avgpool', 'fc']

    LAYER_SHAPES = {
        'conv1':   (64,   56, 56),
        'layer1':  (256,  56, 56),
        'layer2':  (512,  28, 28),
        'layer3':  (1024, 14, 14),
        'layer4':  (2048,  7,  7),
        'avgpool': (2048,),
        'fc':      (1000,),
    }

    def __init__(self, layer='layer3', device='cuda'):
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
        self.to(device)

    def forward(self, x_unnorm):
        mean = IMAGENET_MEAN.to(x_unnorm.device)
        std  = IMAGENET_STD.to(x_unnorm.device)
        x = (x_unnorm - mean) / std
        return self.blocks(x)


# ============================================================================
# 2. DECODER (Dosovitskiy-стиль)
# ============================================================================
class UpConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(self.up(x))))


class Decoder2D(nn.Module):
    """Декодер для пространственных эмбеддингов [C, h, w]."""
    def __init__(self, in_shape, out_size=224):
        super().__init__()
        c, h, w = in_shape
        self.proj = nn.Sequential(
            nn.Conv2d(c, 256, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        n_ups = 0
        cur = h
        while cur < out_size:
            cur *= 2
            n_ups += 1
        chans = [256, 256, 128, 64, 32, 16][:n_ups + 1]
        self.up_blocks = nn.ModuleList([
            UpConvBlock(chans[i], chans[i + 1]) for i in range(n_ups)
        ])
        self.head = nn.Sequential(
            nn.Conv2d(chans[n_ups], 3, 3, 1, 1),
            nn.Sigmoid(),
        )
        self.out_size = out_size

    def forward(self, phi):
        x = self.proj(phi)
        for blk in self.up_blocks:
            x = blk(x)
        x = self.head(x)
        if x.shape[-1] != self.out_size:
            x = nn.functional.interpolate(x, (self.out_size, self.out_size),
                                          mode='bilinear', align_corners=False)
        return x


class Decoder1D(nn.Module):
    """Декодер для векторных эмбеддингов avgpool/fc."""
    def __init__(self, in_dim, out_size=224):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 4096), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4096, 4096),    nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4096, 256 * 7 * 7),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.up_blocks = nn.ModuleList([
            UpConvBlock(256, 256),
            UpConvBlock(256, 128),
            UpConvBlock(128,  64),
            UpConvBlock( 64,  32),
            UpConvBlock( 32,  16),
        ])
        self.head = nn.Sequential(
            nn.Conv2d(16, 3, 3, 1, 1),
            nn.Sigmoid(),
        )
        self.out_size = out_size

    def forward(self, phi):
        x = self.fc(phi).view(-1, 256, 7, 7)
        for blk in self.up_blocks:
            x = blk(x)
        x = self.head(x)
        if x.shape[-1] != self.out_size:
            x = nn.functional.interpolate(x, (self.out_size, self.out_size),
                                          mode='bilinear', align_corners=False)
        return x


def build_decoder(layer):
    shape = ResNetFeatureExtractor.LAYER_SHAPES[layer]
    if len(shape) == 3:
        return Decoder2D(shape)
    return Decoder1D(shape[0])


# ============================================================================
# 3. METRICS
# ============================================================================
def identity_preservation_score(orig, recon, encoder):
    """
    IPS (Information Preservation Score) для ResNet50.

    Семантический аналог биометрического IPS из arcface_hybrid.py.
    Формула идентична:
        IPS(x, x_hat) = cos(Phi(x), Phi(x_hat))
                      = <Phi(x)/||Phi(x)||, Phi(x_hat)/||Phi(x_hat)||>

    Для ResNet50 Phi(x) — эмбеддинг с нужного слоя (по умолчанию layer3,
    тензор [1024,14,14] который сплющивается в вектор перед нормализацией).

    В отличие от ArcFace где IPS измеряет биометрическую идентичность,
    здесь IPS измеряет СЕМАНТИЧЕСКОЕ сходство:
      — высокий IPS означает что реконструкция семантически близка к оригиналу
        (похожий объект, похожая сцена, схожий контекст)
      — IPS > 0.9: объект того же класса, почти идентично
      — IPS > 0.7: похожий объект той же категории
      — IPS > 0.5: семантически схожие объекты
      — IPS < 0.3: разные объекты, инверсия провалилась

    Оба IPS (ResNet и ArcFace) используются для сравнения в курсовой:
    они показывают что ResNet-эмбеддинг восстанавливается с высоким
    семантическим IPS, тогда как ArcFace-эмбеддинг — с высоким
    биометрическим IPS, но низким визуальным качеством.
    """
    with torch.no_grad():
        e_o = encoder(orig).flatten(1)
        e_r = encoder(recon).flatten(1)
        # Явная L2-нормализация (в отличие от ArcFace где она встроена в энкодер)
        e_o = nn.functional.normalize(e_o, p=2, dim=1)
        e_r = nn.functional.normalize(e_r, p=2, dim=1)
        # Скалярное произведение нормализованных векторов = cosine similarity
        return (e_o * e_r).sum(dim=1).mean().item()


def psnr(x, y):
    """
    PSNR (Peak Signal-to-Noise Ratio), дБ.
    PSNR = 10 * log10(1 / MSE)  для изображений в диапазоне [0, 1].
    Выше — лучше. > 30 дБ — хорошее качество, > 40 дБ — отличное.
    """
    mse = ((x - y) ** 2).mean().item()
    return 10 * np.log10(1.0 / mse) if mse > 1e-10 else float('inf')


def ssim_torch(x, y):
    """
    SSIM (Structural Similarity Index) — упрощённая глобальная версия.
    Измеряет структурное сходство через яркость, контраст и структуру.
    SSIM in [-1, 1], выше — лучше. > 0.9 — визуально близко к оригиналу.
    """
    mu_x, mu_y = x.mean(), y.mean()
    var_x = ((x - mu_x) ** 2).mean()
    var_y = ((y - mu_y) ** 2).mean()
    cov = ((x - mu_x) * (y - mu_y)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    return (num / den).item()


def lis_score(ips, ssim_val, psnr_val, time_s, max_time=120.0):
    """
    LIS (Layer/Model Inversion Susceptibility) — единая метрика "опасности"
    модели с точки зрения инверсии. Формула одинакова для ResNet и ArcFace:

        LIS = 0.5 * IPS
            + 0.2 * SSIM
            + 0.15 * min(PSNR / 40, 1)
            + 0.15 * max(0, 1 - t / t_max)

    Компоненты:
      - IPS (вес 0.5) — главный фактор. Для ResNet: семантическое сходство.
        Для ArcFace: биометрическая идентичность.
      - SSIM (вес 0.2) — визуальное структурное сходство.
      - PSNR_norm (вес 0.15) — нормализованное попиксельное качество.
        Делим на 40 дБ (ориентировочный максимум реалистичной реконструкции).
      - speed_norm (вес 0.15) — скорость атаки. Чем быстрее инверсия,
        тем она опаснее. t_max=120 с — граница "медленной" атаки.

    Сумма весов = 1.0, LIS in [0, 1], выше — опаснее.
    """
    psnr_n  = min(max(psnr_val, 0) / 40.0, 1.0)
    speed_n = max(0.0, 1.0 - time_s / max_time)
    return 0.5 * ips + 0.2 * ssim_val + 0.15 * psnr_n + 0.15 * speed_n


# ============================================================================
# 4. РЕГУЛЯРИЗАТОРЫ
# ============================================================================
def alpha_norm(x, alpha=6):
    return x.abs().pow(alpha).sum()


def total_variation(x, beta=2.0):
    dh = x[:, :, 1:, :] - x[:, :, :-1, :]
    dw = x[:, :, :, 1:] - x[:, :, :, :-1]
    return ((dh[:, :, :, :-1] ** 2 + dw[:, :, :-1, :] ** 2 + 1e-8)
            ** (beta / 2)).sum()


# ============================================================================
# 5. ТРИ МЕТОДА ИНВЕРСИИ
# ============================================================================
def invert_mahendran(target, encoder, n_iters=3000, lr=0.01,
                     la=1e-7, ltv=1e-3, verbose=False):
    """Метод 1 — оптимизация в пикселях со старта из шума."""
    with torch.no_grad():
        t_emb = encoder(target)
        norm = (t_emb ** 2).sum().clamp(min=1e-8)

    x = torch.randn_like(target) * 0.5
    x = x.clamp(0, 1).detach().requires_grad_(True)
    opt = optim.Adam([x], lr=lr)

    for i in range(n_iters):
        opt.zero_grad()
        feat = ((encoder(x) - t_emb) ** 2).sum() / norm
        loss = feat + la * alpha_norm(x) + ltv * total_variation(x)
        loss.backward()
        opt.step()
        with torch.no_grad():
            x.clamp_(0, 1)
        if verbose and (i + 1) % 500 == 0:
            print(f'  [Mahendran] {i+1}/{n_iters} feat={feat.item():.5f}')
    return x.detach()


def invert_dosovitskiy(target, encoder, decoder):
    """Метод 2 — один прямой проход декодера."""
    with torch.no_grad():
        return decoder(encoder(target))


def invert_hybrid(target, encoder, decoder, n_iters=400, lr=0.003,
                  la=1e-7, ltv=1e-3, verbose=False):
    """
    Метод 3 — ГИБРИД (главный вклад):
    декодер даёт инициализацию, оптимизация уточняет.
    """
    with torch.no_grad():
        t_emb = encoder(target)
        norm = (t_emb ** 2).sum().clamp(min=1e-8)
        x = decoder(t_emb).clone()

    x = x.detach().requires_grad_(True)
    opt = optim.Adam([x], lr=lr)

    for i in range(n_iters):
        opt.zero_grad()
        feat = ((encoder(x) - t_emb) ** 2).sum() / norm
        loss = feat + la * alpha_norm(x) + ltv * total_variation(x)
        loss.backward()
        opt.step()
        with torch.no_grad():
            x.clamp_(0, 1)
        if verbose and (i + 1) % 200 == 0:
            print(f'  [Hybrid] {i+1}/{n_iters} feat={feat.item():.5f}')
    return x.detach()


# ============================================================================
# 6. ОБУЧЕНИЕ ДЕКОДЕРА
# ============================================================================
def train_decoder(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print(f'Layer: {args.layer}, '
          f'embedding shape: {ResNetFeatureExtractor.LAYER_SHAPES[args.layer]}')

    encoder = ResNetFeatureExtractor(layer=args.layer, device=device)
    encoder.eval()
    decoder = build_decoder(args.layer).to(device).train()
    print(f'Decoder params: {sum(p.numel() for p in decoder.parameters())/1e6:.2f}M')

    # Resume логика
    start_epoch = 0
    ckpt_resume = Path(args.checkpoint_dir) / f'resnet_decoder_{args.layer}.pt'
    if ckpt_resume.exists():
        ckpt = torch.load(ckpt_resume, map_location=device)
        decoder.load_state_dict(ckpt['state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f'Resumed from epoch {start_epoch}')
    else:
        print('Training from scratch')

    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    dataset = ImageFolder(args.data_dir, transform=transform)
    print(f'Dataset size: {len(dataset)}')

    loader = DataLoader(dataset, args.batch_size, shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=torch.cuda.is_available(), drop_last=True)

    opt = optim.Adam(decoder.parameters(), lr=args.lr)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.sample_dir).mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(start_epoch, args.epochs):
        for bi, (imgs, _) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            with torch.no_grad():
                emb = encoder(imgs)
            recon = decoder(emb)
            loss = ((recon - imgs) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % args.log_every == 0:
                print(f'epoch {epoch} | {bi}/{len(loader)} | mse {loss.item():.5f}')
            if step % args.sample_every == 0:
                _save_samples(imgs[:4], recon[:4],
                              Path(args.sample_dir) /
                              f'{args.layer}_step{step:06d}.png',
                              step, args.layer)
            step += 1

        ckpt = Path(args.checkpoint_dir) / f'resnet_decoder_{args.layer}.pt'
        torch.save({'state_dict': decoder.state_dict(),
                    'epoch': epoch, 'layer': args.layer}, ckpt)
        print(f'Saved: {ckpt}')


def _save_samples(orig, recon, path, step, layer):
    n = orig.shape[0]
    fig, ax = plt.subplots(2, n, figsize=(3*n, 6))
    for i in range(n):
        ax[0, i].imshow(orig[i].detach().cpu().permute(1,2,0).clamp(0,1).numpy())
        ax[0, i].axis('off')
        ax[1, i].imshow(recon[i].detach().cpu().permute(1,2,0).clamp(0,1).numpy())
        ax[1, i].axis('off')
    ax[0, 0].set_title('orig'); ax[1, 0].set_title('decoder')
    fig.suptitle(f'Layer {layer}, step {step}')
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# 7. ИНФЕРЕНС И СРАВНЕНИЕ
# ============================================================================
def load_image(path, size=224, device='cuda'):
    img = Image.open(path).convert('RGB')
    transform = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor()])
    return transform(img).unsqueeze(0).to(device)


def to_np(t):
    return t.detach().cpu().clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()


def run_compare(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}\n')

    encoder = ResNetFeatureExtractor(layer=args.layer, device=device)
    encoder.eval()

    decoder = build_decoder(args.layer).to(device).eval()
    ckpt = torch.load(args.decoder_ckpt, map_location=device)
    decoder.load_state_dict(ckpt['state_dict'])

    img = load_image(args.image, size=224, device=device)

    print('[1/3] Mahendran-Vedaldi...')
    t0 = time.time()
    rec_m = invert_mahendran(img, encoder, n_iters=args.iters, lr=0.05,
                             la=1e-7, ltv=1e-3, verbose=True)
    t_m = time.time() - t0

    print('\n[2/3] Dosovitskiy-Brox...')
    t0 = time.time()
    rec_d = invert_dosovitskiy(img, encoder, decoder)
    t_d = time.time() - t0

    print('\n[3/3] Hybrid (ours)...')
    t0 = time.time()
    rec_h = invert_hybrid(img, encoder, decoder,
                          n_iters=args.iters // 4, lr=0.01,
                          la=1e-7, ltv=1e-3, verbose=True)
    t_h = time.time() - t0

    print('\n' + '=' * 70)
    print(f'{"Method":<15} {"sim":>8} {"PSNR":>8} {"SSIM":>8} {"Time":>10} {"LIS":>8}')
    print('-' * 70)
    results = {}
    for name, rec, t in [('Mahendran', rec_m, t_m),
                          ('Dosovitskiy', rec_d, t_d),
                          ('Hybrid', rec_h, t_h)]:
        ips = identity_preservation_score(img, rec, encoder)
        p   = psnr(img, rec)
        s   = ssim_torch(img, rec)
        lis = lis_score(ips, s, p, t)
        results[name] = dict(recon=rec, ips=ips, psnr=p, ssim=s, time=t, lis=lis)
        print(f'{name:<15} {ips:>8.4f} {p:>8.2f} {s:>8.4f} {t:>9.1f}s {lis:>8.4f}')
    print('=' * 70)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.5))
    ax[0].imshow(to_np(img)); ax[0].set_title('Original'); ax[0].axis('off')
    for i, (name, key) in enumerate([('Mahendran', 'Mahendran'),
                                      ('Dosovitskiy', 'Dosovitskiy'),
                                      ('Hybrid (ours)', 'Hybrid')]):
        ax[i+1].imshow(to_np(results[key]['recon'])); ax[i+1].axis('off')
        ax[i+1].set_title(f'{name}\nIPS={results[key]["ips"]:.3f}')
    plt.tight_layout()

    out = Path(args.out_dir) / f'compare_resnet_{args.layer}_{Path(args.image).stem}.png'
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f'\nSaved: {out}')


def run_hybrid_only(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = ResNetFeatureExtractor(layer=args.layer, device=device).eval()
    decoder = build_decoder(args.layer).to(device).eval()
    ckpt = torch.load(args.decoder_ckpt, map_location=device)
    decoder.load_state_dict(ckpt['state_dict'])

    img = load_image(args.image, size=224, device=device)
    rec = invert_hybrid(img, encoder, decoder, args.iters, lr=0.01, verbose=True)
    ips = identity_preservation_score(img, rec, encoder)
    print(f'\nIPS (Information Preservation Score) = {ips:.4f}')

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(to_np(img)); ax[0].set_title('Original'); ax[0].axis('off')
    ax[1].imshow(to_np(rec)); ax[1].set_title(f'Hybrid\nIPS={ips:.3f}'); ax[1].axis('off')
    plt.tight_layout()
    out = Path(args.out_dir) / f'hybrid_resnet_{args.layer}_{Path(args.image).stem}.png'
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f'Saved: {out}')


# ============================================================================
# 8. CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'hybrid', 'compare'],
                        required=True)
    parser.add_argument('--layer', type=str, default='layer3',
                        choices=ResNetFeatureExtractor.LAYER_NAMES)

    parser.add_argument('--image',        type=str, default=None)
    parser.add_argument('--decoder_ckpt', type=str, default=None)
    parser.add_argument('--iters',        type=int, default=2000)
    parser.add_argument('--out_dir',      type=str, default='./outputs')
    parser.add_argument('--data_dir',     type=str, default=None)
    parser.add_argument('--epochs',       type=int, default=5)
    parser.add_argument('--batch_size',   type=int, default=32)
    parser.add_argument('--num_workers',  type=int, default=0)
    parser.add_argument('--lr',           type=float, default=2e-4)
    parser.add_argument('--log_every',    type=int, default=20)
    parser.add_argument('--sample_every', type=int, default=200)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--sample_dir',   type=str, default='./train_samples')

    args = parser.parse_args()

    if args.mode == 'train':
        assert args.data_dir, '--data_dir required'
        train_decoder(args)
    elif args.mode == 'hybrid':
        assert args.image and args.decoder_ckpt, '--image and --decoder_ckpt required'
        run_hybrid_only(args)
    elif args.mode == 'compare':
        assert args.image and args.decoder_ckpt, '--image and --decoder_ckpt required'
        run_compare(args)


if __name__ == '__main__':
    main()