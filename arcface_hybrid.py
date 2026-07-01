"""
Гибридный метод инверсии биометрических эмбеддингов через настоящий ArcFace.

Использует оригинальные веса iresnet50 ArcFace (обученные deepinsight/insightface
на датасете MS1M-ArcFace), портированные в PyTorch проектом nizhib/pytorch-insightface.

Архитектура iresnet встроена прямо в этот файл — никакой зависимости от
устаревших пакетов не требуется. Веса скачиваются автоматически при первом запуске
(или загружаются вручную в ./weights/iresnet50.pth).

Реализованные компоненты:
1. Mahendran & Vedaldi — оптимизация в пикселях
2. Dosovitskiy & Brox — обучаемый декодер
3. Гибридный метод (главный научный вклад) — комбинация двух подходов
4. IPS (Identity Preservation Score) — метрика угрозы приватности
5. LIS (Layer Inversion Susceptibility) — рейтинг "опасности" моделей

Режимы:
    python arcface_hybrid.py --mode train --data_dir ./celeba/img_align_celeba
    python arcface_hybrid.py --mode hybrid --image face.jpg --decoder_ckpt checkpoints/arcface_decoder.pt
    python arcface_hybrid.py --mode compare --image face.jpg --decoder_ckpt checkpoints/arcface_decoder.pt
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# 1. ARCHITECTURE: iresnet50 (оригинальная архитектура ArcFace)
# Источник: deepinsight/insightface, портировано в PyTorch nizhib/pytorch-insightface
# ============================================================================
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class IBasicBlock(nn.Module):
    """Improved Basic Block из ArcFace: BN, Conv, BN, PReLU, Conv, BN."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('IBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in IBasicBlock')
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IResNet(nn.Module):
    """ArcFace IResNet backbone. Принимает 112x112, возвращает 512-d embedding."""
    fc_scale = 7 * 7

    def __init__(self, block, layers, dropout=0, num_features=512,
                 zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None):
        super().__init__()
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)

        self.layer1 = self._make_layer(block, 64,  layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )
        layers = [block(self.inplanes, planes, stride, downsample,
                        self.groups, self.base_width, previous_dilation)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        return x


def _iresnet(layers, **kwargs):
    return IResNet(IBasicBlock, layers, **kwargs)


def iresnet50(**kwargs):
    """ArcFace iresnet50 (главная модель)."""
    return _iresnet([3, 4, 14, 3], **kwargs)


# ============================================================================
# 2. ENCODER WRAPPER — загрузка весов и пред-обработка
# ============================================================================
ARCFACE_WEIGHTS_URL = 'https://sota.nizhib.ai/pytorch-insightface/iresnet50-7f187506.pth'
ARCFACE_LOCAL_PATH = './weights/iresnet50.pth'


def _download_arcface_weights(url: str, dst: str):
    """Скачивает веса ArcFace, если они не лежат локально."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    print(f'Downloading ArcFace weights from {url}...')
    import urllib.request
    try:
        urllib.request.urlretrieve(url, dst)
        print(f'Saved to {dst}')
    except Exception as e:
        raise RuntimeError(
            f'\nFailed to download ArcFace weights: {e}\n'
            f'Please download manually from:\n'
            f'  {url}\n'
            f'and place at:\n'
            f'  {dst}\n'
        )


class ArcFaceEncoder(nn.Module):
    """
    Обёртка над ArcFace iresnet50.
    - вход: [B, 3, H, W] в [0, 1]
    - выход: [B, 512] L2-нормализованных эмбеддингов
    """
    def __init__(self, weights_path=ARCFACE_LOCAL_PATH, device='cuda',
                 download_url=ARCFACE_WEIGHTS_URL):
        super().__init__()
        self.model = iresnet50()

        if not os.path.isfile(weights_path):
            _download_arcface_weights(download_url, weights_path)

        state_dict = torch.load(weights_path, map_location='cpu')
        self.model.load_state_dict(state_dict)

        #self.register_buffer('mean', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        #self.register_buffer('std', torch.tensor([0.5 * 256 / 255] * 3).view(1, 3, 1, 1))
        #self.input_size = 112

        self.model = self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Нормализация ArcFace: mean=0.5, std≈0.5 (точнее: 128/255)
        self.register_buffer('mean', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.5 * 256/255] * 3).view(1, 3, 1, 1))
        self.input_size = 112

    def forward(self, x):
        # x в [0, 1]: приводим к 112x112 и нормализуем
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = nn.functional.interpolate(x, size=(self.input_size, self.input_size),
                                          mode='bilinear', align_corners=False)
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        emb = self.model(x)                                # [B, 512]
        emb = nn.functional.normalize(emb, p=2, dim=1)
        return emb


# ============================================================================
# 3. DECODER (для метода Dosovitskiy-Brox)
# 512-мерный вектор -> изображение лица 112x112 (формат ArcFace)
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


class FaceDecoder(nn.Module):
    """
    Архитектура из Dosovitskiy & Brox для векторных эмбеддингов:
    FC слои -> reshape -> цепочка up-conv блоков.
    Выход 112x112 (формат ArcFace).
    """
    def __init__(self, emb_dim=512, out_size=112):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, 4096), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4096, 4096),    nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4096, 256 * 7 * 7),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 7 -> 14 -> 28 -> 56 -> 112
        self.up_blocks = nn.ModuleList([
            UpConvBlock(256, 256),
            UpConvBlock(256, 128),
            UpConvBlock(128,  64),
            UpConvBlock( 64,  32),
        ])
        self.head = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Sigmoid(),
        )
        self.out_size = out_size

    def forward(self, emb):
        x = self.fc(emb)
        x = x.view(-1, 256, 7, 7)
        for blk in self.up_blocks:
            x = blk(x)
        x = self.head(x)
        if x.shape[-1] != self.out_size:
            x = nn.functional.interpolate(x, size=(self.out_size, self.out_size),
                                          mode='bilinear', align_corners=False)
        return x


# ============================================================================
# 4. METRICS
# ============================================================================
def identity_preservation_score(orig, recon, encoder):
    """
    IPS (Identity Preservation Score) — главный научный вклад.

    Косинусное сходство между эмбеддингами оригинала и реконструкции.
    Использует ту же процедуру верификации, что и production face recognition.

    IPS > 0.5  => биометрическая система идентифицирует как одного человека
    IPS > 0.7  => высокая уверенность в совпадении
    IPS < 0.3  => реконструкция не угрожает идентичности

    В отличие от PSNR/SSIM измеряет именно угрозу приватности.
    """
    with torch.no_grad():
        emb_o = encoder(orig)
        emb_r = encoder(recon)
        return (emb_o * emb_r).sum(dim=1).mean().item()


def psnr(x, y):
    mse = ((x - y) ** 2).mean().item()
    if mse < 1e-10:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


def ssim_torch(x, y):
    """Упрощённый глобальный SSIM (для приближённой оценки)."""
    mu_x = x.mean(); mu_y = y.mean()
    var_x = ((x - mu_x) ** 2).mean()
    var_y = ((y - mu_y) ** 2).mean()
    cov   = ((x - mu_x) * (y - mu_y)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    return (num / den).item()


def lis_score(ips, ssim_val, psnr_val, time_seconds, max_time=120.0):
    """
    LIS (Layer Inversion Susceptibility) — композитная метрика
    "опасности" модели/слоя с точки зрения приватности.
    """
    psnr_norm = min(max(psnr_val, 0) / 40.0, 1.0)
    speed_norm = max(0.0, 1.0 - time_seconds / max_time)
    return 0.5 * ips + 0.2 * ssim_val + 0.15 * psnr_norm + 0.15 * speed_norm


# ============================================================================
# 5. РЕГУЛЯРИЗАТОРЫ для оптимизации
# ============================================================================
def alpha_norm(x, alpha=6):
    return x.abs().pow(alpha).sum()


def total_variation(x, beta=2.0):
    dh = x[:, :, 1:, :] - x[:, :, :-1, :]
    dw = x[:, :, :, 1:] - x[:, :, :, :-1]
    dh = dh[:, :, :, :-1]
    dw = dw[:, :, :-1, :]
    return ((dh ** 2 + dw ** 2 + 1e-8) ** (beta / 2)).sum()


# ============================================================================
# 6. ТРИ МЕТОДА ИНВЕРСИИ
# ============================================================================
def invert_mahendran(target, encoder, n_iters=3000, lr=0.01,
                     lambda_alpha=1e-8, lambda_tv=1e-4, verbose=False):
    """Метод 1: оптимизация в пикселях со старта из шума."""
    with torch.no_grad():
        target_emb = encoder(target)
        norm_emb = (target_emb ** 2).sum().clamp(min=1e-8)

    x = torch.randn_like(target) * 0.5
    x = x.clamp(0, 1).clone().detach().requires_grad_(True)
    opt = optim.Adam([x], lr=lr)

    for i in range(n_iters):
        opt.zero_grad()
        emb = encoder(x)
        feat = ((emb - target_emb) ** 2).sum() / norm_emb
        loss = feat + lambda_alpha * alpha_norm(x) + lambda_tv * total_variation(x)
        loss.backward()
        opt.step()
        with torch.no_grad():
            x.clamp_(0, 1)
        if verbose and (i + 1) % 500 == 0:
            print(f'  [Mahendran] iter {i+1}/{n_iters} | feat_loss {feat.item():.5f}')

    return x.detach()


def invert_dosovitskiy(target, encoder, decoder):
    """Метод 2: один прямой проход через обученный декодер."""
    with torch.no_grad():
        target_emb = encoder(target)
        recon = decoder(target_emb)
    return recon


def invert_hybrid(target, encoder, decoder, n_iters=800, lr=0.005,
                  lambda_alpha=1e-8, lambda_tv=1e-4, verbose=False):
    """
    Метод 3 — ГИБРИД (главный научный вклад):
    1) Декодер даёт правдоподобное "среднее лицо" как инициализацию
    2) Оптимизация уточняет до точного совпадения эмбеддинга
    """
    with torch.no_grad():
        target_emb = encoder(target)
        norm_emb = (target_emb ** 2).sum().clamp(min=1e-8)
        x_init = decoder(target_emb)              # быстрая грубая реконструкция

    x = x_init.clone().detach().requires_grad_(True)
    opt = optim.Adam([x], lr=lr)

    for i in range(n_iters):
        opt.zero_grad()
        emb = encoder(x)
        feat = ((emb - target_emb) ** 2).sum() / norm_emb
        loss = feat + lambda_alpha * alpha_norm(x) + lambda_tv * total_variation(x)
        loss.backward()
        opt.step()
        with torch.no_grad():
            x.clamp_(0, 1)
        if verbose and (i + 1) % 200 == 0:
            print(f'  [Hybrid] iter {i+1}/{n_iters} | feat_loss {feat.item():.5f}')

    return x.detach()


# ============================================================================
# 7. ДАТАСЕТ — CelebA
# ============================================================================
class CelebADataset(Dataset):
    """Все .jpg из директории img_align_celeba/."""
    def __init__(self, root, size=112):
        self.root = Path(root)
        self.files = sorted(self.root.glob('*.jpg'))
        if len(self.files) == 0:
            raise RuntimeError(f'No .jpg files found in {root}')
        self.transform = T.Compose([
            T.Resize(size + 16),
            T.CenterCrop(size),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        return self.transform(img)


# ============================================================================
# 8. ОБУЧЕНИЕ ДЕКОДЕРА (Dosovitskiy-стиль)
# ============================================================================
def train_decoder(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    encoder = ArcFaceEncoder(device=device)
    decoder = FaceDecoder(emb_dim=512, out_size=112).to(device).train()

    # Загружаем чекпоинт если есть
    start_epoch = 0
    ckpt_resume = Path(args.checkpoint_dir) / 'arcface_decoder.pt'
    if ckpt_resume.exists():
        ckpt = torch.load(ckpt_resume, map_location=device)
        decoder.load_state_dict(ckpt['state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f'Resumed from epoch {start_epoch}')
    else:
        print('Training from scratch')

    print(f'Decoder params: {sum(p.numel() for p in decoder.parameters())/1e6:.2f}M')

    dataset = CelebADataset(args.data_dir, size=112)
    print(f'Dataset size: {len(dataset)}')

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=True,
    )

    opt = optim.Adam(decoder.parameters(), lr=args.lr, betas=(0.9, 0.999))

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.sample_dir).mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(start_epoch, args.epochs):
        for batch_idx, imgs in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)

            with torch.no_grad():
                emb = encoder(imgs)

            recon = decoder(emb)

            # MSE по пикселям
            pixel_loss = ((recon - imgs) ** 2).mean()

            # Дополнительно: эмбеддинг реконструкции должен совпадать с целью
            recon_emb = encoder(recon)
            emb_loss = (1.0 - (recon_emb * emb).sum(dim=1)).mean()

            loss = pixel_loss + 0.1 * emb_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % args.log_every == 0:
                print(f'epoch {epoch} | batch {batch_idx}/{len(loader)} | '
                      f'pix {pixel_loss.item():.4f} | emb {emb_loss.item():.4f}')

            if step % args.sample_every == 0:
                _save_decoder_samples(imgs[:4], recon[:4],
                                      Path(args.sample_dir) / f'step{step:06d}.png',
                                      step)
            step += 1

        ckpt_path = Path(args.checkpoint_dir) / 'arcface_decoder.pt'
        torch.save({'state_dict': decoder.state_dict(), 'epoch': epoch}, ckpt_path)
        print(f'Saved: {ckpt_path}')


def _save_decoder_samples(orig, recon, path, step):
    n = orig.shape[0]
    fig, ax = plt.subplots(2, n, figsize=(3 * n, 6))
    for i in range(n):
        ax[0, i].imshow(orig[i].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy())
        ax[0, i].axis('off')
        ax[1, i].imshow(recon[i].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy())
        ax[1, i].axis('off')
    ax[0, 0].set_title('orig'); ax[1, 0].set_title('decoder')
    fig.suptitle(f'Decoder training, step {step}')
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# 9. ИНФЕРЕНС И СРАВНЕНИЕ
# ============================================================================
def load_face(path, size=112, device='cuda'):
    img = Image.open(path).convert('RGB').resize((size, size), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def to_np(t):
    return t.detach().cpu().clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()


def run_compare(args):
    """Запускает все три метода и сравнивает по всем метрикам."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}\n')

    encoder = ArcFaceEncoder(device=device)
    decoder = FaceDecoder(emb_dim=512, out_size=112).to(device).eval()
    ckpt = torch.load(args.decoder_ckpt, map_location=device)
    decoder.load_state_dict(ckpt['state_dict'])

    img = load_face(args.image, size=112, device=device)

    # --- 1. Mahendran ---
    print('[1/3] Mahendran-Vedaldi (optimization from noise)...')
    t0 = time.time()
    #rec_m = invert_mahendran(img, encoder, n_iters=args.iters, lr=0.01, verbose=True)
    rec_m = invert_mahendran(img, encoder, n_iters=args.iters, lr=0.01,
                             lambda_alpha=1e-6, lambda_tv=1e-2, verbose=True)
    t_m = time.time() - t0

    # --- 2. Dosovitskiy ---
    print('\n[2/3] Dosovitskiy-Brox (decoder forward pass)...')
    t0 = time.time()
    rec_d = invert_dosovitskiy(img, encoder, decoder)
    t_d = time.time() - t0

    # --- 3. Hybrid ---
    print('\n[3/3] Hybrid (decoder init + optimization)...')
    t0 = time.time()
    #rec_h = invert_hybrid(img, encoder, decoder,
    #                      n_iters=args.iters // 4, lr=0.005, verbose=True)
    rec_h = invert_hybrid(img, encoder, decoder,
                          n_iters=200, lr=0.001,
                          lambda_alpha=1e-6, lambda_tv=5e-2, verbose=True)
    t_h = time.time() - t0

    # --- Метрики ---
    print('\n' + '=' * 70)
    print(f'{"Method":<15} {"IPS":>8} {"PSNR":>8} {"SSIM":>8} {"Time":>10} {"LIS":>8}')
    print('-' * 70)

    results = {}
    for name, rec, t in [('Mahendran', rec_m, t_m),
                          ('Dosovitskiy', rec_d, t_d),
                          ('Hybrid', rec_h, t_h)]:
        ips = identity_preservation_score(img, rec, encoder)
        p = psnr(img, rec)
        s = ssim_torch(img, rec)
        lis = lis_score(ips, s, p, t)
        results[name] = dict(recon=rec, ips=ips, psnr=p, ssim=s, time=t, lis=lis)
        print(f'{name:<15} {ips:>8.4f} {p:>8.2f} {s:>8.4f} {t:>9.1f}s {lis:>8.4f}')
    print('=' * 70)

    # --- Визуализация ---
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.5))
    ax[0].imshow(to_np(img));   ax[0].set_title('Original');     ax[0].axis('off')
    ax[1].imshow(to_np(rec_m)); ax[1].axis('off')
    ax[1].set_title(f'Mahendran\nIPS={results["Mahendran"]["ips"]:.3f}')
    ax[2].imshow(to_np(rec_d)); ax[2].axis('off')
    ax[2].set_title(f'Dosovitskiy\nIPS={results["Dosovitskiy"]["ips"]:.3f}')
    ax[3].imshow(to_np(rec_h)); ax[3].axis('off')
    ax[3].set_title(f'Hybrid (ours)\nIPS={results["Hybrid"]["ips"]:.3f}')
    plt.tight_layout()

    name = Path(args.image).stem
    out_path = Path(args.out_dir) / f'compare_{name}.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f'\nSaved: {out_path}')


def run_hybrid_only(args):
    """Инвертирует одно лицо гибридным методом."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = ArcFaceEncoder(device=device)
    decoder = FaceDecoder(emb_dim=512, out_size=112).to(device).eval()
    ckpt = torch.load(args.decoder_ckpt, map_location=device)
    decoder.load_state_dict(ckpt['state_dict'])

    img = load_face(args.image, size=112, device=device)

    print('Hybrid inversion in progress...')
    rec = invert_hybrid(img, encoder, decoder,
                        n_iters=args.iters, lr=0.005, verbose=True)

    ips = identity_preservation_score(img, rec, encoder)
    print(f'\nIPS = {ips:.4f}')
    if ips > 0.5:
        print('=> THREAT: production face recognition would identify the person')
    else:
        print('=> SAFE: reconstruction does not preserve identity')

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(to_np(img)); ax[0].set_title('Original'); ax[0].axis('off')
    ax[1].imshow(to_np(rec)); ax[1].axis('off')
    ax[1].set_title(f'Hybrid reconstruction\nIPS = {ips:.3f}')
    plt.tight_layout()
    name = Path(args.image).stem
    out_path = Path(args.out_dir) / f'hybrid_{name}.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f'Saved: {out_path}')


# ============================================================================
# 10. CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'hybrid', 'compare'], required=True)

    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--decoder_ckpt', type=str, default=None)
    parser.add_argument('--iters', type=int, default=3000)
    parser.add_argument('--out_dir', type=str, default='./outputs')

    parser.add_argument('--data_dir', type=str, default=None,
                        help='путь к img_align_celeba/')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--log_every', type=int, default=20)
    parser.add_argument('--sample_every', type=int, default=200)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--sample_dir', type=str, default='./train_samples')

    args = parser.parse_args()

    if args.mode == 'train':
        assert args.data_dir, '--data_dir обязателен для train'
        train_decoder(args)
    elif args.mode == 'hybrid':
        assert args.image and args.decoder_ckpt, '--image и --decoder_ckpt обязательны'
        run_hybrid_only(args)
    elif args.mode == 'compare':
        assert args.image and args.decoder_ckpt, '--image и --decoder_ckpt обязательны'
        run_compare(args)


if __name__ == '__main__':
    main()
