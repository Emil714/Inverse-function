"""
StyleGAN2 + ArcFace: гибридная инверсия с фотореалистичным prior.

Заменяет простой up-conv декодер Dosovitskiy на предобученный StyleGAN2-FFHQ
как генеративный prior. Это принципиальное улучшение:

  Dosovitskiy prior:  MSE-обученный декодер → размытое «среднее лицо»
  StyleGAN2 prior:    генеративная модель  → фотореалистичное лицо

Схема гибридного метода:
  1. ArcFace(x) → 512-d target_emb
  2. MappingNet(target_emb) → w ∈ W+ (18×512) — маппинг в стилевое пространство
  3. StyleGAN2.G(w) → x_init                  — фотореалистичная инициализация
  4. Mahendran_optimize(x_init, target_emb)   → x_final (Hybrid)

Отдельно запускаются:
  - чистая StyleGAN2-инверсия (только шаги 1-3, без оптимизации)
  - гибрид StyleGAN2 + Mahendran (все 4 шага)

Запуск:
    python stylegan_inversion.py --mode compare --image face.jpg

Установка зависимостей:
    pip install torch torchvision Pillow numpy matplotlib
    pip install ninja  # для компиляции кастомных CUDA-операторов StyleGAN2

Веса StyleGAN2-FFHQ скачиваются автоматически при первом запуске (~330 МБ).
Веса ArcFace берутся из ./weights/iresnet50.pth (должны быть уже готовы).
"""

import argparse
import os
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, './stylegan2-ada-pytorch')

import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. ARCFACE ENCODER (копия из arcface_hybrid.py)
# ============================================================================
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2   = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3   = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x
        out = self.bn1(x); out = self.conv1(out); out = self.bn2(out)
        out = self.prelu(out); out = self.conv2(out); out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity

class IResNet(nn.Module):
    fc_scale = 7 * 7
    def __init__(self, block, layers, dropout=0, num_features=512,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super().__init__()
        self.inplanes = 64; self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        self.groups = groups; self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, 3, 1, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.bn2      = nn.BatchNorm2d(512 * block.expansion, eps=1e-05)
        self.dropout  = nn.Dropout(p=dropout, inplace=True)
        self.fc       = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None; previous_dilation = self.dilation
        if dilate: self.dilation *= stride; stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05))
        layers = [block(self.inplanes, planes, stride, downsample,
                        self.groups, self.base_width, previous_dilation)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.prelu(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.layer4(x); x = self.bn2(x); x = torch.flatten(x, 1)
        x = self.dropout(x); x = self.fc(x); x = self.features(x)
        return x

def iresnet50(**kwargs):
    return IResNet(IBasicBlock, [3, 4, 14, 3], **kwargs)

ARCFACE_WEIGHTS = './weights/iresnet50.pth'

class ArcFaceEncoder(nn.Module):
    def __init__(self, weights_path=ARCFACE_WEIGHTS, device='cuda'):
        super().__init__()
        self.model = iresnet50()
        sd = torch.load(weights_path, map_location='cpu')
        self.model.load_state_dict(sd, strict=False)
        self.model = self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.register_buffer('mean',
            torch.tensor([0.5, 0.5, 0.5]).view(1,3,1,1))
        self.register_buffer('std',
            torch.tensor([0.5*256/255]*3).view(1,3,1,1))
        self.input_size = 112
    def forward(self, x):
        if x.shape[-1] != self.input_size:
            x = nn.functional.interpolate(x, (self.input_size, self.input_size),
                                          mode='bilinear', align_corners=False)
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        emb = self.model(x)
        return nn.functional.normalize(emb, p=2, dim=1)


# ============================================================================
# 2. STYLEGAN2 ЗАГРУЗКА И ОБЁРТКА
# ============================================================================
STYLEGAN2_URL  = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-256x256.pkl'
#STYLEGAN2_PATH = './weights/stylegan2-ffhq-256x256.pkl'
STYLEGAN2_PATH = './weights/stylegan2-ffhq-256x256.pkl'

def download_stylegan2():
    """Скачивает веса StyleGAN2-FFHQ-256 если их нет локально."""
    if os.path.isfile(STYLEGAN2_PATH):
        size_mb = os.path.getsize(STYLEGAN2_PATH) / 1e6
        if size_mb > 100:
            print(f'StyleGAN2 weights found: {STYLEGAN2_PATH} ({size_mb:.0f} MB)')
            return
    Path(STYLEGAN2_PATH).parent.mkdir(parents=True, exist_ok=True)
    print('Downloading StyleGAN2-FFHQ-256 weights (~330 MB)...')
    # Пробуем несколько источников
    urls = [
        STYLEGAN2_URL,
        # Альтернатива через HuggingFace зеркало
        'https://huggingface.co/NVlabs/stylegan2/resolve/main/stylegan2-ffhq-256x256.pkl',
    ]
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=120) as r:
                total = int(r.headers.get('Content-Length', 0))
                downloaded = 0
                with open(STYLEGAN2_PATH, 'wb') as f:
                    while True:
                        chunk = r.read(1024*1024)
                        if not chunk: break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            print(f'\r  {downloaded/1e6:.0f}/{total/1e6:.0f} MB',
                                  end='', flush=True)
            print(f'\nSaved: {STYLEGAN2_PATH}')
            return
        except Exception as e:
            print(f'\nFailed from {url[:60]}: {e}')
            if os.path.exists(STYLEGAN2_PATH):
                os.remove(STYLEGAN2_PATH)
    raise RuntimeError(
        '\nCould not download StyleGAN2 weights automatically.\n'
        'Please download manually:\n'
        '  https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan2\n'
        f'and place at: {STYLEGAN2_PATH}\n'
    )

def load_stylegan2(device='cuda'):
    """Загружает StyleGAN2 из .pkl файла через pickle."""
    import pickle, sys
    download_stylegan2()
    with open(STYLEGAN2_PATH, 'rb') as f:
        data = pickle.load(f)
    G = data['G_ema'].to(device).eval()
    for p in G.parameters():
        p.requires_grad_(False)
    print(f'StyleGAN2-FFHQ-256 loaded. '
          f'z_dim={G.z_dim}, w_dim={G.w_dim}, img_res={G.img_resolution}')
    return G


# ============================================================================
# 3. MAPPING NETWORK: ArcFace emb (512) → W+ (18×512)
# ============================================================================
class ArcFaceToW(nn.Module):
    """
    Обучаемая сеть-маппинг из пространства ArcFace-эмбеддингов
    в пространство W+ StyleGAN2.

    Архитектура: несколько FC-слоёв с residual-связью.
    Выход: 18 × 512 (по одному w-вектору на каждый слой стилей StyleGAN2-256).

    Обучается на парах (ArcFace(face_i), StyleGAN_inversion(face_i)).
    Для демонстрации используется оптимизация под конкретное изображение.
    """
    N_STYLES = 14  # StyleGAN2-256 использует 14 стилевых слоёв

    def __init__(self, emb_dim=512, w_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 1024), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1024),    nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1024),    nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, self.N_STYLES * w_dim),
        )
        self.w_dim = w_dim

    def forward(self, emb):
        out = self.net(emb)
        return out.view(-1, self.N_STYLES, self.w_dim)


# ============================================================================
# 4. МЕТРИКИ (идентичны arcface_hybrid.py)
# ============================================================================
def ips(orig, recon, encoder):
    with torch.no_grad():
        e_o = encoder(orig); e_r = encoder(recon)
        return (e_o * e_r).sum(dim=1).mean().item()

def psnr(x, y):
    mse = ((x - y)**2).mean().item()
    return 10 * np.log10(1.0 / mse) if mse > 1e-10 else float('inf')

def ssim_torch(x, y):
    mx, my = x.mean(), y.mean()
    vx = ((x-mx)**2).mean(); vy = ((y-my)**2).mean()
    cov = ((x-mx)*(y-my)).mean()
    c1, c2 = 0.01**2, 0.03**2
    return ((2*mx*my+c1)*(2*cov+c2) / ((mx**2+my**2+c1)*(vx+vy+c2))).item()

def lis_score(ips_val, ssim_val, psnr_val, time_s, max_time=120.0):
    psnr_n  = min(max(psnr_val, 0) / 40.0, 1.0)
    speed_n = max(0.0, 1.0 - time_s / max_time)
    return 0.5*ips_val + 0.2*ssim_val + 0.15*psnr_n + 0.15*speed_n


# ============================================================================
# 5. РЕГУЛЯРИЗАТОРЫ
# ============================================================================
def alpha_norm(x, alpha=6):
    return x.abs().pow(alpha).sum()

def total_variation(x, beta=2.0):
    dh = x[:,:,1:,:] - x[:,:,:-1,:]
    dw = x[:,:,:,1:] - x[:,:,:,:-1]
    return ((dh[:,:,:,:-1]**2 + dw[:,:,:-1,:]**2 + 1e-8)**(beta/2)).sum()


# ============================================================================
# 6. ИНВЕРСИЯ: оптимизация в пространстве W+
# ============================================================================
def stylegan_w_inversion(target_emb, G, encoder, device,
                          n_iters=500, lr=0.01, lambda_reg=0.05,
                          verbose=False):
    """
    Метод 3а — StyleGAN2 prior с W-регуляризацией.

    Оптимизирует w ∈ W+ так, чтобы ArcFace(G(w)) ≈ target_emb.
    Регуляризатор lambda_reg штрафует уход далеко от mean_w —
    это удерживает результат в области правдоподобных лиц.

    Loss:
        L = ||ArcFace(G(w)) - target_emb||² + lambda_reg * ||w - mean_w||²

    Без регуляризатора оптимизатор уходит в "пограничные" области W+
    и генерирует артефакты (распухшие лица, неестественный макияж и т.п.).
    """
    # Считаем mean_w и std_w из 10000 семплов
    with torch.no_grad():
        z_samples = torch.randn(10000, G.z_dim, device=device)
        w_samples = G.mapping(z_samples, None)        # [N, num_ws, 512]
        mean_w = w_samples.mean(0, keepdim=True)      # [1, num_ws, 512]
        std_w  = w_samples.std(0, keepdim=True)       # [1, num_ws, 512]

    w = mean_w.clone().detach().requires_grad_(True)
    opt = optim.Adam([w], lr=lr)

    norm_emb = (target_emb**2).sum().clamp(min=1e-8)

    for i in range(n_iters):
        opt.zero_grad()

        # Генерируем изображение из текущего w
        img_gen = G.synthesis(w, noise_mode='const')
        img_01  = ((img_gen + 1) / 2).clamp(0, 1)

        # ArcFace loss: совпадение эмбеддингов
        emb_gen = encoder(img_01)
        feat_loss = ((emb_gen - target_emb)**2).sum() / norm_emb

        # W-регуляризация: штрафуем уход от mean_w (нормированный на std_w)
        # Это classical W-prior из Image2StyleGAN paper
        w_reg = ((w - mean_w) / (std_w + 1e-8)).pow(2).mean()

        loss = feat_loss + lambda_reg * w_reg
        loss.backward()
        opt.step()

        if verbose and (i+1) % 100 == 0:
            print(f'  [StyleGAN-W] {i+1}/{n_iters} | '
                  f'feat={feat_loss.item():.5f} | w_reg={w_reg.item():.5f}')

    with torch.no_grad():
        img_final = G.synthesis(w, noise_mode='const')
        img_final = ((img_final + 1) / 2).clamp(0, 1)

    return img_final.detach(), w.detach()


def stylegan_pti(target_emb, G, encoder, device,
                 n_iters_w=500, n_iters_pti=300,
                 lr_w=0.01, lr_pti=3e-4,
                 lambda_reg=0.05, verbose=False):
    """
    Pivotal Tuning Inversion (PTI) — Roich et al., 2023.

    Двухэтапный метод:
    1. Pivot search: находим w* через стандартную W+ оптимизацию (как stylegan_w_inversion)
    2. Pivotal Tuning: фиксируем w*, ДООБУЧАЕМ веса генератора G
       чтобы G(w*) точно воспроизводил целевой эмбеддинг.

    Это позволяет StyleGAN восстановить детали (текстура кожи, морщины,
    освещение) которых нет в стандартном W+ пространстве FFHQ.

    Важно: модифицируем G in-place. Возвращаем модифицированный G
    чтобы вызывающий код мог восстановить оригинальные веса при необходимости.
    """
    # === Этап 1: Pivot search ===
    if verbose: print('  [PTI] Stage 1: pivot search in W+...')
    _, w_pivot = stylegan_w_inversion(
        target_emb, G, encoder, device,
        n_iters=n_iters_w, lr=lr_w,
        lambda_reg=lambda_reg, verbose=verbose
    )
    w_pivot = w_pivot.detach()

    # === Этап 2: Pivotal Tuning — дообучаем генератор ===
    if verbose: print('  [PTI] Stage 2: tuning generator weights...')

    # Размораживаем веса G.synthesis
    for p in G.synthesis.parameters():
        p.requires_grad_(True)

    opt_g = optim.Adam(G.synthesis.parameters(), lr=lr_pti)
    norm_emb = (target_emb**2).sum().clamp(min=1e-8)

    for i in range(n_iters_pti):
        opt_g.zero_grad()
        img_gen = G.synthesis(w_pivot, noise_mode='const')
        img_01  = ((img_gen + 1) / 2).clamp(0, 1)
        emb_gen = encoder(img_01)
        feat_loss = ((emb_gen - target_emb)**2).sum() / norm_emb

        feat_loss.backward()
        opt_g.step()

        if verbose and (i+1) % 100 == 0:
            print(f'    [PTI-tune] {i+1}/{n_iters_pti} | feat={feat_loss.item():.5f}')

    # Обратно замораживаем
    for p in G.synthesis.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        img_final = G.synthesis(w_pivot, noise_mode='const')
        img_final = ((img_final + 1) / 2).clamp(0, 1)

    return img_final.detach(), w_pivot


def stylegan_hybrid(target_emb, G, encoder, device,
                    n_iters_w=300, n_iters_pti=200, n_iters_pix=200,
                    lr_w=0.01, lr_pti=3e-4, lr_pix=0.003,
                    lambda_reg=0.05, la=1e-8, ltv=1e-3, verbose=False):
    """
    Метод 3б — StyleGAN2 Hybrid (с PTI):
    Шаг 1: W+ оптимизация → находим pivot w*
    Шаг 2: Pivotal Tuning → дообучаем G чтобы G(w*) точно соответствовал target
    Шаг 3: Mahendran-оптимизация пикселей для финального уточнения

    Это четырёхуровневый гибрид:
      ArcFace prior      (целевой эмбеддинг)
    + StyleGAN W+ prior  (пространство реалистичных лиц)
    + PTI                (адаптация генератора под конкретный таргет)
    + Mahendran          (точное совпадение в пространстве эмбеддингов)
    """
    # Шаги 1+2: PTI
    if verbose: print('  [StyleGAN-Hybrid] Stage 1+2: PTI (W search + G tuning)...')
    x_init, _ = stylegan_pti(
        target_emb, G, encoder, device,
        n_iters_w=n_iters_w, n_iters_pti=n_iters_pti,
        lr_w=lr_w, lr_pti=lr_pti,
        lambda_reg=lambda_reg, verbose=verbose
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Шаг 3: Mahendran pixel-space refinement
    if verbose: print('  [StyleGAN-Hybrid] Stage 3: Mahendran pixel refinement...')
    norm_emb = (target_emb**2).sum().clamp(min=1e-8)
    x = x_init.clone().detach().requires_grad_(True)
    opt = optim.Adam([x], lr=lr_pix)

    for i in range(n_iters_pix):
        opt.zero_grad()
        emb = encoder(x)
        feat = ((emb - target_emb)**2).sum() / norm_emb
        loss = feat + la*alpha_norm(x) + ltv*total_variation(x)
        loss.backward()
        opt.step()
        with torch.no_grad():
            x.clamp_(0, 1)
        if verbose and (i+1) % 100 == 0:
            print(f'    [Mahendran-refine] {i+1}/{n_iters_pix} | feat={feat.item():.5f}')

    return x.detach()


# ============================================================================
# 7. УТИЛИТЫ
# ============================================================================
def load_face(path, size=256, device='cuda'):
    """Загружает изображение лица. StyleGAN2-256 работает с 256×256."""
    img = Image.open(path).convert('RGB').resize((size, size), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(device)

def to_np(t):
    return t.detach().cpu().clamp(0,1).squeeze(0).permute(1,2,0).numpy()

def save_comparison(orig, results_dict, out_path, title=''):
    """Сохраняет сравнительный график для всех методов."""
    n = 1 + len(results_dict)
    fig, ax = plt.subplots(1, n, figsize=(4*n, 4.5))
    ax[0].imshow(to_np(orig)); ax[0].set_title('Original'); ax[0].axis('off')
    for i, (name, data) in enumerate(results_dict.items()):
        ax[i+1].imshow(to_np(data['recon'])); ax[i+1].axis('off')
        ax[i+1].set_title(f'{name}\nIPS={data["ips"]:.3f}')
    if title: fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ============================================================================
# 8. ГЛАВНЫЙ РЕЖИМ: COMPARE — добавляет StyleGAN результаты к готовым
# ============================================================================
def run_compare(args):
    """
    Запускает StyleGAN-W и StyleGAN-Hybrid на одном изображении.
    Результаты можно добавить к таблице из arcface_hybrid.py.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # Очищаем VRAM перед загрузкой моделей
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free = (torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated()) / 1e9
        print(f'VRAM free before loading: {free:.2f} GB')

    # Проверяем наличие весов ArcFace
    if not os.path.isfile(ARCFACE_WEIGHTS):
        raise FileNotFoundError(
            f'ArcFace weights not found: {ARCFACE_WEIGHTS}\n'
            f'Run fix_weights.py first to prepare ArcFace weights.')

    print('Loading ArcFace encoder...')
    encoder = ArcFaceEncoder(device=device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print('Loading StyleGAN2-FFHQ-256...')
    G = load_stylegan2(device=device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free = (torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated()) / 1e9
        print(f'VRAM free after loading all models: {free:.2f} GB')

    img = load_face(args.image, size=256, device=device)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        target_emb = encoder(img)

    results = {}

    # --- StyleGAN W+ оптимизация (без pixel refinement) ---
    print('\n[1/2] StyleGAN2 W+ inversion...')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    t0 = time.time()
    rec_w, _ = stylegan_w_inversion(
        target_emb, G, encoder, device,
        n_iters=args.iters_w, lr=0.01, verbose=True
    )
    t_w = time.time() - t0
    results['StyleGAN-W'] = {
        'recon': rec_w,
        'ips':   ips(img, rec_w, encoder),
        'psnr':  psnr(img, rec_w),
        'ssim':  ssim_torch(img, rec_w),
        'time':  t_w,
    }
    results['StyleGAN-W']['lis'] = lis_score(
        results['StyleGAN-W']['ips'],
        results['StyleGAN-W']['ssim'],
        results['StyleGAN-W']['psnr'],
        t_w
    )

    # --- StyleGAN Hybrid (W+ → pixel refinement) ---
    print('\n[2/2] StyleGAN2 Hybrid (W+ + Mahendran)...')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    t0 = time.time()
    rec_h = stylegan_hybrid(
        target_emb, G, encoder, device,
        n_iters_w=args.iters_w,
        n_iters_pti=getattr(args, 'iters_pti', 200),
        n_iters_pix=args.iters_pix,
        verbose=True
    )
    t_h = time.time() - t0
    results['StyleGAN-Hybrid'] = {
        'recon': rec_h,
        'ips':   ips(img, rec_h, encoder),
        'psnr':  psnr(img, rec_h),
        'ssim':  ssim_torch(img, rec_h),
        'time':  t_h,
    }
    results['StyleGAN-Hybrid']['lis'] = lis_score(
        results['StyleGAN-Hybrid']['ips'],
        results['StyleGAN-Hybrid']['ssim'],
        results['StyleGAN-Hybrid']['psnr'],
        t_h
    )

    # --- Таблица метрик ---
    print('\n' + '='*70)
    print(f'{"Method":<20} {"IPS":>8} {"PSNR":>8} {"SSIM":>8} '
          f'{"Time":>10} {"LIS":>8}')
    print('-'*70)
    for name, d in results.items():
        print(f'{name:<20} {d["ips"]:>8.4f} {d["psnr"]:>8.2f} '
              f'{d["ssim"]:>8.4f} {d["time"]:>9.1f}s {d["lis"]:>8.4f}')
    print('='*70)
    print('\nAdd these rows to your full comparison table from arcface_hybrid.py')

    # --- Визуализация ---
    name = Path(args.image).stem
    out_path = Path(args.out_dir) / f'stylegan_compare_{name}.png'
    save_comparison(img, results, out_path,
                    title='StyleGAN2 prior: W+ inversion vs Hybrid')


# ============================================================================
# 9. CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='StyleGAN2 + ArcFace inversion')
    parser.add_argument('--mode', choices=['compare'], default='compare')
    parser.add_argument('--image',     type=str, required=True)
    parser.add_argument('--iters_w',   type=int, default=500,
                        help='итераций оптимизации в W+')
    parser.add_argument('--iters_pti', type=int, default=200,
                        help='итераций Pivotal Tuning (дообучения генератора)')
    parser.add_argument('--iters_pix', type=int, default=300,
                        help='итераций pixel refinement в гибриде')
    parser.add_argument('--out_dir',   type=str, default='./outputs')
    args = parser.parse_args()
    run_compare(args)

if __name__ == '__main__':
    main()