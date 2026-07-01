"""
Super-Hybrid: StyleGAN2 + Stable Diffusion + Mahendran для инверсии ArcFace.

Идея: объединить сильные стороны двух генеративных prior-ов.
  - StyleGAN2 — специализированный face prior (обучен на FFHQ только лица)
  - Stable Diffusion — general-purpose prior (детали, текстуры, освещение)
  - Mahendran — точное совпадение в пространстве эмбеддингов

Порядок выбран принципиально: StyleGAN → SD → Mahendran.
Обоснование: StyleGAN сразу даёт правдоподобное лицо, SD используется как
"refiner" через img2img с малым strength, что сохраняет идентичность лица от
StyleGAN, но добавляет фотореалистичные детали (текстура кожи, освещение).
Это аналогично подходу SDXL Refiner.

Альтернативный порядок (SD → StyleGAN) не сработает, потому что SD без
img2img даёт шумные реконструкции (мы это видели: DDIM IPS=0.147),
а StyleGAN-инверсия из шума не сможет дать осмысленный результат.

Архитектура супер-гибрида:

    ArcFace(x) → target_emb (512-d)
         │
         ▼
    ┌─────────────────────────────────┐
    │ Stage 1: StyleGAN2 W+ inversion │  ← фотореалистичная инициализация
    │  (опционально с PTI)            │
    └─────────────────────────────────┘
         │ x_stylegan (256x256)
         ▼
    ┌─────────────────────────────────┐
    │ Stage 2: SD img2img refinement  │  ← добавляем детали через диффузию
    │  (low strength = 0.3-0.4)       │
    └─────────────────────────────────┘
         │ x_diffusion (512x512)
         ▼
    ┌─────────────────────────────────┐
    │ Stage 3: Mahendran refinement   │  ← точное совпадение эмбеддинга
    │  (pixel-space optimization)     │
    └─────────────────────────────────┘
         │ x_final
         ▼
       Результат

Запуск:
    python super_hybrid_inversion.py --image face.jpg

Требования:
    ./weights/iresnet50.pth                    ← веса ArcFace
    ./weights/stylegan2-ffhq-256x256.pkl       ← веса StyleGAN2
    ./weights/stable-diffusion-v1-5/           ← веса SD
    ./stylegan2-ada-pytorch/                   ← репозиторий StyleGAN2
"""

import argparse
import os
import sys
import time
import pickle
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Подавляем многочисленные предупреждения StyleGAN/diffusers
warnings.filterwarnings('ignore')

# StyleGAN2 требует свой репозиторий для pickle.load
sys.path.insert(0, './stylegan2-ada-pytorch')


# ============================================================================
# 1. ARCFACE ENCODER
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
        out = self.bn1(x); out = self.conv1(out); out = self.bn2(out)
        out = self.prelu(out); out = self.conv2(out); out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity

class IResNet(nn.Module):
    fc_scale = 7 * 7
    def __init__(self, block, layers, dropout=0, num_features=512):
        super().__init__()
        self.inplanes = 64
        self.dilation = 1
        self.conv1 = nn.Conv2d(3, self.inplanes, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05))
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.prelu(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.bn2(x); x = torch.flatten(x, 1)
        x = self.dropout(x); x = self.fc(x); x = self.features(x)
        return x

def iresnet50(**kwargs):
    return IResNet(IBasicBlock, [3, 4, 14, 3], **kwargs)


ARCFACE_WEIGHTS = './weights/iresnet50.pth'
STYLEGAN2_PATH  = './weights/stylegan2-ffhq-256x256.pkl'
SD_MODEL_ID     = './weights/stable-diffusion-v1-5'


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
            torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer('std',
            torch.tensor([0.5 * 256 / 255] * 3).view(1, 3, 1, 1))
        self.input_size = 112
    def forward(self, x):
        if x.shape[-1] != self.input_size:
            x = nn.functional.interpolate(x,
                (self.input_size, self.input_size),
                mode='bilinear', align_corners=False)
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        emb = self.model(x)
        return nn.functional.normalize(emb, p=2, dim=1)


# ============================================================================
# 2. ЗАГРУЗКА StyleGAN2
# ============================================================================
def load_stylegan2(device='cuda'):
    """Загружает StyleGAN2 из локального pkl-файла."""
    if not os.path.isfile(STYLEGAN2_PATH):
        raise FileNotFoundError(f'StyleGAN2 weights not found: {STYLEGAN2_PATH}')

    print(f'Loading StyleGAN2 from {STYLEGAN2_PATH}...')
    with open(STYLEGAN2_PATH, 'rb') as f:
        data = pickle.load(f)
    G = data['G_ema'].to(device).eval()
    for p in G.parameters():
        p.requires_grad_(False)
    print(f'  StyleGAN2 loaded: z_dim={G.z_dim}, w_dim={G.w_dim}, '
          f'res={G.img_resolution}')
    return G


# ============================================================================
# 3. ЗАГРУЗКА Stable Diffusion (через img2img pipeline)
# ============================================================================
def load_sd_img2img(device='cuda'):
    """
    Загружает SD как img2img pipeline (не text2img).
    img2img позволяет использовать существующее изображение как conditioning.
    """
    from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler

    print(f'Loading Stable Diffusion img2img pipeline ({SD_MODEL_ID})...')
    dtype = torch.float16 if device == 'cuda' else torch.float32
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        SD_MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        feature_extractor=None,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    print('  SD img2img loaded.')
    return pipe


# ============================================================================
# 4. РЕГУЛЯРИЗАТОРЫ для Mahendran
# ============================================================================
def alpha_norm(x, alpha=6):
    return x.abs().pow(alpha).sum()

def total_variation(x, beta=2.0):
    dh = x[:, :, 1:, :] - x[:, :, :-1, :]
    dw = x[:, :, :, 1:] - x[:, :, :, :-1]
    return ((dh[:, :, :, :-1] ** 2 + dw[:, :, :-1, :] ** 2 + 1e-8)
            ** (beta / 2)).sum()


# ============================================================================
# 5. STAGE 1: StyleGAN2 W+ инверсия
# ============================================================================
def stylegan_w_inversion(target_emb, G, encoder, device,
                         n_iters=500, lr=0.01, lambda_reg=0.05,
                         verbose=False):
    """
    Оптимизация в W+ пространстве StyleGAN2.

    Возвращает: (image, w_pivot)
      image — фотореалистичное лицо [1, 3, 256, 256] в [0, 1]
      w_pivot — найденный вектор W+ для использования в PTI
    """
    # Считаем mean_w и std_w для регуляризации
    with torch.no_grad():
        z_samples = torch.randn(10000, G.z_dim, device=device)
        w_samples = G.mapping(z_samples, None)
        mean_w = w_samples.mean(0, keepdim=True)
        std_w  = w_samples.std(0, keepdim=True)

    w = mean_w.clone().detach().requires_grad_(True)
    opt = optim.Adam([w], lr=lr)
    norm_emb = (target_emb ** 2).sum().clamp(min=1e-8)

    for i in range(n_iters):
        opt.zero_grad()
        img_gen = G.synthesis(w, noise_mode='const')
        img_01  = ((img_gen + 1) / 2).clamp(0, 1)
        emb_gen = encoder(img_01)
        feat_loss = ((emb_gen - target_emb) ** 2).sum() / norm_emb
        w_reg = ((w - mean_w) / (std_w + 1e-8)).pow(2).mean()
        loss = feat_loss + lambda_reg * w_reg
        loss.backward()
        opt.step()
        if verbose and (i + 1) % 100 == 0:
            print(f'    [StyleGAN-W] {i+1}/{n_iters} '
                  f'feat={feat_loss.item():.5f} reg={w_reg.item():.5f}')

    with torch.no_grad():
        img_final = G.synthesis(w, noise_mode='const')
        img_final = ((img_final + 1) / 2).clamp(0, 1)

    return img_final.detach(), w.detach()


# ============================================================================
# 6. STAGE 2: SD img2img как refiner
# ============================================================================
def sd_refine(image_stylegan, pipe, target_emb, encoder,
              strength=0.35, guidance_scale=5.0,
              num_steps=30, prompt='a high quality photo of a face',
              device='cuda', verbose=False):
    """
    Применяет SD img2img как refiner поверх StyleGAN результата.

    Принципиально: используем низкий strength (0.3-0.4) — это означает что
    SD добавляет только ~35% шума к изображению и проходит обратный диффузионный
    процесс. Идентичность лица сохраняется от StyleGAN, SD добавляет детали.

    Args:
        image_stylegan: [1, 3, H, W] в [0,1] — результат StyleGAN
        strength: 0 = идентичность (без изменений), 1 = полная регенерация
        guidance_scale: CFG, 5.0 — мягкий guidance (7.5 — стандарт text2img)
    """
    from PIL import Image

    # Конвертируем тензор → PIL для pipe
    img_np = (image_stylegan.detach().cpu()
              .squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)

    # SD работает с 512x512
    if img_pil.size != (512, 512):
        img_pil = img_pil.resize((512, 512), Image.BILINEAR)

    if verbose:
        print(f'    [SD-Refine] running img2img: strength={strength}, '
              f'steps={num_steps}, guidance={guidance_scale}')

    # Запускаем img2img
    with torch.no_grad():
        out = pipe(
            prompt=prompt,
            image=img_pil,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            output_type='pt',  # возвращает torch tensor
        )

    # output_type='pt' возвращает [B, 3, 512, 512] в [0, 1]
    refined = out.images.to(device).float()

    return refined


# ============================================================================
# 7. STAGE 3: Mahendran pixel refinement
# ============================================================================
def mahendran_refine(x_init, target_emb, encoder,
                     n_iters=200, lr=0.003,
                     la=1e-8, ltv=1e-3, verbose=False):
    """
    Финальное уточнение в пространстве пикселей.

    Малый learning rate и небольшое число итераций — не хотим уйти далеко
    от хорошей инициализации в adversarial направление.
    """
    norm_emb = (target_emb ** 2).sum().clamp(min=1e-8)
    x = x_init.clone().detach().requires_grad_(True)
    opt = optim.Adam([x], lr=lr)

    for i in range(n_iters):
        opt.zero_grad()
        emb = encoder(x)
        feat = ((emb - target_emb) ** 2).sum() / norm_emb
        loss = feat + la * alpha_norm(x) + ltv * total_variation(x)
        loss.backward()
        opt.step()
        with torch.no_grad():
            x.clamp_(0, 1)
        if verbose and (i + 1) % 50 == 0:
            print(f'    [Mahendran] {i+1}/{n_iters} feat={feat.item():.5f}')

    return x.detach()


# ============================================================================
# 8. SUPER-HYBRID: главная функция
# ============================================================================
def super_hybrid_inversion(target_emb, G, sd_pipe, encoder, device,
                            n_iters_sg=400,
                            sd_strength=0.35, sd_steps=30,
                            sd_guidance=5.0,
                            n_iters_pix=150,
                            verbose=True):
    """
    Полный пайплайн супер-гибрида.

    Этапы:
      1. StyleGAN2 W+ inversion → фотореалистичное лицо (256×256)
      2. SD img2img refinement  → улучшаем детали через диффузию (512×512)
      3. Mahendran pixel refine → точное совпадение ArcFace эмбеддинга

    Возвращает словарь с промежуточными результатами для визуализации.
    """
    results = {}

    # ── Stage 1: StyleGAN W+ ──────────────────────────────────────────────
    if verbose:
        print('\n=== Stage 1/3: StyleGAN2 W+ inversion ===')
    t0 = time.time()
    x_sg, _ = stylegan_w_inversion(
        target_emb, G, encoder, device,
        n_iters=n_iters_sg, verbose=verbose
    )
    results['stage1_stylegan'] = x_sg.clone()
    if verbose:
        print(f'  Stage 1 done in {time.time()-t0:.1f}s, '
              f'output shape: {tuple(x_sg.shape)}')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Stage 2: SD img2img refinement ────────────────────────────────────
    if verbose:
        print('\n=== Stage 2/3: SD img2img refinement ===')
    t0 = time.time()
    x_sd = sd_refine(
        x_sg, sd_pipe, target_emb, encoder,
        strength=sd_strength, guidance_scale=sd_guidance,
        num_steps=sd_steps, device=device, verbose=verbose
    )
    results['stage2_diffusion'] = x_sd.clone()
    if verbose:
        print(f'  Stage 2 done in {time.time()-t0:.1f}s, '
              f'output shape: {tuple(x_sd.shape)}')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Stage 3: Mahendran pixel refinement ───────────────────────────────
    if verbose:
        print('\n=== Stage 3/3: Mahendran pixel refinement ===')
    t0 = time.time()
    x_final = mahendran_refine(
        x_sd, target_emb, encoder,
        n_iters=n_iters_pix, verbose=verbose
    )
    results['stage3_final'] = x_final.clone()
    if verbose:
        print(f'  Stage 3 done in {time.time()-t0:.1f}s')

    return results


# ============================================================================
# 9. МЕТРИКИ
# ============================================================================
def ips(orig, recon, encoder):
    """Identity Preservation Score = cos(ArcFace(orig), ArcFace(recon))."""
    with torch.no_grad():
        # Приводим к одному размеру
        if orig.shape[-1] != recon.shape[-1]:
            recon_resized = nn.functional.interpolate(
                recon, size=orig.shape[-2:],
                mode='bilinear', align_corners=False)
        else:
            recon_resized = recon
        e_o = encoder(orig)
        e_r = encoder(recon_resized)
        return (e_o * e_r).sum(dim=1).mean().item()


def psnr(x, y):
    # Приводим к одному размеру
    if x.shape[-1] != y.shape[-1]:
        y = nn.functional.interpolate(y, size=x.shape[-2:],
                                       mode='bilinear', align_corners=False)
    mse = ((x - y) ** 2).mean().item()
    return 10 * np.log10(1.0 / mse) if mse > 1e-10 else float('inf')


def ssim_torch(x, y):
    if x.shape[-1] != y.shape[-1]:
        y = nn.functional.interpolate(y, size=x.shape[-2:],
                                       mode='bilinear', align_corners=False)
    mx, my = x.mean(), y.mean()
    vx = ((x - mx) ** 2).mean()
    vy = ((y - my) ** 2).mean()
    cov = ((x - mx) * (y - my)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2) /
            ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))).item()


def lis_score(ips_val, ssim_val, psnr_val, time_s, max_time=120.0):
    psnr_n  = min(max(psnr_val, 0) / 40.0, 1.0)
    speed_n = max(0.0, 1.0 - time_s / max_time)
    return 0.5 * ips_val + 0.2 * ssim_val + 0.15 * psnr_n + 0.15 * speed_n


# ============================================================================
# 10. УТИЛИТЫ
# ============================================================================
def load_face(path, size=256, device='cuda'):
    img = Image.open(path).convert('RGB').resize((size, size), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def to_np(t):
    return t.detach().cpu().clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()


def save_visualization(orig, results, ips_values, out_path):
    """
    Визуализирует все три этапа супер-гибрида + оригинал.
    """
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.5))

    ax[0].imshow(to_np(orig))
    ax[0].set_title('Original', fontsize=11)
    ax[0].axis('off')

    titles_keys = [
        ('Stage 1: StyleGAN-W',    'stage1_stylegan'),
        ('Stage 2: + SD refine',   'stage2_diffusion'),
        ('Stage 3: + Mahendran',   'stage3_final'),
    ]
    for i, (title, key) in enumerate(titles_keys):
        ax[i+1].imshow(to_np(results[key]))
        ax[i+1].set_title(f'{title}\nIPS={ips_values[key]:.3f}', fontsize=10)
        ax[i+1].axis('off')

    fig.suptitle('Super-Hybrid: StyleGAN2 → Stable Diffusion → Mahendran',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ============================================================================
# 11. RUN
# ============================================================================
def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free = (torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated()) / 1e9
        print(f'VRAM free: {free:.2f} GB')

    # ── Загружаем все три модели ─────────────────────────────────────────
    print('\n--- Loading models ---')
    encoder = ArcFaceEncoder(device=device)
    print('  ArcFace loaded')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    G = load_stylegan2(device=device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free = (torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated()) / 1e9
        print(f'  VRAM free after StyleGAN: {free:.2f} GB')

    sd_pipe = load_sd_img2img(device=device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free = (torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated()) / 1e9
        print(f'  VRAM free after SD: {free:.2f} GB')

    # ── Подготовка ────────────────────────────────────────────────────────
    img = load_face(args.image, size=256, device=device)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        target_emb = encoder(img)

    # ── Запускаем супер-гибрид ────────────────────────────────────────────
    print('\n=========================================')
    print('  Super-Hybrid: StyleGAN → SD → Mahendran ')
    print('=========================================')

    t_total = time.time()
    results = super_hybrid_inversion(
        target_emb, G, sd_pipe, encoder, device,
        n_iters_sg=args.iters_sg,
        sd_strength=args.sd_strength,
        sd_steps=args.sd_steps,
        sd_guidance=args.sd_guidance,
        n_iters_pix=args.iters_pix,
        verbose=True,
    )
    total_time = time.time() - t_total

    # ── Подсчёт метрик ────────────────────────────────────────────────────
    print('\n--- Metrics ---')
    metrics = {}
    ips_values = {}
    for stage, x in results.items():
        i = ips(img, x, encoder)
        p = psnr(img, x)
        s = ssim_torch(img, x)
        metrics[stage] = {'ips': i, 'psnr': p, 'ssim': s}
        ips_values[stage] = i

    print('\n' + '=' * 70)
    print(f'{"Stage":<25} {"IPS":>8} {"PSNR":>8} {"SSIM":>8}')
    print('-' * 70)
    print(f'{"Stage 1 (StyleGAN-W)":<25} '
          f'{metrics["stage1_stylegan"]["ips"]:>8.4f} '
          f'{metrics["stage1_stylegan"]["psnr"]:>8.2f} '
          f'{metrics["stage1_stylegan"]["ssim"]:>8.4f}')
    print(f'{"Stage 2 (+ SD refine)":<25} '
          f'{metrics["stage2_diffusion"]["ips"]:>8.4f} '
          f'{metrics["stage2_diffusion"]["psnr"]:>8.2f} '
          f'{metrics["stage2_diffusion"]["ssim"]:>8.4f}')
    print(f'{"Stage 3 (+ Mahendran)":<25} '
          f'{metrics["stage3_final"]["ips"]:>8.4f} '
          f'{metrics["stage3_final"]["psnr"]:>8.2f} '
          f'{metrics["stage3_final"]["ssim"]:>8.4f}')
    print('=' * 70)
    print(f'Total time: {total_time:.1f}s')

    # LIS для финального результата
    final_lis = lis_score(
        metrics['stage3_final']['ips'],
        metrics['stage3_final']['ssim'],
        metrics['stage3_final']['psnr'],
        total_time
    )
    print(f'Super-Hybrid LIS: {final_lis:.4f}')

    # ── Сохранение ────────────────────────────────────────────────────────
    name = Path(args.image).stem
    out_path = Path(args.out_dir) / f'super_hybrid_{name}.png'
    save_visualization(img, results, ips_values, out_path)


# ============================================================================
# 12. CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Super-Hybrid: StyleGAN → SD → Mahendran inversion')
    parser.add_argument('--image',       type=str, required=True)
    parser.add_argument('--iters_sg',    type=int, default=400,
                        help='итераций StyleGAN W+ (300-700)')
    parser.add_argument('--sd_strength', type=float, default=0.35,
                        help='strength для SD img2img (0.3-0.5). '
                             'Меньше = больше сохраняем StyleGAN, '
                             'больше = больше доверяем SD')
    parser.add_argument('--sd_steps',    type=int, default=30,
                        help='шагов DDIM в SD img2img')
    parser.add_argument('--sd_guidance', type=float, default=5.0,
                        help='guidance_scale для SD (3-7)')
    parser.add_argument('--iters_pix',   type=int, default=150,
                        help='итераций Mahendran pixel refinement')
    parser.add_argument('--out_dir',     type=str, default='./outputs')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()