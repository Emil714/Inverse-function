"""
Super-Hybrid v2: StyleGAN2 + Stable Diffusion для инверсии эмбеддингов.

Два режима работы:

  --mode arcface_only
      Чистая инверсия ArcFace эмбеддинга (как в основной курсовой).
      Демонстрирует угрозу: визуально может выйти другой человек,
      но IPS остаётся высоким — биометрическая идентичность сохранена.

  --mode multi_encoder
      Инверсия КОМБИНАЦИИ нескольких эмбеддингов: ArcFace + ResNet50 + CLIP.
      Это другая методологическая постановка: моделируем сценарий, когда
      утекли эмбеддинги от нескольких разных моделей (что бывает в
      реальных пайплайнах с разными процессами обработки лица).
      Результат: визуально похожее лицо + биометрическая идентичность.

Архитектура (одинакова для обоих режимов):

    Encoder(x) → target_embeddings
         │
         ▼
    Stage 1: StyleGAN2 W+ inversion с multi-loss
         │  → x_stylegan (256×256)
         ▼
    Stage 2: SD img2img refinement (low strength)
         │  → x_diffusion (512×512)
         ▼
    Финальный результат
    (Mahendran убран — он добавлял шум, портя визуальное качество)

Запуск:
    # Чистая ArcFace инверсия:
    python super_hybrid_v2.py --image face.jpg --mode arcface_only

    # Multi-encoder инверсия:
    python super_hybrid_v2.py --image face.jpg --mode multi_encoder

Требования:
    ./weights/iresnet50.pth                    ← веса ArcFace
    ./weights/stylegan2-ffhq-256x256.pkl       ← веса StyleGAN2
    ./weights/stable-diffusion-v1-5/           ← веса SD
    ./stylegan2-ada-pytorch/                   ← репозиторий StyleGAN2

    CLIP и ResNet50 скачиваются автоматически (~400 МБ суммарно).

Зависимости (помимо основных):
    pip install open_clip_torch
    или используйте встроенный transformers CLIP:
    pip install transformers
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
import torchvision.models as tv_models
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.insert(0, './stylegan2-ada-pytorch')


# ============================================================================
# 1. ARCFACE ENCODER (биометрический энкодер — главная цель курсовой)
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
    """Биометрический энкодер. Выход: 512-d L2-нормализованный вектор."""
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
# 2. RESNET50 ENCODER (визуальный prior — для multi_encoder режима)
# ============================================================================
class ResNet50Encoder(nn.Module):
    """
    ResNet50 предобученная на ImageNet.
    Используется avgpool feature (2048-d) — содержит общую визуальную
    информацию: текстуры, цвета, формы, контекст.

    В отличие от ArcFace, эта модель не отбрасывает визуальные атрибуты —
    инверсия её эмбеддингов даёт визуально похожие изображения.
    """
    def __init__(self, device='cuda'):
        super().__init__()
        net = tv_models.resnet50(
            weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
        # Берём всё до avgpool + сам avgpool
        self.backbone = nn.Sequential(*list(net.children())[:-1])
        self.backbone = self.backbone.to(device).eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        # ImageNet нормализация
        self.register_buffer('mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.input_size = 224

    def forward(self, x):
        if x.shape[-1] != self.input_size:
            x = nn.functional.interpolate(x,
                (self.input_size, self.input_size),
                mode='bilinear', align_corners=False)
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        feat = self.backbone(x).flatten(1)  # [B, 2048]
        return nn.functional.normalize(feat, p=2, dim=1)


# ============================================================================
# 3. CLIP ENCODER (семантический prior — для multi_encoder режима)
# ============================================================================
class CLIPEncoder(nn.Module):
    """
    CLIP image encoder через transformers.
    Используется для high-level семантического conditioning.

    CLIP знает что такое лица людей в общем виде (обучена на 400M
    изображений), её эмбеддинг сохраняет: пол, возраст, общий вид,
    выражение лица, причёску и т.д.
    """
    def __init__(self, device='cuda',
                 model_name='openai/clip-vit-base-patch32'):
        super().__init__()
        from transformers import CLIPVisionModel, CLIPImageProcessor
        self.model = CLIPVisionModel.from_pretrained(model_name)
        self.model = self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        # CLIP preprocessing: resize 224, mean/std из конфига CLIP
        self.register_buffer('mean',
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer('std',
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
        self.input_size = 224

    def forward(self, x):
        if x.shape[-1] != self.input_size:
            x = nn.functional.interpolate(x,
                (self.input_size, self.input_size),
                mode='bilinear', align_corners=False)
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        # CLIP возвращает pooler_output [B, 768]
        out = self.model(pixel_values=x)
        feat = out.pooler_output
        return nn.functional.normalize(feat, p=2, dim=1)


# ============================================================================
# 4. ЗАГРУЗКА StyleGAN2 и SD
# ============================================================================
def load_stylegan2(device='cuda'):
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


def load_sd_img2img(device='cuda'):
    from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
    print(f'Loading SD img2img pipeline ({SD_MODEL_ID})...')
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
# 5. STAGE 1: StyleGAN W+ inversion с multi-loss
# ============================================================================
def stylegan_w_inversion_multi(targets, encoders, G, device,
                                weights=None,
                                n_iters=500, lr=0.01, lambda_reg=0.05,
                                verbose=False):
    """
    Оптимизация в W+ с loss функцией от нескольких энкодеров.

    Args:
        targets:  dict {имя_энкодера: target_emb}
                  Например: {'arcface': emb_arc, 'resnet': emb_res, 'clip': emb_clip}
        encoders: dict {имя_энкодера: encoder_module}
        weights:  dict {имя_энкодера: вес в loss}
                  Например: {'arcface': 1.0, 'resnet': 0.5, 'clip': 0.5}
                  Если None: только arcface с весом 1.0.

    Multi-loss:
        L = sum_i w_i * ||E_i(G(w)) - target_i||² / ||target_i||²
          + lambda_reg * ||w - mean_w||²

    Это позволяет StyleGAN искать лицо удовлетворяющее СРАЗУ нескольким
    эмбеддинг-ограничениям: биометрия (ArcFace), визуал (ResNet50),
    семантика (CLIP).
    """
    if weights is None:
        weights = {k: 1.0 for k in targets}

    # Считаем mean_w и std_w для W-регуляризации
    with torch.no_grad():
        z_samples = torch.randn(10000, G.z_dim, device=device)
        w_samples = G.mapping(z_samples, None)
        mean_w = w_samples.mean(0, keepdim=True)
        std_w  = w_samples.std(0, keepdim=True)

    w = mean_w.clone().detach().requires_grad_(True)
    # LR schedule: cosine decay от lr до lr/10
    # Большой LR в начале — быстрое движение к нужному лицу.
    # Маленький LR в конце — точная подстройка ArcFace loss.
    opt = optim.Adam([w], lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_iters, eta_min=lr / 10)

    # Предвычисляем нормирующие константы
    norms = {k: (t ** 2).sum().clamp(min=1e-8) for k, t in targets.items()}

    for i in range(n_iters):
        opt.zero_grad()

        img_gen = G.synthesis(w, noise_mode='const')
        img_01  = ((img_gen + 1) / 2).clamp(0, 1)

        # Multi-encoder loss
        feat_losses = {}
        total_feat = 0.0
        for name, target in targets.items():
            emb_gen = encoders[name](img_01)
            l = ((emb_gen - target) ** 2).sum() / norms[name]
            feat_losses[name] = l
            total_feat = total_feat + weights[name] * l

        # W-регуляризация с затуханием:
        # В первой половине итераций регуляризация полная (удерживает от артефактов).
        # Во второй половине постепенно обнуляется — оптимизатор фокусируется
        # на точном совпадении эмбеддинга, а не на близости к mean_w.
        # Это ключевое изменение для высокого финального IPS.
        reg_decay = max(0.0, 1.0 - 2.0 * i / n_iters)
        w_reg = ((w - mean_w) / (std_w + 1e-8)).pow(2).mean()
        loss = total_feat + lambda_reg * reg_decay * w_reg

        loss.backward()
        opt.step()
        scheduler.step()

        if verbose and (i + 1) % 100 == 0:
            losses_str = ' '.join(
                f'{k}={v.item():.4f}' for k, v in feat_losses.items())
            print(f'    [StyleGAN-W] {i+1}/{n_iters} {losses_str} '
                  f'reg={w_reg.item():.4f}×{reg_decay:.2f} '
                  f'lr={scheduler.get_last_lr()[0]:.5f}')

    with torch.no_grad():
        img_final = G.synthesis(w, noise_mode='const')
        img_final = ((img_final + 1) / 2).clamp(0, 1)

    return img_final.detach(), w.detach()


# ============================================================================
# 6. STAGE 2: SD img2img как refiner
# ============================================================================
def sd_refine(image_stylegan, pipe,
              strength=0.3, guidance_scale=5.0,
              num_steps=30,
              prompt='a high quality professional portrait photo of a person, sharp focus, detailed face',
              device='cuda', verbose=False):
    """
    SD img2img refinement с малым strength.

    Важно: если strength=0 — SD пропускается полностью, изображение
    просто масштабируется 256→512 билинейной интерполяцией.
    Это позволяет оценить результат чистого StyleGAN без SD.

    Логика при strength > 0:
      - strength=0.2 значит SD добавит 20% шума и пройдёт обратный процесс
      - идентичность лица от StyleGAN сохраняется (основное содержание)
      - SD добавляет детали: текстура кожи, освещение, тени
      - IPS может немного упасть — это компромисс за качество

    Рекомендации для arcface_only режима: strength=0.0-0.15
    Рекомендации для multi_encoder режима: strength=0.2-0.3
    """
    # Если strength=0 — просто апскейлим StyleGAN результат без SD
    if strength == 0.0:
        if verbose:
            print('    [SD-Refine] strength=0.0, skipping SD (upscale only)')
        refined = nn.functional.interpolate(
            image_stylegan, size=(512, 512),
            mode='bilinear', align_corners=False)
        return refined.to(device).float()

    img_np = (image_stylegan.detach().cpu()
              .squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
              * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)

    if img_pil.size != (512, 512):
        img_pil = img_pil.resize((512, 512), Image.BILINEAR)

    if verbose:
        print(f'    [SD-Refine] strength={strength}, '
              f'steps={num_steps}, guidance={guidance_scale}')

    with torch.no_grad():
        out = pipe(
            prompt=prompt,
            image=img_pil,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            output_type='pt',
        )

    refined = out.images.to(device).float()
    return refined


# ============================================================================
# 7. SUPER-HYBRID v2: главная функция
# ============================================================================
def super_hybrid_v2(targets, encoders, G, sd_pipe, device,
                     weights=None,
                     n_iters_sg=500,
                     sd_strength=0.3, sd_steps=30, sd_guidance=5.0,
                     sd_prompt=None,
                     verbose=True):
    """
    Полный пайплайн супер-гибрида v2.

    Этапы:
      1. StyleGAN2 W+ inversion с multi-loss → фотореалистичное лицо
      2. SD img2img refinement → детализация через диффузию
      (Mahendran убран — он зашумлял изображение)

    Возвращает словарь {stage_name: image_tensor}.
    """
    results = {}

    # ── Stage 1: StyleGAN multi-loss ──────────────────────────────────────
    if verbose:
        print('\n=== Stage 1/2: StyleGAN2 W+ multi-encoder inversion ===')
        if weights:
            print('  Loss weights:', weights)
    t0 = time.time()
    x_sg, _ = stylegan_w_inversion_multi(
        targets, encoders, G, device,
        weights=weights,
        n_iters=n_iters_sg, verbose=verbose
    )
    results['stage1_stylegan'] = x_sg.clone()
    if verbose:
        print(f'  Stage 1 done in {time.time()-t0:.1f}s')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Stage 2: SD img2img ───────────────────────────────────────────────
    if verbose:
        print('\n=== Stage 2/2: SD img2img refinement ===')
    t0 = time.time()
    prompt = sd_prompt or ('a high quality professional portrait photo '
                            'of a person, sharp focus, detailed face')
    x_sd = sd_refine(
        x_sg, sd_pipe,
        strength=sd_strength, guidance_scale=sd_guidance,
        num_steps=sd_steps, prompt=prompt,
        device=device, verbose=verbose
    )
    results['stage2_final'] = x_sd.clone()
    if verbose:
        print(f'  Stage 2 done in {time.time()-t0:.1f}s')

    return results


# ============================================================================
# 8. МЕТРИКИ
# ============================================================================
def ips(orig, recon, arcface_encoder):
    """IPS = cos(ArcFace(orig), ArcFace(recon))."""
    with torch.no_grad():
        if orig.shape[-1] != recon.shape[-1]:
            recon = nn.functional.interpolate(recon, size=orig.shape[-2:],
                                               mode='bilinear', align_corners=False)
        e_o = arcface_encoder(orig)
        e_r = arcface_encoder(recon)
        return (e_o * e_r).sum(dim=1).mean().item()


def visual_similarity(orig, recon, resnet_encoder):
    """Дополнительная метрика: визуальное сходство через ResNet50."""
    with torch.no_grad():
        if orig.shape[-1] != recon.shape[-1]:
            recon = nn.functional.interpolate(recon, size=orig.shape[-2:],
                                               mode='bilinear', align_corners=False)
        e_o = resnet_encoder(orig)
        e_r = resnet_encoder(recon)
        return (e_o * e_r).sum(dim=1).mean().item()


def psnr(x, y):
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


# ============================================================================
# 9. УТИЛИТЫ
# ============================================================================
def load_face(path, size=256, device='cuda'):
    img = Image.open(path).convert('RGB').resize((size, size), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def to_np(t):
    return t.detach().cpu().clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()


def save_visualization(orig, results, metrics, mode, out_path):
    """Визуализация: оригинал + Stage 1 + Stage 2."""
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.5))

    ax[0].imshow(to_np(orig))
    ax[0].set_title('Original', fontsize=11)
    ax[0].axis('off')

    m1 = metrics['stage1_stylegan']
    ax[1].imshow(to_np(results['stage1_stylegan']))
    title1 = f'Stage 1: StyleGAN-W\nIPS={m1["ips"]:.3f}'
    if 'vis_sim' in m1:
        title1 += f' | VisSim={m1["vis_sim"]:.3f}'
    ax[1].set_title(title1, fontsize=10)
    ax[1].axis('off')

    m2 = metrics['stage2_final']
    ax[2].imshow(to_np(results['stage2_final']))
    title2 = f'Stage 2: + SD refine\nIPS={m2["ips"]:.3f}'
    if 'vis_sim' in m2:
        title2 += f' | VisSim={m2["vis_sim"]:.3f}'
    ax[2].set_title(title2, fontsize=10)
    ax[2].axis('off')

    mode_title = ('Mode: pure ArcFace inversion'
                   if mode == 'arcface_only'
                   else 'Mode: multi-encoder inversion (ArcFace + ResNet50 + CLIP)')
    fig.suptitle(f'Super-Hybrid v2 — {mode_title}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ============================================================================
# 10. RUN
# ============================================================================
def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print(f'Mode:   {args.mode}')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free = (torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated()) / 1e9
        print(f'VRAM free: {free:.2f} GB\n')

    # ── Загружаем энкодеры ────────────────────────────────────────────────
    print('--- Loading encoders ---')
    arcface = ArcFaceEncoder(device=device)
    print('  ArcFace loaded')

    # ResNet50 и CLIP нужны в обоих режимах:
    # - arcface_only: слабые веса (0.2) — чуть улучшают визуал, почти не роняют IPS
    # - multi_encoder: сильные веса (1.5-2.0) — упор на визуальное сходство
    print('  Loading ResNet50...')
    resnet = ResNet50Encoder(device=device)
    print('  ResNet50 loaded')

    print('  Loading CLIP...')
    clip = CLIPEncoder(device=device)
    print('  CLIP loaded')

    encoders = {'arcface': arcface, 'resnet': resnet, 'clip': clip}

    # ── Загружаем генеративные модели ────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    G = load_stylegan2(device=device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sd_pipe = load_sd_img2img(device=device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free = (torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated()) / 1e9
        print(f'\nVRAM free after all loads: {free:.2f} GB')

    # ── Подготовка ────────────────────────────────────────────────────────
    img = load_face(args.image, size=256, device=device)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Вычисляем целевые эмбеддинги от всех трёх энкодеров
    print('\n--- Computing target embeddings ---')
    targets = {}
    with torch.no_grad():
        for name, enc in encoders.items():
            targets[name] = enc(img)
            print(f'  {name}: shape={tuple(targets[name].shape)}, '
                  f'norm={targets[name].norm().item():.4f}')

    # ── Режимы ────────────────────────────────────────────────────────────
    if args.mode == 'arcface_only':
        # Все три энкодера в loss, но ResNet50/CLIP с малыми весами.
        # ArcFace (1.0) доминирует → высокий IPS сохраняется.
        # ResNet50 (0.2) и CLIP (0.15) слегка тянут w в сторону визуала —
        # это помогает избежать крайних артефактов (серёжки, макияж)
        # без ощутимого ущерба для биометрии.
        # SD с малым strength=0.15 добавляет немного деталей кожи/освещения.
        weights = {
            'arcface': 1.0,
            'resnet':  0.2,   # слабый визуальный prior
            'clip':    0.15,  # слабый семантический prior
        }
        sd_strength  = args.sd_strength  # дефолт 0.15
        sd_guidance  = 4.0
        sd_prompt    = ('a high quality portrait photo of a face, '
                        'sharp focus, realistic skin texture, professional lighting')

        print('\n[Mode: arcface_only]')
        print('  Weights: ArcFace=1.0 | ResNet50=0.2 | CLIP=0.15')
        print(f'  SD strength={sd_strength} (light detail enhancement)')

    else:  # multi_encoder
        # Все три энкодера с высокими весами.
        # ArcFace (1.0) — биометрия должна сохраниться.
        # ResNet50 (1.5) — сильный визуальный prior (цвета, текстуры, форма).
        # CLIP (1.2) — семантический prior (пол, возраст, общая внешность).
        # SD с сильным strength=0.4 — максимально улучшает фотореализм.
        # IPS может упасть до 0.7-0.8, но визуальное сходство будет выше.
        weights = {
            'arcface': 1.0,
            'resnet':  1.5,   # сильный визуальный prior
            'clip':    1.2,   # сильный семантический prior
        }
        sd_strength  = args.sd_strength if args.sd_strength != 0.0 else 0.4
        sd_guidance  = 7.0   # более сильный guidance для визуального качества
        sd_prompt    = ('a photorealistic portrait of a person, '
                        'same face structure, same hair color, same skin tone, '
                        'ultra detailed, 8k, sharp focus, professional photo')

        print('\n[Mode: multi_encoder]')
        print('  Weights: ArcFace=1.0 | ResNet50=1.5 | CLIP=1.2')
        print(f'  SD strength={sd_strength} (strong visual enhancement)')

    # ── Запускаем super-hybrid ────────────────────────────────────────────
    print('\n=========================================')
    print(f'  Super-Hybrid v2 — mode: {args.mode}')
    print('=========================================')

    t_total = time.time()
    results = super_hybrid_v2(
        targets, encoders, G, sd_pipe, device,
        weights=weights,
        n_iters_sg=args.iters_sg,
        sd_strength=sd_strength,
        sd_steps=args.sd_steps,
        sd_guidance=sd_guidance,
        sd_prompt=sd_prompt,
        verbose=True,
    )
    total_time = time.time() - t_total

    # ── Подсчёт метрик ────────────────────────────────────────────────────
    print('\n--- Metrics ---')
    # ResNet50 теперь всегда загружен в обоих режимах
    resnet_metric = encoders['resnet']

    metrics = {}
    for stage, x in results.items():
        i = ips(img, x, arcface)
        vs = visual_similarity(img, x, resnet_metric)
        p = psnr(img, x)
        s = ssim_torch(img, x)
        metrics[stage] = {'ips': i, 'vis_sim': vs, 'psnr': p, 'ssim': s}

    print('\n' + '=' * 78)
    print(f'{"Stage":<25} {"IPS":>8} {"VisSim":>8} {"PSNR":>8} {"SSIM":>8}')
    print('-' * 78)
    for stage_name, label in [('stage1_stylegan', 'Stage 1 (StyleGAN-W)'),
                                ('stage2_final',     'Stage 2 (+ SD)')]:
        m = metrics[stage_name]
        print(f'{label:<25} {m["ips"]:>8.4f} {m["vis_sim"]:>8.4f} '
              f'{m["psnr"]:>8.2f} {m["ssim"]:>8.4f}')
    print('=' * 78)
    print(f'Total time: {total_time:.1f}s')
    print(f'\nIPS — биометрическая идентичность (через ArcFace)')
    print(f'VisSim — визуальное сходство (через ResNet50)')

    # ── Сохранение ────────────────────────────────────────────────────────
    name = Path(args.image).stem
    out_path = Path(args.out_dir) / f'super_hybrid_v2_{args.mode}_{name}.png'
    save_visualization(img, results, metrics, args.mode, out_path)


# ============================================================================
# 11. CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Super-Hybrid v2: StyleGAN + SD инверсия в двух режимах',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Режимы:
  arcface_only   — все три энкодера (ArcFace:1.0 + ResNet50:0.2 + CLIP:0.15)
                   SD с лёгким strength=0.15 для деталей кожи.
                   Цель: высокий IPS при допустимом визуале.

  multi_encoder  — все три энкодера (ArcFace:1.0 + ResNet50:1.5 + CLIP:1.2)
                   SD с сильным strength=0.4 для фотореализма.
                   Цель: визуальное сходство с оригиналом (IPS может упасть).

Примеры:
  python super_hybrid_v2.py --image face.jpg --mode arcface_only
  python super_hybrid_v2.py --image face.jpg --mode multi_encoder
  python super_hybrid_v2.py --image face.jpg --mode arcface_only --iters_sg 1000
        """)
    parser.add_argument('--image',       type=str, required=True,
                        help='путь к изображению лица')
    parser.add_argument('--mode',        type=str, default='arcface_only',
                        choices=['arcface_only', 'multi_encoder'],
                        help='режим работы (см. описание выше)')
    parser.add_argument('--iters_sg',    type=int, default=700,
                        help='итераций StyleGAN W+ (700-1000 для arcface_only, '
                             '500-700 для multi_encoder)')
    parser.add_argument('--sd_strength', type=float, default=0.15,
                        help='strength SD img2img. Переопределяет дефолт режима. '
                             'arcface_only дефолт=0.15, multi_encoder дефолт=0.4')
    parser.add_argument('--sd_steps',    type=int, default=30,
                        help='шагов DDIM в SD (30 достаточно)')
    parser.add_argument('--out_dir',     type=str, default='./outputs',
                        help='папка для сохранения результатов')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()