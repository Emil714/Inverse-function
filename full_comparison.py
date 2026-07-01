"""
Итоговое сравнение всех методов инверсии ArcFace-эмбеддингов.

Запускает все 6 методов на одном изображении и строит итоговую таблицу:

  ┌─────────────────────┬────────┬────────┬────────┬───────────┬────────┐
  │ Method              │   IPS  │  PSNR  │  SSIM  │     Time  │   LIS  │
  ├─────────────────────┼────────┼────────┼────────┼───────────┼────────┤
  │ Mahendran           │  ...   │  ...   │  ...   │   ...s    │  ...   │
  │ Dosovitskiy         │  ...   │  ...   │  ...   │   ...s    │  ...   │
  │ Hybrid (ours)       │  ...   │  ...   │  ...   │   ...s    │  ...   │
  │ StyleGAN-W          │  ...   │  ...   │  ...   │   ...s    │  ...   │
  │ StyleGAN-Hybrid     │  ...   │  ...   │  ...   │   ...s    │  ...   │
  │ Diffusion-Hybrid    │  ...   │  ...   │  ...   │   ...s    │  ...   │
  └─────────────────────┴────────┴────────┴────────┴───────────┴────────┘

Использование:
    python full_comparison.py --image face.jpg \\
        --arcface_decoder checkpoints/arcface_decoder.pt

Предварительно должны быть готовы:
    - ./weights/iresnet50.pth           (ArcFace веса)
    - checkpoints/arcface_decoder.pt    (обученный декодер Dosovitskiy)
    - StyleGAN2 и SD скачаются автоматически при первом запуске
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ============================================================================
# ИМПОРТ ВСЕХ МЕТОДОВ
# ============================================================================
def import_module(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location('m', path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================================
# РЕГУЛЯРИЗАТОРЫ (локальная копия)
# ============================================================================
def alpha_norm(x, alpha=6):
    return x.abs().pow(alpha).sum()

def total_variation(x, beta=2.0):
    dh = x[:,:,1:,:] - x[:,:,:-1,:]
    dw = x[:,:,:,1:] - x[:,:,:,:-1]
    return ((dh[:,:,:,:-1]**2 + dw[:,:,:-1,:]**2 + 1e-8)**(beta/2)).sum()

def invert_mahendran_local(target, encoder, n_iters=3000, lr=0.01,
                            la=1e-8, ltv=1e-4, verbose=False):
    with torch.no_grad():
        t_emb = encoder(target)
        norm  = (t_emb**2).sum().clamp(min=1e-8)
    x = torch.randn_like(target) * 0.5
    x = x.clamp(0,1).detach().requires_grad_(True)
    opt = optim.Adam([x], lr=lr)
    for i in range(n_iters):
        opt.zero_grad()
        feat = ((encoder(x) - t_emb)**2).sum() / norm
        loss = feat + la*alpha_norm(x) + ltv*total_variation(x)
        loss.backward(); opt.step()
        with torch.no_grad(): x.clamp_(0,1)
        if verbose and (i+1) % 500 == 0:
            print(f'  [Mahendran] {i+1}/{n_iters} feat={feat.item():.5f}')
    return x.detach()


# ============================================================================
# МЕТРИКИ
# ============================================================================
def ips_metric(orig, recon, encoder):
    with torch.no_grad():
        e_o = encoder(orig); e_r = encoder(recon)
        return (e_o * e_r).sum(dim=1).mean().item()

def psnr_metric(x, y):
    mse = ((x-y)**2).mean().item()
    return 10 * np.log10(1.0/mse) if mse > 1e-10 else float('inf')

def ssim_metric(x, y):
    mx, my = x.mean(), y.mean()
    vx = ((x-mx)**2).mean(); vy = ((y-my)**2).mean()
    cov = ((x-mx)*(y-my)).mean()
    c1, c2 = 0.01**2, 0.03**2
    return ((2*mx*my+c1)*(2*cov+c2)/((mx**2+my**2+c1)*(vx+vy+c2))).item()

def lis_metric(ips_val, ssim_val, psnr_val, time_s, max_time=120.0):
    psnr_n  = min(max(psnr_val, 0) / 40.0, 1.0)
    speed_n = max(0.0, 1.0 - time_s / max_time)
    return 0.5*ips_val + 0.2*ssim_val + 0.15*psnr_n + 0.15*speed_n

def compute_metrics(orig, recon, encoder, elapsed):
    i = ips_metric(orig, recon, encoder)
    p = psnr_metric(orig, recon)
    s = ssim_metric(orig, recon)
    l = lis_metric(i, s, p, elapsed)
    return dict(recon=recon, ips=i, psnr=p, ssim=s, time=elapsed, lis=l)


# ============================================================================
# ИТОГОВАЯ ВИЗУАЛИЗАЦИЯ
# ============================================================================
def save_full_comparison(orig, results, out_path):
    """
    Строит итоговую сетку изображений и таблицу метрик под ними.
    """
    n_methods = len(results)
    fig = plt.figure(figsize=(3*(n_methods+1), 7))
    gs = gridspec.GridSpec(2, n_methods+1,
                           height_ratios=[3, 1.8], hspace=0.05)

    def to_np(t):
        return t.detach().cpu().clamp(0,1).squeeze(0).permute(1,2,0).numpy()

    # Оригинал
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(to_np(orig)); ax0.set_title('Original', fontsize=11)
    ax0.axis('off')

    # Реконструкции
    for i, (name, d) in enumerate(results.items()):
        ax = fig.add_subplot(gs[0, i+1])
        ax.imshow(to_np(d['recon'])); ax.axis('off')
        ax.set_title(f'{name}\nIPS={d["ips"]:.3f}', fontsize=9)

    # Таблица метрик
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis('off')

    col_labels = ['Method', 'IPS ↑', 'PSNR ↑', 'SSIM ↑', 'Time', 'LIS ↑']
    rows = []
    for name, d in results.items():
        rows.append([
            name,
            f'{d["ips"]:.4f}',
            f'{d["psnr"]:.2f}',
            f'{d["ssim"]:.4f}',
            f'{d["time"]:.1f}s',
            f'{d["lis"]:.4f}',
        ])

    table = ax_table.table(
        cellText=rows, colLabels=col_labels,
        loc='center', cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Выделяем лучший IPS
    best_ips = max(d['ips'] for d in results.values())
    best_lis = max(d['lis'] for d in results.values())
    for ri, (name, d) in enumerate(results.items()):
        if d['ips'] == best_ips:
            table[ri+1, 1].set_facecolor('#c8f7c5')
        if d['lis'] == best_lis:
            table[ri+1, 5].set_facecolor('#aed6f1')

    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',           type=str, required=True)
    parser.add_argument('--arcface_decoder', type=str,
                        default='./checkpoints/arcface_decoder.pt')
    parser.add_argument('--iters_mahendran', type=int, default=3000)
    parser.add_argument('--iters_w',         type=int, default=500)
    parser.add_argument('--iters_pix',       type=int, default=300)
    parser.add_argument('--ddim_steps',      type=int, default=30)
    parser.add_argument('--out_dir',         type=str, default='./outputs')
    parser.add_argument('--skip_stylegan',   action='store_true')
    parser.add_argument('--skip_diffusion',  action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # --- Загружаем модули ---
    ah  = import_module('./arcface_hybrid.py')
    sg  = import_module('./stylegan_inversion.py') if not args.skip_stylegan else None
    dif = import_module('./diffusion_inversion.py') if not args.skip_diffusion else None

    # --- Энкодер (общий для всех) ---
    encoder = ah.ArcFaceEncoder(device=device)

    # --- Декодер Dosovitskiy ---
    decoder = ah.FaceDecoder(512, 112).to(device).eval()
    ckpt = torch.load(args.arcface_decoder, map_location=device)
    decoder.load_state_dict(ckpt['state_dict'])

    # --- Загружаем изображение (112×112 для ArcFace) ---
    def load_face_112(path):
        img = Image.open(path).convert('RGB').resize((112, 112), Image.BILINEAR)
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(device)

    img = load_face_112(args.image)

    with torch.no_grad():
        target_emb = encoder(img)

    results = {}

    # ---- 1. Mahendran ----
    print('\n[1/6] Mahendran-Vedaldi...')
    t0 = time.time()
    rec = invert_mahendran_local(img, encoder, args.iters_mahendran, verbose=True)
    results['Mahendran'] = compute_metrics(img, rec, encoder, time.time()-t0)

    # ---- 2. Dosovitskiy ----
    print('\n[2/6] Dosovitskiy-Brox...')
    t0 = time.time()
    rec = ah.invert_dosovitskiy(img, encoder, decoder)
    results['Dosovitskiy'] = compute_metrics(img, rec, encoder, time.time()-t0)

    # ---- 3. Hybrid (ours) ----
    print('\n[3/6] Hybrid (ours)...')
    t0 = time.time()
    rec = ah.invert_hybrid(img, encoder, decoder,
                           n_iters=args.iters_mahendran//4, verbose=True)
    results['Hybrid (ours)'] = compute_metrics(img, rec, encoder, time.time()-t0)

    # ---- 4-5. StyleGAN ----
    if sg is not None:
        G = sg.load_stylegan2(device=device)
        with torch.no_grad():
            emb_256 = encoder(img)  # эмбеддинг тот же

        print('\n[4/6] StyleGAN-W inversion...')
        t0 = time.time()
        rec_w, _ = sg.stylegan_w_inversion(
            emb_256, G, encoder, device,
            n_iters=args.iters_w, verbose=True)
        results['StyleGAN-W'] = compute_metrics(
            img, nn.functional.interpolate(rec_w, (112,112),
                                           mode='bilinear', align_corners=False),
            encoder, time.time()-t0)

        print('\n[5/6] StyleGAN-Hybrid...')
        t0 = time.time()
        rec_sh = sg.stylegan_hybrid(
            emb_256, G, encoder, device,
            n_iters_w=args.iters_w, n_iters_pix=args.iters_pix, verbose=True)
        results['StyleGAN-Hybrid'] = compute_metrics(
            img, nn.functional.interpolate(rec_sh, (112,112),
                                           mode='bilinear', align_corners=False),
            encoder, time.time()-t0)

    # ---- 6. Diffusion ----
    if dif is not None:
        pipe = dif.load_stable_diffusion(device=device)

        print('\n[6/6] Diffusion-Hybrid...')
        t0 = time.time()
        rec_dh = dif.diffusion_hybrid(
            img, target_emb, pipe, encoder,
            n_ddim_steps=args.ddim_steps, n_pix_steps=args.iters_pix,
            device=device, verbose=True)
        results['Diffusion-Hybrid'] = compute_metrics(
            img, nn.functional.interpolate(rec_dh, (112,112),
                                           mode='bilinear', align_corners=False),
            encoder, time.time()-t0)

    # ---- Итоговая таблица ----
    print('\n' + '='*75)
    print(f'{"Method":<22} {"IPS":>8} {"PSNR":>8} {"SSIM":>8} '
          f'{"Time":>10} {"LIS":>8}')
    print('-'*75)
    for name, d in results.items():
        marker = ' ◄' if d['ips'] == max(v['ips'] for v in results.values()) else ''
        print(f'{name:<22} {d["ips"]:>8.4f} {d["psnr"]:>8.2f} '
              f'{d["ssim"]:>8.4f} {d["time"]:>9.1f}s {d["lis"]:>8.4f}{marker}')
    print('='*75)
    print('◄ = best IPS')

    # ---- Сохранение ----
    name = Path(args.image).stem
    out_path = Path(args.out_dir) / f'full_comparison_{name}.png'
    save_full_comparison(img, results, out_path)


if __name__ == '__main__':
    main()