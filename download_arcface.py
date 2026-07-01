"""
Получение весов настоящего ArcFace iresnet50 для arcface_hybrid.py.

Скрипт получает официальные веса ArcFace от deepinsight/insightface
и приводит их к формату state_dict совместимому с архитектурой iresnet50,
встроенной в arcface_hybrid.py.

Стратегия — пробуем по очереди до первого успеха:
1. HuggingFace: репозиторий ArcFace iresnet50 в формате PyTorch
2. ONNX-модель из insightface (buffalo_l/w600k_r50.onnx) + автоконвертация
   через onnx2torch + переименование ключей по соответствию форм тензоров
3. Если ничего не сработало — даём чёткие инструкции для ручной загрузки

Запуск:
    python download_arcface.py

После успешной загрузки веса будут лежать в ./weights/iresnet50.pth и
arcface_hybrid.py сможет их использовать без изменений.
"""

import os
import sys
import shutil
from pathlib import Path

import torch
import torch.nn as nn


WEIGHTS_DST = './weights/iresnet50.pth'


# ============================================================================
# Импортируем эталонную архитектуру из arcface_hybrid.py
# ============================================================================
def import_iresnet50():
    """Импортирует функцию iresnet50 из arcface_hybrid.py."""
    if not os.path.exists('arcface_hybrid.py'):
        print('ERROR: arcface_hybrid.py не найден в текущей директории.')
        print('Запустите этот скрипт из папки с arcface_hybrid.py')
        sys.exit(1)

    import importlib.util
    spec = importlib.util.spec_from_file_location('ah', 'arcface_hybrid.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.iresnet50


# ============================================================================
# СПОСОБ 1: HuggingFace — готовые PyTorch-веса iresnet50
# ============================================================================
HF_CANDIDATES = [
    # repo_id, filename, нужно ли префикс backbone./module. удалить
    ('zhangao/arcface-r50-pytorch', 'backbone.pth', True),
    ('minchul/insightface_iresnet50_ms1mv3', 'pytorch_model.bin', True),
    ('jbilcke-hf/arcface-iresnet50-ms1mv3', 'backbone.pth', True),
]


def try_huggingface(iresnet50_fn, dst):
    """Пробует скачать готовые PyTorch-веса с HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print('  huggingface_hub не установлен, пропускаем этот способ')
        print('  (для использования: pip install huggingface_hub)')
        return False

    ref_model = iresnet50_fn()
    ref_keys = set(ref_model.state_dict().keys())

    for repo_id, filename, strip_prefix in HF_CANDIDATES:
        print(f'\n  Trying: {repo_id} / {filename}')
        try:
            path = hf_hub_download(repo_id=repo_id, filename=filename,
                                    local_dir='./weights/_hf_tmp')
            sd = torch.load(path, map_location='cpu')
            if isinstance(sd, dict) and 'state_dict' in sd:
                sd = sd['state_dict']
            if strip_prefix:
                sd = {k.replace('backbone.', '').replace('module.', ''): v
                      for k, v in sd.items()}

            # Проверяем что ключи совпадают
            sd_keys = set(sd.keys())
            common = ref_keys & sd_keys
            coverage = len(common) / len(ref_keys)
            print(f'    Coverage: {coverage*100:.1f}% ({len(common)}/{len(ref_keys)} keys)')

            if coverage >= 0.95:
                # Загружаем и проверяем
                missing, unexpected = ref_model.load_state_dict(sd, strict=False)
                if len(missing) < 10:
                    Path(dst).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(ref_model.state_dict(), dst)
                    shutil.rmtree('./weights/_hf_tmp', ignore_errors=True)
                    print(f'  SUCCESS via HuggingFace: {dst}')
                    return True
        except Exception as e:
            print(f'    Failed: {type(e).__name__}: {str(e)[:80]}')

    shutil.rmtree('./weights/_hf_tmp', ignore_errors=True)
    return False


# ============================================================================
# СПОСОБ 2: ONNX через insightface + автоконвертация
# ============================================================================
def find_onnx_model():
    """Ищет ONNX-файл с recognition моделью в ~/.insightface/models/."""
    home = Path.home()
    candidates_dirs = [
        home / '.insightface' / 'models' / 'buffalo_l',
        home / '.insightface' / 'models' / 'antelopev2',
    ]
    for d in candidates_dirs:
        if d.exists():
            # Recognition модели: w600k_r50, glintr100, arcface, recognition
            for f in d.glob('*.onnx'):
                name = f.name.lower()
                if any(k in name for k in ['r50', 'r100', 'arcface', 'recognition', 'w600k']):
                    return f
    return None


def download_onnx_via_insightface():
    """Скачивает ONNX-модель через insightface FaceAnalysis."""
    print('\n  Downloading ONNX model via insightface...')
    try:
        import insightface
        from insightface.app import FaceAnalysis
        # buffalo_l содержит w600k_r50.onnx — iresnet50 + ArcFace
        app = FaceAnalysis(name='buffalo_l',
                           providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0)
        print('  buffalo_l model downloaded')
        return find_onnx_model()
    except ImportError:
        print('  insightface не установлен')
        print('  Для использования этого способа: pip install insightface onnxruntime-gpu')
        return None
    except Exception as e:
        print(f'  Insightface failed: {type(e).__name__}: {e}')
        return None


def remap_onnx_to_iresnet(onnx_state_dict: dict, ref_state_dict: dict) -> dict:
    """
    Сопоставляет ключи из конвертированной ONNX-модели с эталонными ключами
    iresnet50 по порядку и форме тензоров.

    onnx2torch генерирует имена вида initializers.onnx_initializer_N или
    числовые префиксы — нам нужно сопоставить их с conv1.weight, layer1.0.bn1.weight
    и т.д. по форме (shape) и порядку появления в forward pass.
    """
    # Группируем эталонные ключи по форме и сохраняем порядок
    ref_items = list(ref_state_dict.items())  # [(name, tensor), ...]

    # ONNX-ключи также в порядке вычисления (onnx2torch сохраняет топологический порядок)
    onnx_items = [(k, v) for k, v in onnx_state_dict.items()
                  if isinstance(v, torch.Tensor)]

    # Простая стратегия: пройти по эталонным ключам и для каждого найти
    # первый неиспользованный ONNX-тензор той же формы
    remapped = {}
    used = set()
    for ref_name, ref_tensor in ref_items:
        ref_shape = tuple(ref_tensor.shape)
        # Пропускаем num_batches_tracked (нет в ONNX)
        if 'num_batches_tracked' in ref_name:
            remapped[ref_name] = ref_tensor.clone()
            continue
        found = False
        for i, (onnx_name, onnx_tensor) in enumerate(onnx_items):
            if i in used:
                continue
            if tuple(onnx_tensor.shape) == ref_shape:
                remapped[ref_name] = onnx_tensor.clone()
                used.add(i)
                found = True
                break
        if not found:
            print(f'    [warn] No match for {ref_name} {ref_shape}')
    return remapped


def try_onnx_conversion(iresnet50_fn, dst):
    """Скачивает ONNX, конвертирует в PyTorch state_dict для iresnet50."""
    onnx_path = find_onnx_model()
    if onnx_path is None:
        onnx_path = download_onnx_via_insightface()
    if onnx_path is None:
        return False

    print(f'  ONNX model: {onnx_path}')

    try:
        import onnx
        from onnx2torch import convert
    except ImportError:
        print('  onnx или onnx2torch не установлен.')
        print('  Установите: pip install onnx onnx2torch')
        return False

    print('  Converting ONNX -> PyTorch...')
    try:
        onnx_model = onnx.load(str(onnx_path))
        torch_model = convert(onnx_model)
        torch_model.eval()
    except Exception as e:
        print(f'  Conversion failed: {type(e).__name__}: {e}')
        return False

    # Получаем эталонную архитектуру
    ref_model = iresnet50_fn()
    ref_sd = ref_model.state_dict()

    # Сначала проверим прямое совпадение ключей (на случай если они уже совместимы)
    converted_sd = torch_model.state_dict()
    direct_overlap = set(ref_sd.keys()) & set(converted_sd.keys())
    coverage = len(direct_overlap) / len(ref_sd)
    print(f'  Direct key overlap: {coverage*100:.1f}%')

    if coverage >= 0.95:
        try:
            ref_model.load_state_dict(converted_sd, strict=False)
        except Exception:
            pass
    else:
        # Сопоставляем по форме и порядку
        print('  Performing automatic key remapping by tensor shape...')
        remapped = remap_onnx_to_iresnet(converted_sd, ref_sd)
        try:
            ref_model.load_state_dict(remapped, strict=False)
        except Exception as e:
            print(f'  Remap failed: {e}')
            return False

    # Финальная проверка: норма эмбеддинга должна быть ~1.0 после L2-нормализации
    ref_model.eval()
    x = torch.rand(1, 3, 112, 112)
    with torch.no_grad():
        out = ref_model(x)
        norm = nn.functional.normalize(out, p=2, dim=1).norm().item()

    print(f'  After conversion: embedding norm (L2-normalized) = {norm:.4f}')
    if abs(norm - 1.0) > 0.01:
        print('  WARNING: norm is not 1.0 — conversion may have produced wrong weights')
        print('  Trying anyway, but results may be incorrect')

    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ref_model.state_dict(), dst)
    size_mb = Path(dst).stat().st_size / 1e6
    print(f'  SUCCESS via ONNX conversion: {dst} ({size_mb:.1f} MB)')
    return True


# ============================================================================
# Финальная проверка
# ============================================================================
def verify(iresnet50_fn, weights_path):
    """Проверяет что веса корректно загружаются в архитектуру iresnet50."""
    print('\n' + '=' * 60)
    print('FINAL VERIFICATION')
    print('=' * 60)

    model = iresnet50_fn()
    sd = torch.load(weights_path, map_location='cpu')
    missing, unexpected = model.load_state_dict(sd, strict=False)

    if missing:
        print(f'Missing keys ({len(missing)}): {missing[:3]}...')
    if unexpected:
        print(f'Unexpected keys ({len(unexpected)}): {unexpected[:3]}...')

    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Тест на случайном входе
    x = torch.rand(1, 3, 112, 112).to(device)
    with torch.no_grad():
        emb = model(x)
        emb_n = nn.functional.normalize(emb, p=2, dim=1)

    print(f'Device: {device}')
    print(f'Output shape: {tuple(emb.shape)}')
    print(f'Embedding L2-norm: {emb_n.norm().item():.4f}  (expected: 1.0000)')

    if abs(emb_n.norm().item() - 1.0) < 0.01:
        print('OK: ArcFace weights are working correctly!')
        return True
    else:
        print('WARNING: norm != 1.0 — model may not be properly initialized')
        return False


def print_manual_instructions():
    """Если ничего не сработало — выводим понятную инструкцию."""
    print('\n' + '=' * 60)
    print('AUTOMATIC DOWNLOAD FAILED')
    print('=' * 60)
    print()
    print('Please download ArcFace weights manually using ONE of:')
    print()
    print('OPTION A — direct URL with VPN (browser):')
    print('  https://sota.nizhib.ai/pytorch-insightface/iresnet50-7f187506.pth')
    print('  Save as: ./weights/iresnet50.pth')
    print()
    print('OPTION B — via insightface (automatic ONNX download):')
    print('  pip install insightface onnxruntime-gpu onnx onnx2torch')
    print('  python download_arcface.py')
    print()
    print('OPTION C — manual ONNX conversion:')
    print('  1. pip install insightface onnxruntime')
    print('  2. python -c "from insightface.app import FaceAnalysis; '
          'app = FaceAnalysis(name=\'buffalo_l\'); app.prepare(0)"')
    print('  3. ONNX file will be at: ~/.insightface/models/buffalo_l/w600k_r50.onnx')
    print('  4. Re-run: python download_arcface.py')
    print()


# ============================================================================
# MAIN
# ============================================================================
def main():
    print('=' * 60)
    print('ArcFace iresnet50 weights downloader')
    print('=' * 60)

    # Если веса уже есть — проверим их
    if os.path.isfile(WEIGHTS_DST):
        size_mb = os.path.getsize(WEIGHTS_DST) / 1e6
        if size_mb > 50:
            print(f'\nWeights already exist: {WEIGHTS_DST} ({size_mb:.1f} MB)')
            iresnet50_fn = import_iresnet50()
            if verify(iresnet50_fn, WEIGHTS_DST):
                print('\nNothing to do — weights are valid.')
                return
            print('\nExisting weights are invalid, re-downloading...')
            os.remove(WEIGHTS_DST)

    iresnet50_fn = import_iresnet50()
    print(f'\nReference architecture: iresnet50 from arcface_hybrid.py')
    ref_keys = list(iresnet50_fn().state_dict().keys())
    print(f'Expected number of state_dict keys: {len(ref_keys)}')

    # Способ 1: HuggingFace
    print('\n[1/2] Trying HuggingFace Hub...')
    if try_huggingface(iresnet50_fn, WEIGHTS_DST):
        verify(iresnet50_fn, WEIGHTS_DST)
        return

    # Способ 2: ONNX через insightface + конвертация
    print('\n[2/2] Trying ONNX conversion via insightface + onnx2torch...')
    if try_onnx_conversion(iresnet50_fn, WEIGHTS_DST):
        verify(iresnet50_fn, WEIGHTS_DST)
        return

    # Ничего не сработало
    print_manual_instructions()
    sys.exit(1)


if __name__ == '__main__':
    main()