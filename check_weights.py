"""
Диагностика весов ArcFace.
Запуск: python check_weights.py
"""
import torch
import torch.nn as nn
import importlib.util

# Загружаем архитектуру из arcface_hybrid.py
spec = importlib.util.spec_from_file_location('ah', 'arcface_hybrid.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Загружаем веса
print('Loading weights...')
sd = torch.load('./weights/iresnet50.pth', map_location='cpu')

# Проверяем совпадение ключей
model = mod.iresnet50()
ref_keys = set(model.state_dict().keys())
sd_keys  = set(sd.keys())

print(f'Keys in model:      {len(ref_keys)}')
print(f'Keys in weights:    {len(sd_keys)}')
print(f'Missing in weights: {ref_keys - sd_keys}')
print(f'Extra in weights:   {sd_keys - ref_keys}')

# Загружаем веса в модель
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f'Missing keys:    {missing}')
print(f'Unexpected keys: {unexpected}')

# Ищем NaN прямо в весах
print('\nChecking for NaN in weight tensors...')
nan_keys = [k for k, v in sd.items() if torch.isnan(v).any()]
print(f'Keys with NaN: {nan_keys[:10] if nan_keys else "none — weights are clean"}')

inf_keys = [k for k, v in sd.items() if torch.isinf(v).any()]
print(f'Keys with Inf: {inf_keys[:10] if inf_keys else "none"}')

# Тест forward pass
print('\nRunning forward pass...')
model.eval()
x = torch.rand(1, 3, 112, 112)
with torch.no_grad():
    out = model(x)

print(f'Output shape:    {tuple(out.shape)}')
print(f'Output has NaN:  {torch.isnan(out).any().item()}')
print(f'Output has Inf:  {torch.isinf(out).any().item()}')
print(f'Output range:    [{out.min().item():.4f}, {out.max().item():.4f}]')

# Тест с L2-нормализацией (как в ArcFaceEncoder.forward)
out_norm = nn.functional.normalize(out, p=2, dim=1)
norm_val = out_norm.norm().item()
print(f'L2-norm after normalize: {norm_val:.4f}  (expected: 1.0000)')

# Промежуточная диагностика — ищем где именно возникает NaN
print('\nLayer-by-layer NaN check...')
model2 = mod.iresnet50()
model2.load_state_dict(sd, strict=False)
model2.eval()

x2 = torch.rand(1, 3, 112, 112)
with torch.no_grad():
    h = model2.conv1(x2)
    print(f'  after conv1:   NaN={torch.isnan(h).any().item()}, '
          f'range=[{h.min():.3f}, {h.max():.3f}]')
    h = model2.bn1(h)
    print(f'  after bn1:     NaN={torch.isnan(h).any().item()}, '
          f'range=[{h.min():.3f}, {h.max():.3f}]')
    h = model2.prelu(h)
    print(f'  after prelu:   NaN={torch.isnan(h).any().item()}')
    h = model2.layer1(h)
    print(f'  after layer1:  NaN={torch.isnan(h).any().item()}')
    h = model2.layer2(h)
    print(f'  after layer2:  NaN={torch.isnan(h).any().item()}')
    h = model2.layer3(h)
    print(f'  after layer3:  NaN={torch.isnan(h).any().item()}')
    h = model2.layer4(h)
    print(f'  after layer4:  NaN={torch.isnan(h).any().item()}')
    h = model2.bn2(h)
    print(f'  after bn2:     NaN={torch.isnan(h).any().item()}')
    h = torch.flatten(h, 1)
    h = model2.fc(h)
    print(f'  after fc:      NaN={torch.isnan(h).any().item()}, '
          f'range=[{h.min():.3f}, {h.max():.3f}]')
    h = model2.features(h)
    print(f'  after features (BN1d): NaN={torch.isnan(h).any().item()}, '
          f'range=[{h.min():.3f}, {h.max():.3f}]')

print('\nDone.')