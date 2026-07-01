"""
Диагностика и исправление NaN в bn1.
Запуск: python fix_weights.py
"""
import torch
import torch.nn as nn
import importlib.util

spec = importlib.util.spec_from_file_location('ah', 'arcface_hybrid.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

sd = torch.load('./weights/iresnet50.pth', map_location='cpu')

# ============================================================
# Диагностика bn1
# ============================================================
print('=== bn1 diagnostics ===')
print(f'bn1.weight:       {sd["bn1.weight"][:5]}')
print(f'bn1.bias:         {sd["bn1.bias"][:5]}')
print(f'bn1.running_mean: {sd["bn1.running_mean"][:5]}')
print(f'bn1.running_var:  {sd["bn1.running_var"][:5]}')
print(f'bn1.num_batches_tracked: {sd["bn1.num_batches_tracked"]}')

print(f'\nbn1.running_var min: {sd["bn1.running_var"].min().item():.6f}')
print(f'bn1.running_var max: {sd["bn1.running_var"].max().item():.6f}')
print(f'Negative values in running_var: {(sd["bn1.running_var"] < 0).sum().item()}')
print(f'Zero values in running_var:     {(sd["bn1.running_var"] == 0).sum().item()}')

# Проверим все BN-слои на проблемные running_var
print('\n=== All BatchNorm running_var issues ===')
problematic = []
for k, v in sd.items():
    if 'running_var' in k:
        n_bad = (v <= 0).sum().item()
        if n_bad > 0:
            problematic.append((k, n_bad, v.min().item()))
            print(f'  {k}: {n_bad} bad values, min={v.min().item():.6f}')

if not problematic:
    print('  No problematic running_var found')

# ============================================================
# Исправление: сбросить running stats у проблемных BN слоёв
# ============================================================
print('\n=== Fixing running stats ===')
fixed = 0
for k, v in sd.items():
    if 'running_var' in k:
        bad_mask = v <= 0
        if bad_mask.any():
            sd[k][bad_mask] = 1.0  # сброс в 1.0 (нейтральное значение)
            # Соответствующий running_mean тоже сбрасываем
            mean_key = k.replace('running_var', 'running_mean')
            if mean_key in sd:
                sd[mean_key][bad_mask] = 0.0
            fixed += bad_mask.sum().item()
            print(f'  Fixed {bad_mask.sum().item()} values in {k}')

if fixed == 0:
    print('  Nothing to fix in running_var')
    # Значит проблема в другом — проверим num_batches_tracked
    print('\n=== Checking num_batches_tracked ===')
    for k, v in sd.items():
        if 'num_batches_tracked' in k and v.item() == 0:
            print(f'  Zero batches tracked: {k}')

# ============================================================
# Принудительный сброс running stats bn1 (если всё ещё NaN)
# ============================================================
print('\n=== Force-resetting bn1 running stats ===')
sd['bn1.running_mean'] = torch.zeros(64)
sd['bn1.running_var']  = torch.ones(64)
sd['bn1.num_batches_tracked'] = torch.tensor(1)

# Сбросим running stats всех BN слоёв с нулевым num_batches_tracked
for k in list(sd.keys()):
    if 'num_batches_tracked' in k:
        if sd[k].item() == 0:
            var_key  = k.replace('num_batches_tracked', 'running_var')
            mean_key = k.replace('num_batches_tracked', 'running_mean')
            if var_key in sd:
                sd[var_key]  = torch.ones_like(sd[var_key])
            if mean_key in sd:
                sd[mean_key] = torch.zeros_like(sd[mean_key])
            sd[k] = torch.tensor(1)
            print(f'  Reset: {k.replace(".num_batches_tracked", "")}')

# ============================================================
# Проверяем после исправления
# ============================================================
print('\n=== Verification after fix ===')
model = mod.iresnet50()
model.load_state_dict(sd, strict=True)
model.eval()

x = torch.rand(1, 3, 112, 112)
with torch.no_grad():
    out = model(x)
    out_norm = nn.functional.normalize(out, p=2, dim=1)
    norm_val = out_norm.norm().item()

print(f'Output has NaN: {torch.isnan(out).any().item()}')
print(f'L2-norm: {norm_val:.4f}  (expected: 1.0000)')

if abs(norm_val - 1.0) < 0.01:
    # Сохраняем исправленные веса
    torch.save(sd, './weights/iresnet50.pth')
    print('\nSUCCESS: Fixed weights saved to ./weights/iresnet50.pth')
    print('You can now run arcface_hybrid.py normally.')
else:
    print(f'\nNaN still present — deeper issue.')
    print('Running layer-by-layer check after fix:')
    with torch.no_grad():
        h = model.conv1(x)
        print(f'  conv1:    NaN={torch.isnan(h).any().item()}')
        h = model.bn1(h)
        print(f'  bn1:      NaN={torch.isnan(h).any().item()}, '
              f'range=[{h.min():.3f}, {h.max():.3f}]')
        h = model.prelu(h)
        print(f'  prelu:    NaN={torch.isnan(h).any().item()}')
        h = model.layer1(h)
        print(f'  layer1:   NaN={torch.isnan(h).any().item()}')
        h = model.layer2(h)
        print(f'  layer2:   NaN={torch.isnan(h).any().item()}')