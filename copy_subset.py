import shutil
from pathlib import Path

src = Path('./celeba/img_align_celeba')
dst = Path('./celeba/subset_50k')
dst.mkdir(exist_ok=True)

files = sorted(src.glob('*.jpg'))[:50000]
for i, f in enumerate(files):
    shutil.copy(f, dst / f.name)
    if i % 5000 == 0:
        print(f'Copied {i}/{len(files)}')
print('Done')