from pathlib import Path

import mkdocs_gen_files

for path in sorted(Path('notebooks').glob('*.ipynb')):
    with mkdocs_gen_files.open(f'examples/{path.name}', 'wb') as f:
        f.write(path.read_bytes())
