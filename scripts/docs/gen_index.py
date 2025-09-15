from pathlib import Path

import mkdocs_gen_files

filename = 'index.md'
_contents = Path('README.md').read_text()
_contents = _contents.replace('docs/docs/img/logo.jpg', 'img/logo.jpg')
_contents = _contents.replace(
    '[examples](examples)', '[examples](learn/examples/index.md)'
)
_contents = _contents.replace(
    '[`.pre-commit-config.yaml`](.pre-commit-config.yaml)', '`.pre-commit-config.yaml`'
)

with mkdocs_gen_files.open(filename, 'w') as f:
    f.write(_contents)

mkdocs_gen_files.set_edit_path(filename, 'scripts/docs/gen_index.py')
