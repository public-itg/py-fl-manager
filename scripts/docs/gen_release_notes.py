from pathlib import Path

import mkdocs_gen_files

filename = 'release-notes.md'

_contents = Path('CHANGELOG.md').read_text()
_start = _contents.find('##')
_contents = '---\nhide:\n - navigation\n---\n# Release Notes\n' + _contents[_start:]

with mkdocs_gen_files.open(filename, 'w') as f:
    f.write(_contents)

mkdocs_gen_files.set_edit_path(filename, 'scripts/docs/gen_release_notes.py')
