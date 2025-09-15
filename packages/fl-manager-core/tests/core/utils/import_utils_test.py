import sys
import tempfile
from pathlib import Path

import pytest

from fl_manager.core.utils.import_utils import ImportUtils


@pytest.fixture(scope='module')
def dummy_package() -> str:
    """
    dummy_fl_manager
    |-- core [w/ __init__.py]
    |   `-- components [w/ __init__.py]
    |       `-- dummy [w/ __init__.py]
    `-- components
        `-- dummy
            `-- extra [w/ __init__.py]
    """
    tmpdir = tempfile.TemporaryDirectory()
    tmpdir_path = Path(tmpdir.name)
    package_name = 'dummy_fl_manager'
    package_dir = tmpdir_path / package_name
    core_dir = package_dir / 'core'
    core_components_dir = core_dir / 'components'
    dummy_component_dir = core_components_dir / 'dummy'
    extra_dummy_component_dir = package_dir / 'components' / 'dummy' / 'extra'

    for e in [
        core_dir,
        core_components_dir,
        dummy_component_dir,
        extra_dummy_component_dir,
    ]:
        e.mkdir(parents=True, exist_ok=True)
        (e / '__init__.py').touch()

    sys.path.insert(0, tmpdir.name)
    yield package_name
    sys.path.remove(tmpdir.name)
    tmpdir.cleanup()


def test_dynamic_import(mocker, dummy_package):
    mocker.patch.object(ImportUtils, '_PKG_NAME', dummy_package)
    mock_iter_import_pkg = mocker.patch.object(ImportUtils, 'iter_import_pkg')
    ImportUtils.dynamic_registry_item_import('dummy', 'components')
    assert mock_iter_import_pkg.call_count == 2
    assert mock_iter_import_pkg.call_args_list[0][0] == (
        '.dummy',
        f'{dummy_package}.core.components',
    )
    assert mock_iter_import_pkg.call_args_list[1][0] == (
        '.dummy',
        f'{dummy_package}.components',
    )


def test_iter_import_pkg(mocker, dummy_package):
    mocker.patch.object(ImportUtils, '_PKG_NAME', dummy_package)
    r = ImportUtils.iter_import_pkg('.dummy', f'{dummy_package}.components')
    assert [e.__name__ for e in r] == [
        'dummy_fl_manager.components.dummy',
        'dummy_fl_manager.components.dummy.extra',
    ]


def test_missing_package_relative_import():
    with pytest.raises(TypeError):
        ImportUtils.iter_import_pkg('.dummy')
