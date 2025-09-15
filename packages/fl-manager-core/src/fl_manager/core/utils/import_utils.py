import logging
from importlib import import_module
from pkgutil import iter_modules
from types import ModuleType
from typing import Optional, List

logger = logging.getLogger(__name__)


class ImportUtils:
    _PKG_NAME = 'fl_manager'

    @classmethod
    def dynamic_registry_item_import(
        cls, name: str, subpackage: Optional[str] = None
    ) -> None:
        """
        From the base package specified as the class attribute, tries to import given module from base package
        and main subpackages inside it, appending the given subpackage if any.

        This enables the ability to find extensions to the main package, (e.g. components inside another integration).

        Args:
            name (str): Name of the module to import.
            subpackage (Optional[str]): Name of the subpackage where the module lives.

        Returns:
            None
        """
        _pkg_modules: list = []
        _main_module = cls._import_module_or_none(cls._PKG_NAME)
        if _main_module is not None:
            _pkg_modules = [
                e.name
                for e in iter_modules(_main_module.__path__, prefix=f'{cls._PKG_NAME}.')
            ]
            _pkg_modules.append(cls._PKG_NAME)
        _subpackage = subpackage if subpackage is not None else ''
        for _module in _pkg_modules:
            cls.iter_import_pkg(f'.{name}', '.'.join([_module, _subpackage]).strip('.'))

    @staticmethod
    def iter_import_pkg(
        name: str, package: Optional[str] = None
    ) -> Optional[List[ModuleType]]:
        """
        Imports specified module and submodules (if package) inside it (one level loop).

        Args:
            name (str): name of the module to import.
            package (Optional[str]): name of the package for relative module.

        Returns:
            Optional[List[ModuleType]]: A list of imported modules.
        """
        _module = ImportUtils._import_module_or_none(name, package)
        if _module is None:
            return
        _package = ((package or '') + name).strip('.')
        _imported = [_module]
        _imported.extend(
            [
                import_module(f'.{module_name}', package=_package)
                for (path, module_name, is_pkg) in iter_modules(_module.__path__)
                if is_pkg
            ]
        )
        return _imported

    @staticmethod
    def _import_module_or_none(
        name: str, package: Optional[str] = None
    ) -> Optional[ModuleType]:
        if ImportUtils._is_relative(name) and package is None:
            raise TypeError(f'Missing package for relative import: {name}')
        try:
            _module = import_module(name, package)
            return _module
        except ModuleNotFoundError:
            logger.debug(f'Not found module (name={name}, package={package})')

    @staticmethod
    def _is_relative(name: str) -> bool:
        return name.startswith('.')
