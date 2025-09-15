import abc

import pytest

from fl_manager.core.meta_registry import MetaRegistry
from fl_manager.core.schemas.registry_item import RegistryItem
from fl_manager.core.utils.registry import InstanceRegistry, ClassRegistry


class BaseItem(metaclass=abc.ABCMeta):
    def __init__(self, item_name: str):
        self._name = item_name

    @abc.abstractmethod
    def run(self):
        raise NotImplementedError()


class GreetingItem(BaseItem):
    def run(self):
        return f'Hello {self._name}'


@pytest.fixture(scope='module')
def meta_registry():
    _base_item_registry = ClassRegistry[BaseItem](BaseItem)
    _base_item_registry.register(name='greeting')(GreetingItem)
    m = InstanceRegistry[ClassRegistry](ClassRegistry, allow_replacements=False)
    m.register(name='base')(_base_item_registry)
    return m


def test_registry_item(mocker, meta_registry):
    mocker.patch.object(MetaRegistry, 'get', new=meta_registry.get)
    _item = RegistryItem(
        registry_id='base',
        name='greeting',
        keyword_arguments={'item_name': 'FL Manager'},
    )
    _instance = _item.instance
    assert isinstance(_instance, GreetingItem)
    assert _instance.run() == 'Hello FL Manager'


def test_invalid_registry_item(mocker, meta_registry):
    mocker.patch.object(MetaRegistry, 'get', new=meta_registry.get)
    with pytest.raises(ValueError) as ex:
        _item = RegistryItem(
            registry_id='base',
            name='walk',
            keyword_arguments={'item_name': 'FL Manager'},
        )
    assert 'not found in registry' in str(ex.value)


def test_missing_arguments(mocker, meta_registry):
    mocker.patch.object(MetaRegistry, 'get', new=meta_registry.get)
    with pytest.raises(ValueError) as ex:
        _item = RegistryItem(registry_id='base', name='greeting')
    assert 'item_name' in str(ex.value)
