import pytest

from fl_manager.core.utils.registry import ClassRegistry, InstanceRegistry


class Animal:
    pass


class Dog(Animal):
    pass


class Cat(Animal):
    pass


class Plant:
    pass


class Rose(Plant):
    pass


@pytest.fixture(scope='module')
def dog():
    return Dog()


@pytest.fixture(scope='module')
def cat():
    return Cat()


@pytest.fixture(scope='module')
def rose():
    return Rose()


@pytest.fixture
def animal_class_registry():
    return ClassRegistry[Animal](Animal, allow_replacements=True)


@pytest.fixture
def animal_class_registry_without_replacements():
    return ClassRegistry[Animal](Animal, allow_replacements=False)


@pytest.fixture
def animal_instance_registry():
    return InstanceRegistry[Animal](Animal, allow_replacements=True)


def test_class_registry(animal_class_registry_without_replacements):
    animal_class_registry_without_replacements.register('dog')(Dog)
    animal_class_registry_without_replacements.register('cat')(Cat)
    with pytest.raises(RuntimeError):
        animal_class_registry_without_replacements.register('dog', replace=True)(Dog)
    assert len(animal_class_registry_without_replacements.list()) == 2
    assert all(
        [e in ['dog', 'cat'] for e in animal_class_registry_without_replacements.list()]
    )
    assert isinstance(animal_class_registry_without_replacements.create('dog'), Dog)


def test_class_registry_with_replacements(animal_class_registry):
    animal_class_registry.register('animal')(Dog)
    assert isinstance(animal_class_registry.create('animal'), Dog)
    animal_class_registry.register('animal', replace=True)(Cat)
    assert len(animal_class_registry.list()) == 1
    assert all([e in ['animal'] for e in animal_class_registry.list()])
    assert isinstance(animal_class_registry.create('animal'), Cat)


def test_class_registry_invalid_item(animal_class_registry, dog):
    with pytest.raises(ValueError):
        animal_class_registry.register('plant')(Rose)
    with pytest.raises(ValueError):
        animal_class_registry.register('animal')(dog)


def test_instance_registry(dog, cat):
    _animal_instance_registry = InstanceRegistry[Animal](
        Animal, allow_replacements=True
    )
    _animal_instance_registry.register('animal')(dog)
    assert _animal_instance_registry.get('animal') == dog
    _animal_instance_registry.register('animal', replace=True)(cat)
    assert _animal_instance_registry.get('animal') == cat


@pytest.mark.parametrize(
    'registry', ['animal_class_registry', 'animal_instance_registry']
)
def test_non_existent_item(registry, request):
    registry = request.getfixturevalue(registry)
    with pytest.raises(ValueError):
        registry.get('animal')


def test_class_registry_type(animal_instance_registry, rose):
    with pytest.raises(ValueError):
        animal_instance_registry.register('plant')(rose)
