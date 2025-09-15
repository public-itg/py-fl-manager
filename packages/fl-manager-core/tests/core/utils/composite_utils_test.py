import abc
from typing import List

import pytest

from fl_manager.core.utils.composite_utils import CompositeUtils


class BaseElement(metaclass=abc.ABCMeta):
    @property
    def is_composite(self) -> bool:
        return False

    @abc.abstractmethod
    def run(self, acc: int):
        raise NotImplementedError()


class Leaf(BaseElement):
    def __init__(self, value: int):
        self._value = value

    def run(self, acc: int):
        return self._value + acc


class Composite(BaseElement):
    def __init__(self):
        self._children: List[BaseElement] = []

    @property
    def is_composite(self) -> bool:
        return True

    def add(self, element: BaseElement):
        self._children.append(element)

    def run(self, acc: int):
        _acc = acc
        for child in self._children:
            _acc = child.run(_acc)
        return _acc


@pytest.fixture(scope='module')
def composite():
    return Composite()


@pytest.fixture(scope='module')
def leaf_a():
    return Leaf(2)


@pytest.fixture(scope='module')
def leaf_b():
    return Leaf(3)


def test_composite(composite, leaf_a, leaf_b):
    cmp = CompositeUtils.leafs_to_composite(composite, [leaf_a, leaf_b])
    assert len(cmp._children) == 2
    assert cmp.run(2) == leaf_b.run(leaf_a.run(2))


def test_invalid_composite(leaf_a, leaf_b):
    class _A:
        pass

    with pytest.raises(AssertionError):
        CompositeUtils.leafs_to_composite(leaf_a, [leaf_b])
        CompositeUtils.leafs_to_composite(_A, [leaf_a, leaf_b])
