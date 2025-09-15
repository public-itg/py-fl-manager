from typing import List, Any


class CompositeUtils:
    @staticmethod
    def leafs_to_composite(composite: Any, leafs: List[Any]) -> Any:
        assert all(
            [
                getattr(composite, _attr, None) is not None
                for _attr in ['is_composite', 'add']
            ]
        ), 'invalid class.'
        assert composite.is_composite, 'not composite object.'
        for leaf in leafs:
            composite.add(leaf)
        return composite
