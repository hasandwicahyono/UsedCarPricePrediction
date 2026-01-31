from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Generic, Optional, TypeVar

T = TypeVar("T")


def normalize_key(name: str) -> str:
    return name.strip().lower()


@dataclass
class Registry(Generic[T]):
    normalize: Callable[[str], str] = normalize_key
    _items: Dict[str, T] = field(default_factory=dict)

    def register(self, name: str, item: T) -> None:
        key = self.normalize(name) if self.normalize else name
        self._items[key] = item

    def register_many(self, items: Dict[str, T]) -> None:
        for name, item in items.items():
            self.register(name, item)

    def get(self, name: str) -> Optional[T]:
        key = self.normalize(name) if self.normalize else name
        return self._items.get(key)

    def items(self) -> Dict[str, T]:
        return dict(self._items)

    def keys(self):
        return self._items.keys()

    def __contains__(self, name: str) -> bool:
        return self.get(name) is not None

    def __getitem__(self, name: str) -> T:
        item = self.get(name)
        if item is None:
            raise KeyError(name)
        return item


METRIC_REGISTRY: Registry[type] = Registry()
MODEL_REGISTRY: Registry[type] = Registry(normalize=lambda s: s)
RESIDUAL_REGISTRY: Registry[type] = Registry(normalize=lambda s: s)
SHAP_EXPLAINER_REGISTRY: Registry[type] = Registry()
