"""Per-ToolBox deployment context — injected at invoke time, never in LLM schema."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

T = TypeVar("T")

INJECTED_CONTEXT_PARAM = "_tool_context"


class ToolContextError(KeyError):
    """Required context slot is missing or has the wrong type."""


@dataclass
class ToolContextBundle:
    """Keyed deployment bindings owned by one :class:`ToolBox` instance."""

    _slots: dict[str, object] = field(default_factory=dict)
    _close_hooks: dict[str, Callable[[], Awaitable[None]]] = field(default_factory=dict)

    def bind(
        self,
        key: str,
        value: object,
        *,
        aclose: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._slots[key] = value
        if aclose is not None:
            self._close_hooks[key] = aclose

    def has(self, key: str) -> bool:
        return key in self._slots

    def require(self, key: str, typ: type[T]) -> T:
        value = self._slots.get(key)
        if value is None or not isinstance(value, typ):
            raise ToolContextError(key)
        return value

    def view(self, keys: tuple[str, ...]) -> ToolContextView:
        return ToolContextView(self, keys)

    async def aclose(self) -> None:
        for hook in self._close_hooks.values():
            await hook()
        self._slots.clear()
        self._close_hooks.clear()


@dataclass(frozen=True)
class ToolContextView:
    """Slice of a bundle visible to one tool (only its declared ``context_keys``)."""

    _bundle: ToolContextBundle
    _keys: tuple[str, ...]

    def require(self, key: str, typ: type[T]) -> T:
        if key not in self._keys:
            raise ToolContextError(key)
        return self._bundle.require(key, typ)
