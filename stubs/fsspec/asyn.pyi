from asyncio import AbstractEventLoop
from collections.abc import Awaitable, Callable
from typing import ParamSpec, TypeVar

P = ParamSpec('P')
T = TypeVar('T')

def sync(loop: AbstractEventLoop, func: Callable[P, Awaitable[T]], *args: P.args, **kwargs: P.kwargs) -> T: ...
