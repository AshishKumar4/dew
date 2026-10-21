from asyncio import AbstractEventLoop
from collections.abc import Awaitable, Callable
from typing import ParamSpec, TypeVar

from .spec import AbstractFileSystem

P = ParamSpec('P')
T = TypeVar('T')

def get_loop() -> AbstractEventLoop: ...
def sync(loop: AbstractEventLoop, func: Callable[P, Awaitable[T]], *args: P.args, **kwargs: P.kwargs) -> T: ...

class AsyncFileSystem(AbstractFileSystem):
    loop: AbstractEventLoop | None
    asynchronous: bool
