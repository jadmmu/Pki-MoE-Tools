from abc import ABC, abstractmethod
from typing import Any, Iterable


class TraceSource(ABC):
    condition: str
    backend_name: str

    @abstractmethod
    def capture(self, prompt: str, **kwargs: Any) -> Iterable[dict[str, Any]]:
        raise NotImplementedError
