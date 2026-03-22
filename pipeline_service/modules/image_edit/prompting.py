from abc import ABC, abstractmethod
from typing import Annotated, Any, Optional, List
from pydantic import BaseModel, Field, BeforeValidator

def ensure_string_tuple(s: Any) -> List[str] | Any:
    if isinstance(s, str):
        return [s,]
    return list(s)

TextPrompts = Annotated[List[str], BeforeValidator(ensure_string_tuple)]

class Prompting(ABC):
    @abstractmethod
    def model_dump(self, *args, **kwargs):
        pass

    @abstractmethod
    def __len__(self):
        pass

class TextPrompting(BaseModel, Prompting):
    prompt: TextPrompts = Field(alias="positive")
    negative_prompt: Optional[TextPrompts] = Field(default=None, alias="negative")

    def __len__(self):
        return len(self.prompt)