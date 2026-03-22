from pydantic import AfterValidator, BeforeValidator
from typing import Annotated, Any, Literal, TypeAlias, Union
from pydantic_tensor.types import Int, Float, BFloat
from schemas.tensors import TorchTensor, Bool


AnyTensor: TypeAlias = TorchTensor[Any, Any]

# Integer Tensors
IntegerTensor: TypeAlias = TorchTensor[Any, Int]
IntTensor: TypeAlias = TorchTensor[Any, Literal["int32"]]
LongTensor: TypeAlias = TorchTensor[Any, Literal["int64"]]

# Floating Point Tensors
FloatingTensor: TypeAlias = TorchTensor[Any, Union[Float, BFloat]]
HalfTensor: TypeAlias = TorchTensor[Any, Literal["float16"]]
BFloatTensor: TypeAlias = TorchTensor[Any, Literal["bfloat16"]]
FloatTensor: TypeAlias = TorchTensor[Any, Literal["float32"]]
DoubleTensor: TypeAlias = TorchTensor[Any, Literal["float64"]]
BoolTensor: TypeAlias = Annotated[TorchTensor[Any, Any], BeforeValidator(lambda t: t.byte()), AfterValidator(lambda t: t.bool())]
