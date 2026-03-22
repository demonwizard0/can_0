import torch
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler, SerializationInfo
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import SchemaValidator, core_schema
from pydantic_tensor.backend.torch import TorchInterface
from pydantic_tensor.delegate import NumpyDelegate
from pydantic_tensor.pydantic.dtype import build_dtype_schema
from pydantic_tensor.pydantic.shape import postprocess_shape_schema
from pydantic_tensor.types import JSONTensor
from pydantic_tensor.types import Shape_T
from pydantic_tensor.types import DTypes as P_DTypes

from pydantic_tensor.utils.type_annotation import default_any, extract_type_annotation


from typing import Any, Generic, Union, Literal, TypeVar

Bool = Literal["bool"]
DTypes = Union[P_DTypes, Bool]

DType_T = TypeVar("DType_T", bound=DTypes)


class TorchTensor(Generic[Shape_T, DType_T]):
    @staticmethod
    def __get_pydantic_core_schema__(
        source: type[Any], handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        shape_anno, dtype_anno = map(default_any, extract_type_annotation(source, TorchTensor))
        shape_schema = postprocess_shape_schema(handler(shape_anno))
        dtype_schema = build_dtype_schema(dtype_anno)

        shape_validator = SchemaValidator(shape_schema)
        dtype_validator = SchemaValidator(dtype_schema)

        def deserialize(x: Any) -> torch.Tensor:
            return NumpyDelegate.from_json_tensor(x, [TorchInterface]).deserialize(TorchInterface)

        def serialize_tensor(x: torch.Tensor, info: SerializationInfo) -> JSONTensor | torch.Tensor:
            if "json" in info.mode:
                return NumpyDelegate.from_tensor(x, [TorchInterface]).serialize()
            return x

        def validate_tensor(x: Any) -> torch.Tensor:
            # Allow wrapped pydantic_tensor.Tensor instances.
            if hasattr(x, "value"):
                x = getattr(x, "value")
            if not TorchInterface.is_tensor_type(x):
                msg = f'expected "torch.Tensor", got "{type(x).__module__}.{type(x).__name__}"'
                raise ValueError(msg)
            str_dtype = TorchInterface.dtype_to_str(TorchInterface.extract_dtype(x))
            shape_validator.validate_python(TorchInterface.extract_shape(x))
            dtype_validator.validate_python(str_dtype)
            return x

        json_schema = core_schema.no_info_after_validator_function(
            deserialize,
            core_schema.typed_dict_schema(
                {
                    "shape": core_schema.typed_dict_field(shape_schema),
                    "dtype": core_schema.typed_dict_field(dtype_schema),
                    "data": core_schema.typed_dict_field(core_schema.str_schema()),
                }
            ),
        )

        python_schema = core_schema.union_schema(
            [core_schema.no_info_plain_validator_function(validate_tensor), json_schema],
        )

        return core_schema.json_or_python_schema(
            python_schema=python_schema,
            json_schema=json_schema,
            serialization=core_schema.plain_serializer_function_ser_schema(serialize_tensor, info_arg=True),
        )

    @staticmethod
    def __get_pydantic_json_schema__(
        core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        json_schema = handler(core_schema)
        json_schema["properties"]["dtype"]["format"] = "base64"
        return json_schema