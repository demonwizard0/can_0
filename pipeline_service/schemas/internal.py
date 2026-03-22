from typing import Annotated, TypeAlias, TypeVar

from pydantic import GetCoreSchemaHandler
from pydantic_core import PydanticOmit, core_schema

T = TypeVar("T")

class InternalAnnotation:
    def __get_pydantic_core_schema__(self, source_type, handler: GetCoreSchemaHandler):
        # For plain classes (your “arbitrary types”), avoid handler(...) entirely
        # so we don't require arbitrary_types_allowed.
        if isinstance(source_type, type):
            schema = core_schema.is_instance_schema(source_type)
        else:
            # For non-plain typing constructs, just let pydantic handle it normally
            # (optional; you can force type-only behavior by returning any_schema()).
            schema = handler(source_type)

        # Always omit this field during serialization.
        schema["serialization"] = core_schema.plain_serializer_function_ser_schema(
            lambda _: PydanticOmit,
            when_used="always",
        )
        return schema

Internal: TypeAlias = Annotated[T, InternalAnnotation()]
