from copy import deepcopy
from typing import Any, Optional, ClassVar, Type, TypeVar
from pydantic import BaseModel, create_model

OverridableModelT = TypeVar("OverridableModelT", bound="OverridableModel")

class OverridableModel(BaseModel):
    """
    A base model that can generate an 'Optional' counterpart for overrides
    and merge them using built-in Pydantic methods.
    """
    Overrides: ClassVar[Type[BaseModel]]

    def overrided(self, overrides_instance: Optional[BaseModel]) -> OverridableModelT:
        """
        Merges a partial Pydantic override model into the current instance's fields.
        Returns a new instance of the model with applied overrides.
        """
        if not overrides_instance:
            return self

        overrides_dict = overrides_instance.model_dump(exclude_none=True)
        
        current_data = self.model_dump()
        
        current_data.update(overrides_dict)
        
        return type(self)(**current_data)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        super().__pydantic_init_subclass__(**kwargs)
        cls.Overrides = cls._create_optional_counterpart()

    @classmethod
    def _create_optional_counterpart(cls) -> Type[BaseModel]:
        """
        Dynamically creates the optional version of this class, preserving FieldInfo.
        """
        fields: dict[str, Any] = {}
        for name, field_info in cls.model_fields.items():
            new_field_info = deepcopy(field_info)
            new_field_info.default = None
            new_field_info.annotation = Optional[field_info.annotation] 

            fields[name] = (new_field_info.annotation, new_field_info)
        
        DynamicOverrides = create_model(f"{cls.__name__}Overrides", **fields)
        return DynamicOverrides